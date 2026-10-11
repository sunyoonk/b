"""Aspect-wise residual quantization over FACE's LLM-vocabulary codebook.

FACE produces A collaborative aspect representations z_a (the post-transformer
tensor called z_e in FACE.forward). Here every aspect is quantized
independently over depth, against the one mapped vocabulary codebook C that
FACE already owns:

    r_a^(1)   = z_a
    k_a^(h)   = argmin_j || r_a^(h) - C_j ||^2
    c_a^(h)   = C[k_a^(h)]
    r_a^(h+1) = r_a^(h) - c_a^(h)
    z_q,a     = sum_{h <= L_a} c_a^(h)

Depth 1 reuses FACE's own selection (greedy over aspects, a word chosen by one
aspect is masked for the later aspects of the same entity), so max_depth=1
reproduces FACE exactly. Depths >= 2 are plain nearest-neighbour by default;
the other reuse policies are opt-in.

No stopping rule lives here. max_depth is only a safety cap; the per-aspect
lengths L_a are an input to `reconstruct`, to be decided elsewhere later.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from FACE import FACE

logger = logging.getLogger(__name__)

# none   : depth >= 2 is plain nearest-neighbour over the whole codebook
# aspect : a token cannot repeat inside one aspect's sequence
# entity : a token cannot repeat anywhere in the entity (greedy in aspect order,
#          the depth-1 FACE rule extended over depth)
REUSE_POLICIES = ('none', 'aspect', 'entity')


def _sq_dist(r, codebook):
    """Same expansion FACE uses: r [M, D], codebook [V, D] -> [M, V]."""
    return torch.sum(r**2, dim=1, keepdim=True) + \
        torch.sum(codebook**2, dim=1) - 2 * \
        torch.matmul(r, codebook.t())


def depth_mask(lengths, max_depth):
    """lengths [N, A] (ints in [0, max_depth]) -> bool [N, A, max_depth]."""
    if lengths.min() < 0 or lengths.max() > max_depth:
        raise ValueError(f"lengths must lie in [0, {max_depth}]")
    steps = torch.arange(max_depth, device=lengths.device)
    return steps < lengths.unsqueeze(-1)


@torch.no_grad()
def duplicate_token_stats(token_indices, lengths=None):
    """Repeated-codeword statistics for token_indices [N, A, L].

    A token counts as a repeat when the same vocabulary index already occurred
    earlier. "Earlier" is by depth inside an aspect, and by (aspect, depth) in
    aspect-major order inside an entity. Only positions within `lengths` count.
    """
    N, A, L = token_indices.shape
    if lengths is None:
        valid = torch.ones_like(token_indices, dtype=torch.bool)
    else:
        valid = depth_mask(lengths, L)

    # within one aspect, over depth
    same = token_indices.unsqueeze(-1) == token_indices.unsqueeze(-2)          # N,A,L,L
    earlier = torch.tril(torch.ones(L, L, dtype=torch.bool, device=token_indices.device), -1)
    pair_ok = valid.unsqueeze(-1) & valid.unsqueeze(-2) & earlier
    aspect_repeat = (same & pair_ok).any(-1)                                     # N,A,L
    n_deep = valid[..., 1:].sum().item()

    # within one entity, over every (aspect, depth)
    flat = token_indices.reshape(N, A * L)
    flat_valid = valid.reshape(N, A * L)
    same_e = flat.unsqueeze(-1) == flat.unsqueeze(-2)
    earlier_e = torch.tril(torch.ones(A * L, A * L, dtype=torch.bool, device=flat.device), -1)
    entity_repeat = (same_e & flat_valid.unsqueeze(-1) & flat_valid.unsqueeze(-2) & earlier_e).any(-1)

    # across aspects at the same depth
    per_depth_cross = []
    for h in range(L):
        col, v = token_indices[:, :, h], valid[:, :, h]
        s = (col.unsqueeze(-1) == col.unsqueeze(-2)) & v.unsqueeze(-1) & v.unsqueeze(-2)
        s = s & ~torch.eye(A, dtype=torch.bool, device=col.device)
        per_depth_cross.append(s.any(-1)[v].float().mean().item() if v.any() else 0.0)

    n_valid = valid.sum().item()
    return {
        'n_tokens': n_valid,
        'within_aspect_repeat_rate': aspect_repeat.sum().item() / max(n_deep, 1),
        'aspects_with_repeat_frac': aspect_repeat.any(-1)[valid.any(-1)].float().mean().item()
        if valid.any() else 0.0,
        'within_entity_repeat_rate': entity_repeat.sum().item() / max(n_valid, 1),
        'cross_aspect_same_depth_rate': per_depth_cross,
    }


class ResidualVocabQuantizer(nn.Module):
    """Residual quantization, per aspect, over an existing FACE `Quantizer`.

    The FACE quantizer is borrowed, not owned: this module adds no parameters
    or buffers, and never builds a second codebook. The codebook is recomputed
    from `codebook_mapping(codebook_tensor_pca)` once per call and shared by
    every aspect and depth.
    """

    def __init__(self, quantizer, max_depth=1, depth1_exclusive=True, reuse_policy='none',
                 track_duplicates=True):
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if reuse_policy not in REUSE_POLICIES:
            raise ValueError(f"reuse_policy must be one of {REUSE_POLICIES}")
        # bypass nn.Module registration so the FACE quantizer is not listed twice
        # in the parent's state_dict
        object.__setattr__(self, 'base', quantizer)
        self.max_depth = max_depth
        self.depth1_exclusive = depth1_exclusive
        self.reuse_policy = reuse_policy
        self.track_duplicates = track_duplicates
        self.last_duplicate_stats = None

    def _select_depth1(self, r, codebook):
        N, A, D = r.shape
        if self.depth1_exclusive:
            if A != self.base.word_num:
                raise ValueError(f"FACE depth-1 selection needs A == word_num ({self.base.word_num}), got {A}")
            return self.base.select_indices(r.reshape(-1, D), codebook).reshape(N, A)
        return torch.argmin(_sq_dist(r.reshape(-1, D), codebook), dim=1).reshape(N, A)

    def _select_deeper(self, r, codebook, used_aspect, used_entity):
        N, A, D = r.shape
        dist = _sq_dist(r.reshape(-1, D), codebook).reshape(N, A, -1)
        if self.reuse_policy == 'none':
            return torch.argmin(dist, dim=-1)
        if self.reuse_policy == 'aspect':
            return torch.argmin(dist.masked_fill(used_aspect, float('inf')), dim=-1)
        # 'entity': greedy in aspect order, as FACE does at depth 1
        rows = torch.arange(N, device=r.device)
        picks = []
        for a in range(A):
            k = torch.argmin(dist[:, a, :].masked_fill(used_entity, float('inf')), dim=-1)
            used_entity[rows, k] = True
            picks.append(k)
        return torch.stack(picks, dim=1)

    def forward(self, z_a, max_depth=None):
        """z_a [N, A, D] -> dict of residual-quantization outputs (L = max_depth)

            token_indices     [N, A, L]      FACE vocabulary idx
            lm_token_ids      [N, A, L]      LM tokenizer ids
            codeword_vectors  [N, A, L, D]   c_a^(h), differentiable w.r.t. codebook_mapping
            residuals         [N, A, L+1, D] r_a^(1..L+1), residuals[:, :, 0] == z_a
            mapped_codebook   [V, D]         the single codebook used at every depth
        """
        L = self.max_depth if max_depth is None else max_depth
        if L < 1:
            raise ValueError("max_depth must be >= 1")
        N, A, D = z_a.shape
        codebook = self.base.get_mapped_codebook()
        V = codebook.shape[0]

        used_aspect = used_entity = None
        if self.reuse_policy == 'aspect':
            used_aspect = torch.zeros(N, A, V, dtype=torch.bool, device=z_a.device)
        elif self.reuse_policy == 'entity':
            used_entity = torch.zeros(N, V, dtype=torch.bool, device=z_a.device)

        r = z_a
        residuals, indices, codewords = [r], [], []
        for h in range(L):
            with torch.no_grad():
                if h == 0:
                    k = self._select_depth1(r, codebook)
                else:
                    k = self._select_deeper(r, codebook, used_aspect, used_entity)
                if used_aspect is not None:
                    used_aspect.scatter_(-1, k.unsqueeze(-1), True)
                if used_entity is not None and h == 0:
                    used_entity.scatter_(-1, k, True)
            c = codebook[k]                    # N, A, D
            r = r - c
            indices.append(k)
            codewords.append(c)
            residuals.append(r)

        token_indices = torch.stack(indices, dim=2)
        out = {
            'token_indices': token_indices,
            'lm_token_ids': self.base.lm_token_ids(token_indices),
            'codeword_vectors': torch.stack(codewords, dim=2),
            'residuals': torch.stack(residuals, dim=2),
            'mapped_codebook': codebook,
        }
        if self.track_duplicates:
            self.last_duplicate_stats = duplicate_token_stats(token_indices)
            out['duplicate_stats'] = self.last_duplicate_stats
            logger.debug("duplicate-token stats: %s", self.last_duplicate_stats)
        return out

    def words(self, token_indices, lengths=None):
        """Text representation: the ordered vocabulary words of each aspect,
        [N][A][L_a] -> str. Built from indices only, never from a summed
        vector (a sum of codewords is not a word)."""
        vocab = self.base.vocabulary
        N, A, L = token_indices.shape
        k = token_indices.tolist()
        ln = [[L] * A for _ in range(N)] if lengths is None else lengths.tolist()
        return [[[vocab[t] for t in k[n][a][:ln[n][a]]] for a in range(A)] for n in range(N)]

    @staticmethod
    def reconstruct(codeword_vectors, lengths=None):
        """z_q,a = sum_{h <= L_a} c_a^(h): [N, A, L, D] (+ lengths [N, A]) -> [N, A, D]."""
        if lengths is None:
            return codeword_vectors.sum(dim=2)
        mask = depth_mask(lengths, codeword_vectors.shape[2])
        return (codeword_vectors * mask.unsqueeze(-1).to(codeword_vectors.dtype)).sum(dim=2)


class AdaptiveFACE(FACE):
    """FACE with the aspect-level quantizer swapped for ResidualVocabQuantizer.

    Construction runs FACE.__init__ unchanged (same RNG draws, same
    parameters), and the residual quantizer adds nothing to the state_dict, so
    FACE checkpoints load with strict=True. With depth=1 the forward pass is
    FACE.forward.
    """

    def __init__(self, input_dim, word_num, word_dim, dataset_name, llm_name,
                 max_depth=1, depth1_exclusive=True, reuse_policy='none'):
        super().__init__(input_dim, word_num, word_dim, dataset_name, llm_name)
        self.residual_quantizer = ResidualVocabQuantizer(
            self.quantizer, max_depth=max_depth, depth1_exclusive=depth1_exclusive,
            reuse_policy=reuse_policy)

    def _vq_terms(self, rq, lengths=None):
        """Per-depth FACE VQ loss, L_VQ^(h) = 0.75*codebook + 0.25*commitment,
        with r_a^(h) in place of z_a. At h=1 this is FACE's term on the same
        shapes. Returns a list of H scalars.
        """
        D = self.word_dim
        codewords, residuals = rq['codeword_vectors'], rq['residuals']
        L = codewords.shape[2]
        mask = None if lengths is None else depth_mask(lengths, L)
        terms = []
        for h in range(L):
            c = codewords[:, :, h].reshape(-1, D)
            r = residuals[:, :, h].reshape(-1, D)
            if mask is None:
                vq = torch.mean((c - r.detach())**2)
                commit = torch.mean((r - c.detach())**2)
            else:
                m = mask[:, :, h].reshape(-1, 1).to(c.dtype)
                denom = (m.sum() * D).clamp(min=1)
                vq = ((c - r.detach())**2 * m).sum() / denom
                commit = ((r - c.detach())**2 * m).sum() / denom
            terms.append(0.75 * vq + 0.25 * commit)
        return terms

    def _vq_loss(self, rq, lengths=None):
        """L_RQ = mean_h L_VQ^(h); equals FACE's vq_loss at depth 1."""
        terms = self._vq_terms(rq, lengths)
        return terms[0] if len(terms) == 1 else torch.stack(terms).mean()

    def decode(self, z_q):
        """FACE decoder: z_q [..., A, D] -> [..., input_dim]."""
        return self.linear_encoder.reverse(self.transformer_decoder(z_q))

    def adapt_losses(self, x, depth=None, lambda_recon=1.0):
        """Step-2 residual-adaptation objective. Nothing textual is involved.

        For every prefix h, z_q^(<=h) = sum_{l<=h} c^(l) lives in CF /
        quantization space only. It is passed straight-through (as FACE does)
        to the unchanged FACE decoder and compared with stopgrad(e):

            L_prefix_recon = mean_h MSE(Decoder(z_q^(<=h)), sg(e))
            L_RQ           = mean_h L_VQ^(h)
            L_adapt        = L_RQ + lambda_recon * L_prefix_recon
        """
        H = self.residual_quantizer.max_depth if depth is None else depth
        N = x.shape[0]
        _, z_e = self.encode_aspects(x)
        rq = self.residual_quantizer(z_e, max_depth=H)

        prefix = rq['codeword_vectors'].cumsum(dim=2)                   # N, A, H, D
        prefix_st = z_e.unsqueeze(2) + (prefix - z_e.unsqueeze(2)).detach()
        # one decoder pass for all depths: [H*N, A, D], depth-major
        dec_in = prefix_st.permute(2, 0, 1, 3).reshape(H * N, self.word_num, self.word_dim)
        decoded = self.decode(dec_in).reshape(H, N, -1)                  # H, N, input_dim
        target = x.detach()
        recon_terms = [F.mse_loss(decoded[h], target) for h in range(H)]
        vq_terms = self._vq_terms(rq)

        rq_loss = torch.stack(vq_terms).mean()
        prefix_recon = torch.stack(recon_terms).mean()
        return {
            'loss': rq_loss + lambda_recon * prefix_recon,
            'rq_loss': rq_loss,
            'prefix_recon': prefix_recon,
            'vq_per_depth': vq_terms,
            'recon_per_depth': recon_terms,
            'decoded': decoded,
            'prefix': prefix,
            'z_a': z_e,
            'rq': rq,
        }

    def forward(self, x, stage, depth=None, lengths=None, return_aux=False):
        linear_out, z_e = self.encode_aspects(x)
        return self.forward_aspects(z_e, x, stage, depth, lengths, return_aux, linear_out=linear_out)

    def forward_aspects(self, z_e, x, stage, depth=None, lengths=None, return_aux=False,
                        linear_out=None):
        """forward() from precomputed aspects z_e [N, A, D]; x is the entity
        embedding e (the reconstruction target)."""
        L = self.residual_quantizer.max_depth if depth is None else depth

        rq = self.residual_quantizer(z_e, max_depth=L)
        z_q_hard = self.residual_quantizer.reconstruct(rq['codeword_vectors'], lengths)  # N, A, D

        vq_loss = self._vq_loss(rq, lengths)
        z_q = z_e + (z_q_hard - z_e).detach()  # straight-through

        if stage == "map":
            collaborative_representations_2 = None
        elif L == 1:
            collaborative_representations_2 = self.quantizer.get_collaborative_representations(z_q)
        else:
            # FACE's text path pinv-maps one vector per aspect back to one word
            # embedding. A sum of several codewords is not a word, and how the
            # multi-token sequence should be written is a later design step.
            raise NotImplementedError("text alignment is only defined for depth=1 so far")

        trans_decoder_out = self.transformer_decoder(z_q)
        decoded = self.linear_encoder.reverse(trans_decoder_out)
        recons_loss = F.mse_loss(decoded, x.detach())

        if return_aux:
            aux = dict(rq)
            aux.update({
                'e': x,
                'linear_out': linear_out,
                'z_a': z_e,
                'z_q_hard': z_q_hard,
                'z_q_st': z_q,
                'lengths': lengths,
            })
            return decoded, vq_loss, recons_loss, collaborative_representations_2, aux
        return decoded, vq_loss, recons_loss, collaborative_representations_2

    @torch.no_grad()
    def tokenize(self, x, depth=None):
        _, z_e = self.encode_aspects(x)
        rq = self.residual_quantizer(z_e, max_depth=depth)
        rq['z_a'] = z_e
        return rq
