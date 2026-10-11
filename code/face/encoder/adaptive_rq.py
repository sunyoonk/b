"""PVI-guided adaptive residual quantization: sequential halting DURING training.

For every (entity, aspect):
    depth 1 is mandatory:   c^(1) = Q(z_a)            (FACE's depth-1 rule)
    for h = 2 .. H_max, only while the pair is still active:
        one candidate       c~^(h) = Q(r^(h))
        g^(h) = log p_cur(y | S, c^(<h), c~^(h)) - log p_prev(y | S, c^(<h))
        g > 0  -> accept: c^(h) = c~^(h), r^(h+1) = r^(h) - c^(h), VQ term counted
        g <= 0 -> reject: nothing is appended, the residual is untouched, and the
                  pair is never queried again (VQ loss on the rejected proposal
                  only when vq_on='queried'; see below)
The loop stops as soon as no pair is active. Nothing is generated past a stop
and nothing is truncated afterwards (no post-hoc pruning).

The controller runs under no_grad and only yields a discrete mask; no gradient
from g or BERT reaches the RQ modules.

Two masks: qmask^(h) = the candidate at depth h was queried (generated and
scored), amask^(h) = it was accepted. Always amask <= qmask; at a stopping
depth qmask = 1, amask = 0. `vq_on` chooses which mask carries the VQ loss:
'accepted' (original V1 rule: a rejected candidate gets no loss) or 'queried'
(main method: the queried stopping proposal also learns to quantize its
residual, so a poor candidate is not rejected forever). Either way a rejected
candidate never enters z_q, never updates the residual, and enables no
deeper query.
"""
import torch
import torch.nn.functional as F

STOP_HMAX, STOP_REJECTED, STOP_NO_CONTROLLER = 0, 1, 2


def _sq_dist(r, codebook):
    return torch.sum(r ** 2, dim=1, keepdim=True) + torch.sum(codebook ** 2, dim=1) - \
        2 * torch.matmul(r, codebook.t())


def _vq_pair(c, r):
    """FACE's VQ/commitment per (entity, aspect) pair, mean over D -> [M].
    With every k=1 the masked mean equals FACE's vq_loss."""
    return 0.75 * ((c - r.detach()) ** 2).mean(-1) + 0.25 * ((r - c.detach()) ** 2).mean(-1)


def adaptive_quantize(quantizer, z_a, rows, controller, H_max, depth1_exclusive=True, vq_on='accepted'):
    """z_a [N, A, D] (fixed); rows [N] global entity rows (for the controller).

    Returns tokens [N,A,H_max] (-1 = not selected), mask, k [N,A], gains
    (nan where no candidate was scored), cand_tokens (every scored candidate,
    accepted or not), queries [N,A] (nearest-codeword lookups), stop_reason,
    z_q [N,A,D] = sum of accepted codewords, residual [N,A,D] = r^(k+1),
    L_arq (masked mean over the `vq_on` mask) and n_terms. Diagnostics:
    qmask, r_norm / cand_sqdist [N,A,H_max] (nan where not queried).
    """
    if vq_on not in ('accepted', 'queried'):
        raise ValueError("vq_on must be 'accepted' or 'queried'")
    N, A, D = z_a.shape
    dev = z_a.device
    z = z_a.detach()
    C = quantizer.get_mapped_codebook()                      # one codebook for every depth
    tokens = torch.full((N, A, H_max), -1, dtype=torch.long, device=dev)
    cand_tokens = tokens.clone()
    mask = torch.zeros(N, A, H_max, dtype=torch.bool, device=dev)
    gains = torch.full((N, A, H_max), float('nan'), device=dev)
    queries = torch.zeros(N, A, dtype=torch.long, device=dev)
    stop_reason = torch.full((N, A), STOP_HMAX, dtype=torch.long, device=dev)
    r_norm = torch.full((N, A, H_max), float('nan'), device=dev)
    cand_sqdist = torch.full((N, A, H_max), float('nan'), device=dev)

    # depth 1: mandatory for every pair
    with torch.no_grad():
        if depth1_exclusive:
            k1 = quantizer.select_indices(z.reshape(-1, D), C).reshape(N, A)
        else:
            k1 = torch.argmin(_sq_dist(z.reshape(-1, D), C), 1).reshape(N, A)
    queries += 1
    c1 = C[k1]
    tokens[:, :, 0] = k1
    cand_tokens[:, :, 0] = k1
    mask[:, :, 0] = True
    terms = [_vq_pair(c1.reshape(-1, D), z.reshape(-1, D))]
    with torch.no_grad():
        r_norm[:, :, 0] = z.norm(dim=-1)
        cand_sqdist[:, :, 0] = ((z - c1) ** 2).mean(-1)
    r = z - c1
    zq = c1
    active = torch.ones(N, A, dtype=torch.bool, device=dev)

    for h in range(2, H_max + 1):
        if not active.any():
            break
        n_i, a_i = active.nonzero(as_tuple=True)
        avail = controller.available(h, rows[n_i], a_i) if controller is not None \
            else torch.zeros(len(n_i), dtype=torch.bool, device=dev)
        stop_reason[n_i[~avail], a_i[~avail]] = STOP_NO_CONTROLLER
        n_i, a_i = n_i[avail], a_i[avail]
        active = torch.zeros_like(active)
        if len(n_i) == 0:
            break
        r_act = r[n_i, a_i]
        with torch.no_grad():
            cand = torch.argmin(_sq_dist(r_act.detach(), C.detach()), 1)
        queries[n_i, a_i] += 1
        cand_tokens[n_i, a_i, h - 1] = cand
        with torch.no_grad():
            g = controller.gains(h, rows[n_i], a_i, tokens[n_i, a_i, :h - 1], cand).float()
        gains[n_i, a_i, h - 1] = g
        acc = g > 0
        stop_reason[n_i[~acc], a_i[~acc]] = STOP_REJECTED
        with torch.no_grad():
            r_norm[n_i, a_i, h - 1] = r_act.norm(dim=-1)
            cand_sqdist[n_i, a_i, h - 1] = ((r_act - C[cand]) ** 2).mean(-1)
        if vq_on == 'queried' and (~acc).any():
            # the rejected stopping proposal still learns to quantize r^(h);
            # it is not committed (no z_q, no residual update, no deeper query)
            terms.append(_vq_pair(C[cand[~acc]], r_act[~acc]))
        na, aa, ca = n_i[acc], a_i[acc], cand[acc]
        if len(na) == 0:
            break
        c = C[ca]
        r_in = r[na, aa]
        terms.append(_vq_pair(c, r_in))
        tokens[na, aa, h - 1] = ca
        mask[na, aa, h - 1] = True
        r = r.index_put((na, aa), r_in - c)
        zq = zq.index_put((na, aa), zq[na, aa] + c)
        active[na, aa] = True

    allterms = torch.cat(terms)
    n_terms = allterms.numel()
    return {
        'tokens': tokens, 'mask': mask, 'qmask': cand_tokens >= 0, 'k': mask.sum(-1), 'gains': gains,
        'cand_tokens': cand_tokens, 'r_norm': r_norm, 'cand_sqdist': cand_sqdist, 'vq_on': vq_on,
        'queries': queries, 'stop_reason': stop_reason, 'z_q': zq, 'residual': r,
        'L_arq': allterms.sum() / max(n_terms, 1), 'n_terms': n_terms, 'mapped_codebook': C,
    }


def adaptive_map_step(model, z_a, e, rows, controller, H_max, lambda_rq=1.0, lambda_recon=1.0,
                      depth1_exclusive=True, vq_on='accepted'):
    """L_adaptive_map = lambda_rq * L_ARQ + lambda_recon * MSE(Dec(Z_q*), sg(e)).
    The decoder gets FACE's straight-through z_a + sg(Z_q* - z_a), shape [N, A, D]."""
    out = adaptive_quantize(model.quantizer, z_a, rows, controller, H_max, depth1_exclusive, vq_on)
    z = z_a.detach()
    dec_in = z + (out['z_q'] - z).detach()
    decoded = model.decode(dec_in)
    recon = F.mse_loss(decoded, e.detach())
    out.update({'decoded': decoded, 'L_recon': recon,
                'loss': lambda_rq * out['L_arq'] + lambda_recon * recon})
    return out


class PVIController:
    """Frozen (f_prev, f_cur) pairs keyed by (side, aspect, h).

    Builds X_prev = S_e [SEP] words(c^(<h)) and X_cur = S_e [SEP] words(c^(<h)) + word(c~^(h))
    from the ACTUAL vocabulary strings, and returns g = log q - log pi in nats.
    """

    def __init__(self, pairs, texts, labels, vocab, is_item):
        self.pairs = pairs                       # {(side, aspect, h): ControllerPair}
        self.texts, self.vocab = texts, vocab
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.is_item = torch.as_tensor(is_item)
        self.calls = []                          # (h, n) per gains() call, for accounting

    def _side(self, rows):
        return self.is_item[rows.cpu()]

    def available(self, h, rows, aspects):
        side = self._side(rows)
        ok = [((('item' if s else 'user'), int(a), h) in self.pairs) for s, a in zip(side.tolist(), aspects.tolist())]
        return torch.tensor(ok, dtype=torch.bool, device=rows.device)

    @torch.no_grad()
    def gains(self, h, rows, aspects, prefix, cand):
        g = torch.empty(len(rows), device=rows.device)
        side = self._side(rows)
        keys = [(('item' if s else 'user'), int(a), h) for s, a in zip(side.tolist(), aspects.tolist())]
        pre, cd, rr, aa = prefix.tolist(), cand.tolist(), rows.tolist(), aspects.tolist()
        for key in set(keys):
            idx = [i for i, k in enumerate(keys) if k == key]
            ta = [self.texts[rr[i]] for i in idx]
            bp = [' '.join(self.vocab[t] for t in pre[i]) for i in idx]
            bc = [(bp[j] + ' ' + self.vocab[cd[i]]).strip() for j, i in enumerate(idx)]
            y = self.labels[[rr[i] for i in idx], [aa[i] for i in idx]]
            _, _, gi = self.pairs[key].gain(ta, bp, bc, y)
            g[torch.tensor(idx, device=rows.device)] = gi.to(rows.device).float()
        self.calls.append((h, len(rows)))
        return g
