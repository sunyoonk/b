# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class RPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        # VSID (item-wise variable-length semantic IDs). Off unless mask_mode != 'none' and slot_mask_path is
        # set; when off, every code path below is the original RPG one.
        # AdaRQ: `depth_path` gives a per-item stopping depth d_i; the slot set is then the level prefix 1..d_i and
        # the same masked input / loss / score / graph code below is reused.
        self.use_depth = bool(config.get('depth_path'))
        self.use_slot_weight = bool(config.get('slot_weight_path'))  # pvi_weighted scores, every slot kept
        self.use_slot_mask = self.use_depth or self.use_slot_weight or (
            config.get('mask_mode', 'none') != 'none' and bool(config.get('slot_mask_path')))
        if self.use_slot_mask:
            if self.use_slot_weight:
                weight = self.tokenizer.item_slot_weights(self.dataset, config['slot_weight_path'])
                self.item_slot_weight = weight.to(self.config['device'])
                mask, prior_logp = torch.ones_like(weight, dtype=torch.bool), None
            elif self.use_depth:
                mask, prior_logp = self.tokenizer.item_depth_mask(self.dataset, config['depth_path']), None
            else:
                mask, prior_logp = self.tokenizer.item_slot_tensors(self.dataset, config['slot_mask_path'])
            self.item_slot_mask = mask.to(self.config['device'])            # (n_items, n_digit) bool, row 0 = padding
            self.item_prior_logp = None if prior_logp is None else prior_logp.to(self.config['device'])
            self.agg_mask = config.get('agg_mask', 'kept')                    # kept | all
            self.score_mode = config.get('score', 'centered_impute')          # centered_impute | entropy_impute | mean_kept | sum_kept | prior_impute
            if self.score_mode == 'centered':                                  # AdaRQ name of the same score
                self.score_mode = 'centered_impute'
            if self.use_slot_weight:
                self.score_mode = 'pvi_weighted'
            # optional read-out temperature for the probability-based ablations (entropy_impute + T*); None = training T
            self.score_temperature = config.get('score_temperature', None)
            self.graph_mode = config.get('graph', 'shared_norm')              # shared_norm | full
            assert self.agg_mask in ('kept', 'all') and self.graph_mode in ('shared_norm', 'full')
            assert self.score_mode in ('centered_impute', 'entropy_impute', 'mean_kept', 'sum_kept', 'prior_impute', 'pvi_weighted',
                                       'rel_logq', 'rel_logq_cond')
            # rel_logq: per-level log q of the training-target token frequency (Laplace +1), natural log, (n_digit * K,)
            self.token_logq = self._token_logq().to(self.config['device'])
            # rel_logq_cond (MDL): item-level log q_l(c_il | c_i,<l), popularity-weighted prefix marginal, (n_items, n_digit)
            if self.score_mode == 'rel_logq_cond':
                self.item_logq_cond = self._item_logq_cond(config.get('q_beta', 1.0)).to(self.config['device'])
            assert self.score_mode != 'prior_impute' or self.item_prior_logp is not None, 'prior_impute needs prior_logp'

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len + (1 if config.get('user_sid_path') else 0),
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )

        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        self.temperature = self.config['temperature']
        # user prefix (adaptive user ID): an .npz with users (raw ids), tokens [U, n_digit] (per-slot codes 0..K-1) and
        # mask [U, n_digit]; the user is encoded as the mean of its kept token embeddings and prepended to the history.
        self.use_user = bool(config.get('user_sid_path'))
        if self.use_user:
            import numpy as _np
            z = _np.load(config['user_sid_path'], allow_pickle=True)
            U = len(dataset.user2id)
            ut = torch.zeros((U, self.tokenizer.n_digit), dtype=torch.long); um = torch.zeros((U, self.tokenizer.n_digit), dtype=torch.bool)
            off = torch.arange(self.tokenizer.n_digit) * self.tokenizer.codebook_size + 1
            for r, u in enumerate(z['users'].tolist()):
                if u in dataset.user2id:
                    ut[dataset.user2id[u]] = torch.from_numpy(z['tokens'][r].astype(_np.int64)) + off
                    um[dataset.user2id[u]] = torch.from_numpy(z['mask'][r].astype(bool))
            self.user_tokens, self.user_mask = ut.to(self.config['device']), um.to(self.config['device'])
            self.user_gwords = None
            if 'gwords' in z.files and config.get('shared_vocab_path'):
                gw = torch.zeros((U, self.tokenizer.n_digit), dtype=torch.long)
                for r, u in enumerate(z['users'].tolist()):
                    if u in dataset.user2id:
                        gw[dataset.user2id[u]] = torch.from_numpy(z['gwords'][r].astype(_np.int64))
                self.user_gwords = gw.to(self.config['device'])
        # shared vocabulary: RPG token t = slot * K + code + 1 is looked up in ONE table through gmap[t] (a word used at
        # several aspects / depths has one embedding, as in InATTo's single shared codebook); per-slot logits stay
        # restricted to that slot's codes. gmap file: npz 'gmap' [n_digit, K] global row ids (row 0 = unused).
        self.shared_vocab = bool(config.get('shared_vocab_path'))
        if self.shared_vocab:
            import numpy as _np
            gm = torch.from_numpy(_np.load(config['shared_vocab_path'])['gmap'].astype(_np.int64)).view(-1)
            self.register_buffer('gmap', torch.cat([torch.zeros(1, dtype=torch.long), gm, torch.zeros(1, dtype=torch.long)]))
            self.shared_emb = nn.Embedding(int(gm.max()) + 1, self.config['n_embd'])
            nn.init.normal_(self.shared_emb.weight, std=self.config.get('initializer_range', 0.02))
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        # Graph-constrained decoding
        self.generate_w_decoding_graph = False
        self.decode_mode = None  # AdaRQ evaluation only: 'trie_astar' (exact top-k by A* over the SID prefix trie)
        self._trie = None
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

    def _code_valid(self):
        """[n_digit, K] bool: codes that some item actually uses at that slot (kept slots only). With
        valid_code_mask the per-slot softmax runs over these candidates W_l only; otherwise over all K codes, where
        unused codes (shared row 0 under the shared vocabulary) still take probability mass."""
        if getattr(self, '_valid_cache', None) is None:
            L, K = self.tokenizer.n_digit, self.tokenizer.codebook_size
            codes = self.item_id2tokens[1:] - 1 - torch.arange(L, device=self.item_id2tokens.device) * K
            kept = self.item_slot_mask[1:].bool() if self.use_slot_mask else torch.ones_like(codes, dtype=torch.bool)
            v = torch.zeros(L, K, dtype=torch.bool, device=codes.device)
            for l in range(L):
                c = codes[:, l][kept[:, l]]
                if len(c):
                    v[l, c] = True
            v[v.sum(1) == 0] = True   # slots never kept: leave unmasked (no target, no score)
            self._valid_cache = v
        return self._valid_cache

    def tok_emb(self, tokens):
        return self.shared_emb(self.gmap[tokens]) if getattr(self, 'shared_vocab', False) else self.gpt2.wte(tokens)

    def slot_table(self):
        """[n_digit * K, d] output token embeddings (rows of tokens 1..n_digit*K)."""
        if getattr(self, 'shared_vocab', False):
            return self.shared_emb.weight[self.gmap[1:-1]]
        return self.gpt2.wte.weight[1:-1]

    def item_log_popularity(self) -> torch.Tensor:
        """log (n_i + 1) of the training targets (i_2..i_{T-2} of every user), (n_items,), padding row 0."""
        targets = torch.tensor([self.dataset.item2id[it] for s in self.dataset.all_item_seqs.values() for it in s[:-2][1:]])
        return (torch.bincount(targets, minlength=self.dataset.n_items).double() + 1).log().float()

    def _token_logq(self) -> torch.Tensor:
        """log q_l(c): per-level frequency of the training targets' tokens (i_2..i_{T-2} of every user), Laplace +1."""
        K, L = self.tokenizer.codebook_size, self.tokenizer.n_digit
        counts = torch.ones(L, K, dtype=torch.float64)
        targets = [self.dataset.item2id[it] for s in self.dataset.all_item_seqs.values() for it in s[:-2][1:]]
        codes = self.item_id2tokens[torch.tensor(targets, dtype=torch.long)].cpu() - 1 - torch.arange(L) * K
        if self.config.get('logq_active') and self.use_slot_mask:
            # q_l(c) = distribution of slot l's code over the training targets whose slot l is ACTIVE, Laplace +1 over
            # the slot's candidate codes W_l. (Default branch below: every target counted, inactive slots as code 0,
            # Laplace over all K codes - code 0 is a real word, so its q was inflated by the inactive targets.)
            tid = torch.tensor(targets, dtype=torch.long)
            act = self.item_slot_mask[tid].bool().cpu()
            valid = self._code_valid().cpu()
            counts = valid.double().clone()                       # Laplace +1 on candidates, 0 elsewhere
            for l in range(L):
                counts[l] += torch.bincount(codes[act[:, l], l], minlength=K).double()
            counts[counts.sum(-1) == 0] = 1.0                     # slots never active: uniform (never scored)
            return (counts / counts.sum(-1, keepdim=True)).clamp(min=1e-30).log().float().view(-1)
        for l in range(L):
            counts[l] += torch.bincount(codes[:, l], minlength=K).double()
        return (counts / counts.sum(-1, keepdim=True)).log().float().view(-1)

    def _item_logq_cond(self, q_beta=1.0) -> torch.Tensor:
        """ln q_l(c_il | c_i,<l) = ln Z(c_i,<=l) - ln Z(c_i,<l), Z(b) = sum_{j in b} (n_j + 1)^q_beta over the items under
        prefix b (n_j: training-target count). Telescopes: sum_{l<=d} ln q_l = ln Z(b_d(i)) - ln Z(root), so with the
        within-node prior pi(i | b) = w_i / Z(b) the score sum_{l<=d}[ln p - ln q] + ln w_i is minus the MDL code length
        of the target up to a per-user constant. (n_items, n_digit), padding row 0."""
        L, K = self.tokenizer.n_digit, self.tokenizer.codebook_size
        codes = (self.item_id2tokens[1:].cpu() - 1 - torch.arange(L) * K).numpy()
        w = (self.item_log_popularity()[1:].double().numpy() * q_beta)          # ln w_i = q_beta * ln(n_i + 1)
        w = np.exp(w - w.max())
        out = np.zeros((codes.shape[0] + 1, L))
        prev = np.log(np.full(codes.shape[0], w.sum()))
        for l in range(L):
            _, inv = np.unique(codes[:, :l + 1], axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            cur = np.log(np.bincount(inv, weights=w)[inv])
            out[1:, l] = cur - prev
            prev = cur
        return torch.from_numpy(out).float()

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def _nested_slot_mask(self):
        """Nested dropout (training only): per batch, every item gets a content depth d ~ U[1, nd_text] and a
        collaborative depth k ~ U[0, nd_cf]; slots beyond them are dropped from its kept mask (dedup slot unchanged).
        Content slots are 0..nd_text-1, collaborative slots nd_cf_from..nd_cf_from+nd_cf-1."""
        Lc, Kc, cf0 = int(self.config['nd_text']), int(self.config.get('nd_cf', 0)), int(self.config.get('nd_cf_from', 0))
        n_items, dev = self.item_slot_mask.shape[0], self.item_slot_mask.device
        d = torch.randint(1, Lc + 1, (n_items, 1), device=dev); k = torch.randint(0, Kc + 1, (n_items, 1), device=dev)
        lv = torch.arange(self.item_slot_mask.shape[1], device=dev)[None, :]
        keep = torch.ones_like(self.item_slot_mask, dtype=torch.bool)
        keep &= ~((lv < Lc) & (lv >= d))                                     # content slots beyond d
        if Kc:
            keep &= ~((lv >= cf0) & (lv < cf0 + Kc) & (lv - cf0 >= k))       # collaborative slots beyond k
        return self.item_slot_mask.bool() & keep

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_tokens = self.item_id2tokens[batch['input_ids']]
        nested = self.training and self.use_slot_mask and self.config.get('nested_dropout')
        slot_mask = self._nested_slot_mask() if nested else self.item_slot_mask
        if self.use_slot_mask and self.agg_mask == 'kept':
            # VSID: input item = mean of the embeddings of its kept slot tokens
            keep = slot_mask[batch['input_ids']].unsqueeze(-1).to(self.gpt2.wte.weight.dtype)
            agg_max = self.config.get('agg_max_level')
            if agg_max:   # history items are encoded with their first agg_max semantic levels (+ dedup/CF slots) only
                keep = keep.clone(); keep[..., agg_max:self.n_pred_head - 3, :] = 0
            ds = self.config.get('input_drop_slot')
            if ds is not None:   # collision (ordinal) slot: supervised and scored, but not part of the input mean x_t
                keep = keep.clone(); keep[..., int(ds), :] = 0
            input_embs = (self.tok_emb(input_tokens) * keep).sum(dim=-2) / keep.sum(dim=-2).clamp(min=1)
        else:
            input_embs = self.tok_emb(input_tokens).mean(dim=-2)
        attn = batch['attention_mask']
        if self.use_user:
            uk = self.user_mask[batch['user_ids']].unsqueeze(-1).to(self.gpt2.wte.weight.dtype)
            if getattr(self, 'user_gwords', None) is not None:   # user words looked up directly in the shared word table
                u_emb = (self.shared_emb(self.user_gwords[batch['user_ids']]) * uk).sum(-2) / uk.sum(-2).clamp(min=1)
            else:
                u_emb = (self.tok_emb(self.user_tokens[batch['user_ids']]) * uk).sum(-2) / uk.sum(-2).clamp(min=1)
            input_embs = torch.cat([u_emb.unsqueeze(1), input_embs], 1)
            attn = torch.cat([torch.ones_like(attn[:, :1]), attn], 1)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=attn
        )
        if self.use_user:   # drop the user position: outputs stay aligned with the item positions
            outputs.last_hidden_state = outputs.last_hidden_state[:, 1:]
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            label_mask = batch['labels'].view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.slot_table()
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
            if self.config.get('valid_code_mask'):
                vm = self._code_valid()
                token_logits = [lg.masked_fill(~vm[i], -1e4) for i, lg in enumerate(token_logits)]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            if self.use_slot_mask:
                # VSID: CE only on the target item's kept slots, averaged over kept tokens
                target_mask = slot_mask[batch['labels'].view(-1)[label_mask]].float()  # (n_targets, n_digit)
                rf = self.config.get('residual_from')
                if rf is not None:   # residual deepening: loss only on the deeper semantic levels
                    target_mask = target_mask.clone(); target_mask[:, :int(rf)] = 0; target_mask[:, self.n_pred_head - 3:] = 0
                ce = torch.stack([
                    F.cross_entropy(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1, reduction='none')
                    for i in range(self.n_pred_head)
                ], dim=1)
                if self.config.get('loss_norm', 'token') == 'item':
                    # item-level normalisation: every target item weighs the same whatever its SID length
                    outputs.loss = ((ce * target_mask).sum(1) / target_mask.sum(1).clamp(min=1)).mean()
                else:
                    outputs.loss = (ce * target_mask).sum() / target_mask.sum().clamp(min=1)
            else:
                losses = [
                    self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1)
                    for i in range(self.n_pred_head)
                ]
                outputs.loss = torch.mean(torch.stack(losses))
        return outputs

    def score_items(self, token_logits, item_ids):
        """VSID item scores (only used when use_slot_mask). token_logits: (B, n_tokens) per-slot log-probs;
        item_ids: (B, K). Returns (B, K).
          centered_impute (main): sum_kept l~_j(c_ij), l~_j(c) = l_j(c) - mean_c l_j(c) with l_j = cos / T_train;
              a missing slot contributes 0. log_softmax only shifts l_j by a per-slot constant, so
              l~ = log p - mean_c log p. All slots kept -> n_digit * (original score) - const(user): same ranking as
              RPG; the ranking does not depend on T (l~ scales by 1/T for every item).
          entropy_impute: (sum_kept log p_j(c_ij) + sum_missing sum_c p_j(c) log p_j(c)) / n_digit
          mean_kept / sum_kept: mean / sum over kept slots
          prior_impute: (sum_kept log p_j(c_ij) + sum_missing log p_hat_j(c_ij)) / n_digit
        """
        idx = self.item_id2tokens[item_ids] - 1                                     # (B, K, n_digit)
        lp = torch.gather(token_logits.unsqueeze(1).expand(-1, idx.shape[1], -1), -1, idx)
        sw = getattr(self, 'item_score_weight', None)   # optional soft per-level score weights (scoring only)
        keep = self.item_slot_mask[item_ids] if sw is None else sw[item_ids]
        kept_sum = (lp * keep).sum(-1)
        if self.score_mode == 'rel_logq':
            # sum_{l in A_i} [log p_l(c_il | h; T) - log q_l(c_il)] (T = score_temperature, fitted on held-out data)
            sc = ((lp - getattr(self, 'logq_alpha', 1.0) * self.token_logq[idx]) * keep).sum(-1)   # alpha: PMI exponent (1 = rel_logq)
            if self.config.get('score_norm') == 'mean':
                # mean over the item's active slots, no popularity prior (pop_beta is ignored)
                return sc / keep.float().sum(-1).clamp(min=1)
            if getattr(self, 'pop_beta', 0.0):   # length-independent item popularity prior: + beta * log n_i
                sc = sc + self.pop_beta * self.item_logpop[item_ids]
            if getattr(self, 'len_gamma', 0.0):  # length-bias correction: + gamma * (# semantic levels not kept)
                n_sem = self.config.get('n_sem_levels', 16)
                sc = sc + self.len_gamma * (n_sem - self.item_slot_mask[item_ids][..., :n_sem].float().sum(-1))
            return sc
        if self.score_mode == 'rel_logq_cond':
            # MDL: sum_{l in A_i} [log p_l(c_il | h; T) - log q_l(c_il | c_i,<l)] + beta * log(n_i + 1)
            sc = ((lp - self.item_logq_cond[item_ids]) * keep).sum(-1)
            if getattr(self, 'pop_beta', 0.0):
                sc = sc + self.pop_beta * self.item_logpop[item_ids]
            return sc
        if self.score_mode == 'pvi_weighted':
            # sum_j w_ij * l~_j(c_ij), l~ = centered logits (log p minus its per-slot mean); w from slot_weight_path
            per_slot = token_logits.view(token_logits.shape[0], self.n_pred_head, -1)
            centered = lp - per_slot.mean(-1).unsqueeze(1)
            return (centered * self.item_slot_weight[item_ids]).sum(-1)
        if self.score_mode == 'centered_impute':
            per_slot = token_logits.view(token_logits.shape[0], self.n_pred_head, -1)    # (B, n_digit, codebook)
            slot_mean = per_slot.mean(-1)                                                # (B, n_digit)
            return kept_sum - (slot_mean.unsqueeze(1) * keep).sum(-1)
        if self.score_mode == 'sum_kept':
            return kept_sum
        if self.score_mode == 'mean_kept':
            return kept_sum / keep.sum(-1).clamp(min=1)
        if self.score_mode == 'entropy_impute':
            per_slot = token_logits.view(token_logits.shape[0], self.n_pred_head, -1)    # (B, n_digit, codebook)
            neg_entropy = (per_slot.exp() * per_slot).sum(-1)                            # (B, n_digit): E_p[log p]
            missing = (neg_entropy.unsqueeze(1) * (~keep)).sum(-1)
        else:  # prior_impute
            missing = (self.item_prior_logp[item_ids] * (~keep)).sum(-1)
        return (kept_sum + missing) / self.n_pred_head

    def build_ii_sim_mat(self):
        # Assuming n_digit=32, codebook_size=256
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # 4) Prepare an output similarity matrix
        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # shape: (chunk_i_size, 32)
            tokens_i = self.item_id2tokens[i_start:i_end]  # sub-block for items i

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # shape: (chunk_j_size, 32)
                tokens_j = self.item_id2tokens[j_start:j_end]  # sub-block for items j

                # We want to compute a sub-block of shape: (chunk_i_size, chunk_j_size).
                # For each digit k in [0..31], we look up token_sims_01[k, tokens_i[i, k], tokens_j[j, k]].

                # We'll accumulate the similarity for each of the 32 digits
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)
                shared = self.use_slot_mask and self.graph_mode == 'shared_norm'
                if shared:  # VSID: only slots kept by both items count
                    mask_i = self.item_slot_mask[i_start:i_end]
                    mask_j = self.item_slot_mask[j_start:j_end]

                # We'll do a small loop over k=0..31 (which is constant = 32).
                # Each token_sims_01[k] is (256, 256). We gather from it using:
                #   row indices = tokens_i[:, k]
                #   col indices = tokens_j[:, k]
                #
                # The typical approach is:
                #   sub = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                # Then sum them up across k.
                for k in range(n_digit):
                    # row_inds shape: (block_size_i,)
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    # col_inds shape: (block_size_j,)
                    col_inds = tokens_j[:, k] - k * codebook_size - 1

                    # token_sims_01[k] -> shape (256, 256)
                    # row-gather => shape (block_size_i, 256)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    # col-gather across dim=1 => shape (block_size_i, block_size_j)
                    temp = temp.index_select(1, col_inds)
                    if shared:
                        temp = temp * (mask_i[:, k].unsqueeze(1) & mask_j[:, k].unsqueeze(0))

                    # Accumulate
                    sum_block += temp

                # Now take the average across the 32 digits
                if shared:  # sum over shared slots / sqrt(|A_i| |A_j|)
                    n_i = mask_i.sum(-1).float().unsqueeze(1)
                    n_j = mask_j.sum(-1).float().unsqueeze(0)
                    avg_block = sum_block / torch.sqrt(n_i * n_j).clamp(min=1)
                else:
                    avg_block = sum_block / n_digit

                # Write back into the final item_item_sim
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        batch_size = token_logits.shape[0]

        # Initialize visited nodes tracking
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Randomly sample num_beams distinct node IDs in [1..n_nodes]
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Add initial nodes to visited set
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        for sid in range(self.propagation_steps):
            # Find neighbors of these top num_beams nodes
            #      adjacency_list is 0-based internally => need node_id-1
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])

                # Add neighbors to visited set
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                if self.use_slot_mask:
                    scores = self.score_items(token_logits[batch_id:batch_id + 1], neighbors_in_batch.unsqueeze(0))[0]
                else:
                    scores = torch.gather(
                        input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                        dim=-1,
                        index=(self.item_id2tokens[neighbors_in_batch] - 1)
                    ).mean(dim=-1)

                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        # Convert visited counts to tensor
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)

        token_emb = self.slot_table()
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        temperature = self.temperature
        if self.use_slot_mask and self.score_temperature:
            temperature = self.score_temperature  # VSID ablation: calibrated read-out temperature
        logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / temperature for i in range(self.n_pred_head)]
        if self.config.get('valid_code_mask'):
            vm = self._code_valid()
            logits = [lg.masked_fill(~vm[i], -1e4) for i, lg in enumerate(logits)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

        if self.decode_mode == 'trie_astar':
            return self.trie_astar_generate(token_logits, n_return_sequences)
        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            outputs = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
            return outputs
        elif self.use_slot_mask:
            all_items = torch.arange(1, self.dataset.n_items, device=token_logits.device)
            item_logits = self.score_items(token_logits, all_items.unsqueeze(0).expand(token_logits.shape[0], -1))
            if getattr(self, 'prior_fn', None) is not None and getattr(self, 'prior_gamma', 0.0):
                # collaborative prior: + gamma * log p_prior(i | history) (scripts/cuit_fuse.py)
                item_logits = item_logits + self.prior_gamma * self.prior_fn(batch)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            return preds.unsqueeze(-1)
        else:
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),              # (batch_size, n_items, n_tokens)
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)  # (batch_size, n_items, code_dim)
            ).mean(dim=-1)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            return preds.unsqueeze(-1)

    # ------------------------------------------------------------------------------------------------------------------
    # AdaRQ: exact top-k by A* over the semantic-ID prefix trie (evaluation only)
    def _level_values(self, token_logits):
        """Per-level additive item-score table v[b, l, c] with score(u, i) = sum_{l < d_i} v[b, l, c_il]
        (same ranking as score_items / the original RPG mean)."""
        lp = token_logits.view(token_logits.shape[0], self.n_pred_head, -1)
        if not self.use_slot_mask:
            return lp                                   # original RPG: mean over all levels = sum / n_digit
        if self.score_mode == 'centered_impute':
            return lp - lp.mean(-1, keepdim=True)
        if self.score_mode == 'rel_logq':
            return lp - getattr(self, 'logq_alpha', 1.0) * self.token_logq.view(1, self.n_pred_head, -1)
        if self.score_mode == 'sum_kept':
            return lp
        raise NotImplementedError(f'trie_astar needs a level-additive score, not {self.score_mode}')

    def _build_trie(self):
        """Nodes: children {code: node}, items at the node, depth, min / max item depth in the subtree."""
        K = self.tokenizer.codebook_size
        L = self.n_pred_head
        codes = (self.item_id2tokens[1:].cpu() - 1 - torch.arange(L) * K).tolist()
        if self.use_slot_mask:
            mask = self.item_slot_mask[1:].cpu()
            depth = mask.sum(-1)
            assert torch.equal(mask, torch.arange(L).unsqueeze(0) < depth.unsqueeze(1)), 'trie_astar needs prefix (depth) masks'
            depth = depth.tolist()
        else:
            depth = [L] * len(codes)
        children, items, ndepth = [{}], [[]], [0]
        for item, (c, d) in enumerate(zip(codes, depth), start=1):
            node = 0
            for l in range(d):
                nxt = children[node].get(c[l])
                if nxt is None:
                    nxt = len(children)
                    children[node][c[l]] = nxt
                    children.append({}); items.append([]); ndepth.append(l + 1)
                node = nxt
            items[node].append(item)
        dmin, dmax = [L + 1] * len(children), [-1] * len(children)
        for node in range(len(children) - 1, -1, -1):     # children have larger ids than their parent
            if items[node]:
                dmin[node] = min(dmin[node], ndepth[node]); dmax[node] = max(dmax[node], ndepth[node])
            for ch in children[node].values():
                dmin[node] = min(dmin[node], dmin[ch]); dmax[node] = max(dmax[node], dmax[ch])
        self._trie = (children, items, ndepth, dmin, dmax)

    @torch.no_grad()
    def trie_astar_generate(self, token_logits, n_return_sequences):
        """A* over the prefix trie: g = sum of v over the node's prefix, h = max over the subtree's possible stopping
        depths d of sum_{t <= l < d} max_c v[l, c] (admissible), items are emitted with their exact score, so the
        first n emitted items are the exact top-n (ties in any order). Users are searched in parallel by
        `astar_workers` forked processes (default 1). Popped heap entries per user are appended to self.astar_stats."""
        global _ASTAR_TRIE
        if self._trie is None:
            self._build_trie()
        _ASTAR_TRIE = self._trie
        V = self._level_values(token_logits).double().cpu()
        cum = torch.cat([torch.zeros(V.shape[0], 1, dtype=V.dtype), V.max(-1).values.cumsum(-1)], -1).tolist()
        args = [(v, c, n_return_sequences) for v, c in zip(V.tolist(), cum)]
        workers = int(self.config.get('astar_workers', 1) or 1)
        if workers > 1 and len(args) > 1:
            if getattr(self, '_astar_pool', None) is None:
                import multiprocessing as mp
                self._astar_pool = mp.get_context('fork').Pool(workers)   # forked after the trie exists
            results = self._astar_pool.map(_astar_one, args)
        else:
            results = [_astar_one(a) for a in args]
        self.astar_stats = getattr(self, 'astar_stats', []) + [n for _, n in results]
        return torch.tensor([r for r, _ in results], device=token_logits.device).unsqueeze(-1)


_ASTAR_TRIE = None


def _astar_one(args):
    """One user's A* (module level so that forked workers can run it on the inherited trie)."""
    import heapq
    v, C, n = args
    children, items, ndepth, dmin, dmax = _ASTAR_TRIE

    def h(node):
        return max(C[dmin[node]:dmax[node] + 1]) - C[ndepth[node]]
    heap = [(-h(0), 0, 0, 0, 0.0)]                        # (-f, tie, kind 0 node / 1 item, id, g)
    tie, res, n_pop = 1, [], 0
    while heap and len(res) < n:
        _, _, kind, x, g = heapq.heappop(heap)
        n_pop += 1
        if kind == 1:
            res.append(x)
            continue
        for it in items[x]:
            heapq.heappush(heap, (-g, tie, 1, it, g)); tie += 1
        if children[x]:
            vl = v[ndepth[x]]
            for code, ch in children[x].items():
                gc = g + vl[code]
                heapq.heappush(heap, (-(gc + h(ch)), tie, 0, ch, gc)); tie += 1
    return res + [res[-1]] * (n - len(res)), n_pop
