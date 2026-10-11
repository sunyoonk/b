"""CUIT: FACE (LightGCN_vq) with the aspect quantizer extended to residual quantization over the same word codebook.

Identical to LightGCN_vq except:
  * self.vqraf = AdaptiveFACE(max_depth = H): residual vocabulary quantization, level 1 = FACE's quantizer (cross-aspect
    exclusion), VQ loss = mean over levels of FACE's 0.75 * codebook + 0.25 * commitment term.
  * map stage: every level active (dense), so the codebook mapping / decoder learn all depths.
  * align stage: per-entity depths (users, items) are fixed from .npz files (decided between the two stages by the
    information-exhaustion test); the FACE text path is generalised from one word per aspect to the active words of
    every aspect: [prompt] w_{1,1} , w_{1,2} , ... , w_{A,d_A}  (FACE's prompt and comma separator). Each word vector is the
    pinv-mapped codeword with a straight-through gradient to the residual it quantised (at depth 1 this is FACE's STE
    to z_a). Loss, weights, temperature, batch construction and training loop are FACE's.
Checkpoints: map loads ./encoder/checkpoint/lightgcn/lightgcn-<data>-<seed>.pth, align loads this model's _map.pth.
"""
import os
import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, cal_align_loss
from models.general_cf.lightgcn_vq import LightGCN_vq
from FACE_adaptive import AdaptiveFACE, depth_mask
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class LightGCN_rqvq(BaseModel):
    def __init__(self, data_handler):
        BaseModel.__init__(self, data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False
        self.usrprf_repre = t.tensor(configs['usrprf_repre']).float().cuda()
        self.itmprf_repre = t.tensor(configs['itmprf_repre']).float().cuda()
        self.layer_num = self.hyper_config['layer_num']
        self.word_num = self.hyper_config['word_num']
        self.word_dim = self.hyper_config['word_dim']
        self.vq_weight = self.hyper_config['vq_weight']
        self.recons_weight = self.hyper_config['recons_weight']
        self.align_weight = self.hyper_config['align_weight']
        self.H = int(self.hyper_config.get('max_depth', 8))
        self.vqraf = AdaptiveFACE(self.embedding_size, self.word_num, self.word_dim, configs['data']['name'], configs['llm'],
                                  max_depth=self.H)
        name, data, seed = configs['model']['name'], configs['data']['name'], configs['train']['seed']
        if configs['stage'] == 'map':
            p = f"./encoder/checkpoint/lightgcn/lightgcn-{data}-{seed}.pth"
            self.load_state_dict(t.load(p), strict=False); print(f"Successfully load model from {p}")
            if os.environ.get('FACE_MAP_INIT'):   # warm start from a trained (original FACE) map checkpoint
                p = os.environ['FACE_MAP_INIT']; self.load_state_dict(t.load(p)); print(f"Warm start from {p}")
            self.len_user = self.len_item = None
        else:
            p = f"./encoder/checkpoint/{name}/{name}-{data}-{seed}_map{os.environ.get('FACE_MAP_SUFFIX', '')}.pth"
            self.load_state_dict(t.load(p)); print(f"Successfully load model from {p}")
            if os.environ.get('FACE_FULL_DEPTH') == '1':   # fallback: align with every level active (depth decided after align)
                self.len_user = t.full((self.user_num, self.word_num), self.H).long().cuda()
                self.len_item = t.full((self.item_num, self.word_num), self.H).long().cuda()
            else:
                dd = os.environ.get('FACE_DEPTH_DIR')   # optional: depth files of another artifact dir
                du = f'{dd}/depth_user_face.npy' if dd else self.hyper_config['depth_user']
                di = f'{dd}/depth_item_face.npy' if dd else self.hyper_config['depth_item']
                self.len_user = t.tensor(np.load(du)).long().cuda()   # [n_users, A]
                self.len_item = t.tensor(np.load(di)).long().cuda()   # [n_items, A]

    _propagate = LightGCN_vq._propagate
    forward = LightGCN_vq.forward
    _pick_embeds = LightGCN_vq._pick_embeds
    full_predict = LightGCN_vq.full_predict

    def _text_repr(self, rq, lengths, n_user_rows):
        """FACE text path for variable depth: prompt + active words separated by commas -> MiniLM sentence embedding."""
        q = self.vqraf.quantizer
        cw, res = rq['codeword_vectors'], rq['residuals'][:, :, :-1]                       # [N, A, H, D]
        st = cw + (res - res.detach())                                                     # value = codeword, grad -> residual
        N, A, H, D = st.shape
        words = q.reverse_codebook_mapping(st.reshape(-1, D)).reshape(N, A * H, -1)       # [N, A*H, d_tok]
        m = depth_mask(lengths, H).reshape(N, A * H)                                      # active words
        comma = q.prompt_comma.reshape(1, 1, -1).expand(N, A * H, -1)
        inter = t.stack([words, comma], 2).reshape(N, 2 * A * H, -1)                      # w , w , ...
        im = t.stack([m, m], 2).reshape(N, 2 * A * H)
        last = (im.cumsum(1) == im.sum(1, keepdim=True)) & im                             # drop the trailing comma
        im = im & ~(last & (t.arange(2 * A * H, device=im.device) % 2 == 1)[None])
        # gather the active positions to the front (keeps word order), pad the rest
        order = t.argsort((~im).int(), dim=1, stable=True)
        seq = t.gather(inter, 1, order[..., None].expand(-1, -1, inter.shape[-1]))
        msk = t.gather(im, 1, order)
        L = int(msk.sum(1).max())
        seq, msk = seq[:, :L], msk[:, :L]
        pe = q.prompt_embedding                                                            # [3, T_p, d]: user, item, item
        pr = t.cat([pe[0:1].expand(n_user_rows, -1, -1), pe[1:2].expand(N - n_user_rows, -1, -1)], 0)
        x = t.cat([pr, seq], 1)
        am = t.cat([t.ones(N, pr.shape[1], dtype=t.long, device=x.device), msk.long()], 1)
        x = q.embedding_model.add_special_tokens_for_embeddings(x)
        am = t.cat([am.new_ones(N, 1), am, am.new_ones(N, 1)], 1)
        return q.embedding_model.encode_embeddings(token_embeddings=x, attention_mask=am)

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)
        ancs, poss, negs = batch_data
        x = t.cat([anc_embeds, pos_embeds, neg_embeds], dim=0)
        lengths = None
        if self.len_user is not None:
            lengths = t.cat([self.len_user[ancs], self.len_item[poss], self.len_item[negs]], 0)
        linear_out, z_e = self.vqraf.encode_aspects(x)
        rq = self.vqraf.residual_quantizer(z_e, max_depth=self.H)
        z_q_hard = self.vqraf.residual_quantizer.reconstruct(rq['codeword_vectors'], lengths)
        vq_loss = self.vqraf._vq_loss(rq, lengths)
        # prefix reconstruction: every prefix h = 1..H (capped at the entity's depth in align) must decode to e on its
        # own, so level 1 stays FACE's code (h = 1 is FACE's recon term) instead of collapsing to a shared offset
        # (with a full-sum-only recon, map v1 gave 1 word per aspect at level 1 for all users).
        N = x.shape[0]
        lens = lengths if lengths is not None else t.full((N, self.word_num), self.H, device=x.device, dtype=t.long)
        pre = t.stack([self.vqraf.residual_quantizer.reconstruct(rq['codeword_vectors'], lens.clamp(max=h))
                       for h in range(1, self.H + 1)], 0)                                    # [H, N, A, D]
        pre_st = z_e.unsqueeze(0) + (pre - z_e.unsqueeze(0)).detach()
        dec = self.vqraf.decode(pre_st.reshape(self.H * N, self.word_num, -1)).reshape(self.H, N, -1)
        per_h = ((dec - x.detach().unsqueeze(0)) ** 2).mean(dim=(1, 2))                       # [H]
        if os.environ.get('FACE_L1_FULL') == '1':
            # v3: level 1 keeps FACE's full recon weight (as in original FACE); deeper prefixes share one more unit
            recons_loss = per_h[0] + per_h[1:].mean()
        else:
            recons_loss = per_h.mean()
        if configs['stage'] == 'map':
            align_loss = t.tensor(0.0, device=x.device)
        else:
            colla = self._text_repr(rq, lengths, anc_embeds.shape[0])
            ancprf, posprf, negprf = self._pick_embeds(self.usrprf_repre, self.itmprf_repre, batch_data)
            align_loss = cal_align_loss(colla, t.cat([ancprf, posprf, negprf], dim=0))
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        loss = bpr_loss + self.vq_weight * vq_loss + self.recons_weight * recons_loss + self.align_weight * align_loss
        return loss, {'bpr_loss': bpr_loss, 'vq_loss': vq_loss, 'recons_loss': recons_loss, 'align_loss': align_loss}

    @t.no_grad()
    def dump_codes(self, path):
        """Full-depth codes of every user and item (rows = FACE ids) after the stage."""
        self.is_training = False
        u, i = self.forward(self.adj, 1.0)
        out = {}
        for side, e in (('user', u), ('item', i)):
            C = []
            for s in range(0, e.shape[0], 1024):
                _, z = self.vqraf.encode_aspects(e[s:s + 1024])
                C.append(self.vqraf.residual_quantizer(z, max_depth=self.H)['token_indices'].cpu().numpy())
            out[f'{side}_codes'] = np.concatenate(C)
        np.savez(path, **out); print(f'codes -> {path}', {k: v.shape for k, v in out.items()})
