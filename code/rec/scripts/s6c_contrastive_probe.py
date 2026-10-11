"""Step 6c: state-conditional CONTINUOUS probe (von Mises-Fisher family; first tried: softmax over all N entities, which underfit
(dev CE above log N after 3 epochs))

vMF version (used): p(v_e | s) = C_d(kappa) exp(kappa q(s)^T v_e), q(s) = normalize(W BERT(text(s))), kappa global (MLE on dev,
Banerjee approximation). log C_d(kappa) cancels in S = log p(v|s+t) - log p(v|s), so S = kappa (cos(s+t) - cos(s)).
Training maximises cos(q, v_e).

Old description: (fine-grained behavioural target; replaces the 64-cluster CE of s6).

Why: with y_e = one of G = 64 behaviour clusters, the measurable information is capped at log 64 and a depth-1 ID that
already separates almost every entity leaves nothing for deeper words to add (continue rate = alpha at every depth).
Here the class of entity e is e itself, represented by its cross-fitted SASRec behaviour vector v_e (G = N_side):
    q(s)            = normalize(W BERT(text(s)))                         (64-d)
    log p(e | s)    = q(s)^T v_e / tau  -  logsumexp_{e' in side} q(s)^T v_e' / tau      (tau learned)
i.e. the same V-information probe with each entity's behaviour vector as a fixed class prototype; nearby behaviours share
probability mass, so finer refinements of the ID can register. Input text, random-depth training, 2-fold cross-fitting,
optimiser and early stopping are identical to s6. Exposes the s6 interface (V, logits, y_all, tok_idx, fold, A, H, NU, N)
so s8_perm_depth.py --probe=s6c_contrastive_probe runs the same permutation test.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ap = argparse.ArgumentParser()
ap.add_argument('--art', default=os.environ.get('V3F_ART', '')); ap.add_argument('--H', type=int, default=8)
ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--patience', type=int, default=1)
ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=3e-5)
ap.add_argument('--max_len', type=int, default=256); ap.add_argument('--seed', type=int, default=2023)
ap.add_argument('--side', default='item')
ap.add_argument('--head_lr', type=float, default=1e-3)
ap.add_argument('--vecs', default=os.environ.get('BEHAVIOR_VECS', ''))
a, _ = ap.parse_known_args()
dev = 'cuda'
R = torch.load(f'{a.art}/rq_H{a.H}.pt', weights_only=False)
tok_idx, NU, vocab = R['token_indices'].numpy(), R['n_user'], R['vocab']
N, A, H = tok_idx.shape
VEC = torch.tensor(np.load(a.vecs), dtype=torch.float32)                      # [N, 64], unit rows
assert VEC.shape[0] == N
LO, HI = (NU, N) if a.side == 'item' else (0, NU)
PROTO = VEC[LO:HI].to(dev)                                                   # class prototypes of this side
y_all = torch.zeros(N, dtype=torch.long)                                    # s8 gathers column 0 of logits()
tk = AutoTokenizer.from_pretrained('bert-base-uncased')
rng = np.random.default_rng(a.seed); fold = np.zeros(N, np.int64)
for lo, n in ((0, NU), (NU, N - NU)):                                        # identical to s6 / s2_vinfo.py
    p = rng.permutation(n) + lo; fold[p[n // 2:]] = 1


def state_text(r, d):
    return 'Codewords: ' + ' | '.join(f'a{k}: ' + ' '.join(vocab[int(t)] for t in tok_idx[r, k, :d[k]]) for k in range(A))


def enc(rows, deps):
    return tk([state_text(r, d) for r, d in zip(rows, deps)], truncation=True, max_length=a.max_len, padding=True, return_tensors='pt').to(dev)


class V(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased'); self.head = nn.Linear(768, PROTO.shape[1])
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(0.07))))

        self.register_buffer('kappa', torch.tensor(1.0))

    def forward(self, e):
        return F.normalize(self.head(self.bert(**e).pooler_output).float(), dim=-1)    # q(s), unit [B, 64]


@torch.no_grad()
def cosines(model, rows, deps, bs=256):
    model.eval(); out = []
    for s in range(0, len(rows), bs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            q = model(enc(rows[s:s + bs], deps[s:s + bs])).float()
        out.append((q * VEC[rows[s:s + bs]].to(dev)).sum(-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def logits(model, rows, deps, bs=256):
    """s8 interface: [n, 1] = kappa * cos(q(s), v_e) = log p(v_e | s) up to the state-independent log C_d(kappa); y = 0."""
    return (model.kappa.cpu() * cosines(model, rows, deps, bs))[:, None]


def kappa_mle(r, d=64):
    return float(r * (d - r ** 2) / max(1 - r ** 2, 1e-6))                   # Banerjee et al. (2005)


if __name__ == '__main__':
    os.makedirs(f'{a.art}/joint_models', exist_ok=True); log = {}
    rows_side = np.arange(LO, HI)
    for f in (0, 1):
        tr_all = rows_side[fold[rows_side] != f]; g = np.random.default_rng(a.seed + f); tr_all = g.permutation(tr_all)
        nd = len(tr_all) // 20; dv, tr = tr_all[:nd], tr_all[nd:]
        dv_d = g.integers(0, H + 1, (len(dv), A))
        torch.manual_seed(a.seed + f); model = V().to(dev)
        # BERT at the s6 rate; the new projection head and the temperature at head_lr (with lr 3e-5 tau stayed at its init
        # and the probe underfit: dev CE 9.22 vs log N 9.40 after 3 epochs)
        hp = [p for n, p in model.named_parameters() if not n.startswith('bert.')]
        opt = torch.optim.AdamW([{'params': model.bert.parameters(), 'lr': a.lr}, {'params': hp, 'lr': a.head_lr}], weight_decay=0.01)
        best, bad, state = 1e9, 0, None
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); perm = g.permutation(tr); deps = g.integers(0, H + 1, (len(perm), A)); tot = 0.0
            for s in range(0, len(perm), a.bs):
                r = perm[s:s + a.bs]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    lg = model(enc(r, deps[s:s + a.bs]))
                loss = 1 - (lg.float() * VEC[r].to(dev)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item() * len(r)
            rc = cosines(model, dv, dv_d).mean().item(); dce = -rc
            print(f'[fold {f}] epoch {ep + 1} train loss {tot / len(perm):.4f} dev mean cos {rc:.4f} (kappa {kappa_mle(rc):.1f}, '
                  f'{time.time() - t0:.0f}s)', flush=True)
            if dce < best - 1e-4:
                best, bad, state = dce, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad > a.patience:
                    break
        state['kappa'] = torch.tensor(kappa_mle(-best))
        torch.save(state, f'{a.art}/joint_models/{a.side}_fold{f}.pt'); log[f'{a.side}_fold{f}'] = {'dev_mean_cos': -best, 'kappa': kappa_mle(-best), 'epochs': ep + 1}
        del model, opt; torch.cuda.empty_cache()
    json.dump(log, open(f'{a.art}/joint_log_{a.side}.json', 'w'), indent=1)
