"""CUIT Module A (RPG repo) — collaborative-context labels θ_e from sequential transitions. Ported from
InATTo_1/scripts/cuit_context_labels.py (same SASRec / Procrustes / spherical k-means / τ / G1 procedure); data come
from the RPG AmazonReviews2014 dataset (same LOO split as RPG / DiffGRM), folds = genrec.dataset.user_fold_index.

1. Users split into 5 folds (md5(f"{seed}:{raw_user_id}") % 5).
2. For each fold k, a SASRec (2 blocks, d 64, 2 heads, max_len 50, dropout 0.5, Adam 1e-3, batch 256, full-softmax
   next-item CE) is trained on the training parts of the users OUTSIDE fold k; early stopping on those users'
   validation NDCG@10 (patience 10).
3. Every training-part event of a fold-k user, e = (h = i_{<t}, i_t) with t ≥ 2 (1-based), gets the state of model k
   at the last history position (target not included).
4. Fold models are aligned to fold 0 by orthogonal Procrustes on item embeddings (items seen ≥ 5 times in both
   models' training data); states are rotated, L2-normalised, and clustered once by spherical k-means (Z = 256,
   3 restarts, fixed seed).
5. θ_e(z) = softmax_z(cos(b_e, μ_z) / τ); τ ∈ {0.02, 0.05, 0.1, 0.2, 0.5} maximises the held-out log-likelihood of an
   item-level null g_i (Dirichlet-smoothed mean θ of item i's events) on an inner 80/20 user split, scored on the
   held-out events' cluster assignments. Recommendation metrics are not used for τ.
6. G1: rank the next item from θ alone (score_i = Σ_z θ_e(z) P(i | z), P from the 80 % users' soft counts) on the
   20 % users' events; compare NDCG@10 / Recall@10 with popularity. Pass if ≥ +20 %.

Outputs <out>/events.npz (user, fold, t, item, theta float16 [E, Z]), centers.npy, meta.json (τ, G1, SASRec runs).
(events.parquet in the spec; pyarrow is not in the environment, so npz.)

    .venv/bin/python scripts/cuit_context.py --category=<Category> --out=results/cuit_rq/context [--folds=5]
"""

from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
TAUS = (0.02, 0.05, 0.1, 0.2, 0.5)


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def user_folds(user2id: dict, n_folds: int, seed: int) -> np.ndarray:
    fold = np.zeros(len(user2id), dtype=np.int64)
    for raw, uid in user2id.items():
        fold[uid] = int(hashlib.md5(f"{seed}:{raw}".encode()).hexdigest(), 16) % n_folds
    return fold


class SASRec(nn.Module):
    """SASRec (Kang & McAuley 2018), left padding, item 0 = pad. Same implementation as the verified teacher in
    GRID/suffid/intent.py (test R@10 0.0823 on that category there): Xavier init (N(0,1) embeddings make the initial logits
    ~d and training crawls) and a finite attention mask so left-padded query rows do not produce NaN."""

    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2, n_heads: int = 2, max_len: int = 50,
                 dropout: float = 0.5):
        super().__init__()
        self.d, self.num_heads, self.max_len = d, n_heads, max_len
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList([nn.LayerNorm(d, eps=1e-8) for _ in range(n_blocks)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
                                   for _ in range(n_blocks)])
        self.ffn_ln = nn.ModuleList([nn.LayerNorm(d, eps=1e-8) for _ in range(n_blocks)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d),
                                                nn.Dropout(dropout)) for _ in range(n_blocks)])
        self.last_ln = nn.LayerNorm(d, eps=1e-8)
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq):
        L = seq.shape[1]
        h = self.item_emb(seq) * self.d ** 0.5 + self.pos_emb(torch.arange(L, device=seq.device))
        pad = seq == 0
        h = self.drop(h) * (~pad).unsqueeze(-1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        mask = (causal.unsqueeze(0) | pad.unsqueeze(1)).float() * -1e9
        mask = mask.repeat_interleave(self.num_heads, dim=0)
        for ln, attn, ln2, ffn in zip(self.attn_ln, self.attn, self.ffn_ln, self.ffn):
            q = ln(h)
            h = q + attn(q, h, h, attn_mask=mask, need_weights=False)[0]
            h = ln2(h)
            h = h + ffn(h)
            h = h * (~pad).unsqueeze(-1)
        return self.last_ln(h)

    def item_weights(self):
        return self.item_emb.weight[1:]


def _left_pad(seqs, L):
    out = np.zeros((len(seqs), L), dtype=np.int64)
    for r, s in enumerate(seqs):
        s = list(s)[-L:]
        if s:
            out[r, -len(s):] = s
    return out


def train_sasrec(train_seqs, val_pairs, n_items, device, seed, max_len=50, epochs=200, patience=10, batch=256,
                 log=print):
    """train_seqs: list of item-id lists (training parts); val_pairs: list of (history, target). Returns model."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = SASRec(n_items, max_len=max_len).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.98))
    seqs = [np.asarray(s) + 1 for s in train_seqs if len(s) >= 2]
    best, best_state, bad, ep = -1.0, None, 0, 0
    vx = torch.as_tensor(_left_pad([np.asarray(h) + 1 for h, _ in val_pairs], max_len), device=device)
    vt = torch.as_tensor([t for _, t in val_pairs], device=device)
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(len(seqs))
        for b in range(0, len(perm), batch):
            chunk = [seqs[i][-(max_len + 1):] for i in perm[b:b + batch]]
            x = torch.as_tensor(_left_pad([c[:-1] for c in chunk], max_len), device=device)
            y = torch.as_tensor(_left_pad([c[1:] for c in chunk], max_len), device=device)
            logits = m(x) @ m.item_weights().T
            loss = F.cross_entropy(logits[y > 0], y[y > 0] - 1)
            assert torch.isfinite(loss), 'non-finite SASRec loss'
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            nd = []
            for b in range(0, len(vx), 1024):
                sc = m(vx[b:b + 1024])[:, -1] @ m.item_weights().T
                rank = (sc > sc.gather(1, vt[b:b + 1024, None])).sum(1) + 1
                nd.append(torch.where(rank <= 10, 1.0 / torch.log2(rank.float() + 1), torch.zeros_like(rank, dtype=torch.float)))
            ndcg = torch.cat(nd).mean().item()
        if ndcg > best:
            best, bad, best_state = ndcg, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    m.load_state_dict(best_state)
    m.eval()
    log(f"    SASRec: best valid NDCG@10 {best:.4f} after {ep + 1} epochs")
    return m, best, ep + 1


@torch.no_grad()
def event_states(m, train_seqs, users, max_len=50):
    """States for events t >= 2 (1-based) of the given users: (user, t, item, state)."""
    U, T, I, S = [], [], [], []
    for u in users:
        s = np.asarray(train_seqs[u])
        if len(s) < 2:
            continue
        for t in range(1, len(s)):                                 # 0-based target index; history s[:t]
            U.append(u); T.append(t + 1); I.append(int(s[t]))
            S.append(_left_pad([s[:t] + 1], max_len)[0])
    states = []
    dev = next(m.parameters()).device
    S = np.stack(S)
    for b in range(0, len(S), 2048):
        states.append(m(torch.as_tensor(S[b:b + 2048], device=dev))[:, -1].cpu())
    return np.asarray(U), np.asarray(T), np.asarray(I), torch.cat(states).numpy()


def procrustes(src, dst):
    u, _, vt = np.linalg.svd(src.T @ dst)
    return u @ vt


def spherical_kmeans(X, Z, seed, restarts=3, iters=50, device="cpu"):
    X = F.normalize(torch.as_tensor(X, device=device, dtype=torch.float32), dim=1)
    best, best_c = -np.inf, None
    for r in range(restarts):
        g = torch.Generator(device="cpu").manual_seed(seed + r)
        C = X[torch.randperm(len(X), generator=g)[:Z].to(device)].clone()
        for _ in range(iters):
            a = (X @ C.T).argmax(1)
            Cn = torch.zeros_like(C).index_add_(0, a, X)
            empty = Cn.norm(dim=1) == 0
            Cn[empty] = C[empty]
            C = F.normalize(Cn, dim=1)
        obj = (X @ C.T).max(1).values.sum().item()
        if obj > best:
            best, best_c = obj, C
    return best_c.cpu().numpy(), best


def soft_theta(X, C, tau, device="cpu"):
    X = F.normalize(torch.as_tensor(X, device=device, dtype=torch.float32), dim=1)
    return torch.softmax((X @ torch.as_tensor(C, device=device).T) / tau, dim=1).cpu().numpy()


def item_null(theta, items, n_items, alpha=1.0):
    Z = theta.shape[1]
    g = np.zeros((n_items, Z)); np.add.at(g, items, theta)
    n = np.bincount(items, minlength=n_items)[:, None]
    return (g + alpha / Z) / (n + alpha)


def main():
    from accelerate import Accelerator
    from genrec.utils import get_config, get_dataset, parse_command_line_args
    from genrec.dataset import user_fold_index
    args = parse_command_line_args(sys.argv[1:])
    sys.argv = ["cuit_context"]
    out = Path(args.pop("out")); folds = int(args.pop("folds", 5)); Z = int(args.pop("Z", 256))
    run_folds = args.pop("run_folds", None); max_epochs = int(args.pop("max_epochs", 200)); seed = int(args.pop("seed", 2024))
    cfg = get_config("RPG", "AmazonReviews2014", None, dict(args, log_dir=str(out / "logs") + "/"))
    cfg["accelerator"] = Accelerator()
    ds = get_dataset("AmazonReviews2014")(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    users = list(ds.all_item_seqs)
    # 0-based item index = RPG item id − 1 (RPG id 0 is padding)
    hist = [np.asarray([ds.item2id[x] - 1 for x in ds.all_item_seqs[u][:-2]], dtype=np.int64) for u in users]
    val_t = {i: ds.item2id[ds.all_item_seqs[u][-2]] - 1 for i, u in enumerate(users) if len(ds.all_item_seqs[u]) > 2}
    n_items = ds.n_items - 1
    fold = np.array([user_fold_index(u, folds, seed) for u in users])
    out.mkdir(parents=True, exist_ok=True)
    meta = {"folds": folds, "seed": seed, "Z": Z, "sasrec": {}}
    allU, allT, allI, allS, allF, emb, counts = [], [], [], [], [], {}, {}
    for k in range(int(run_folds) if run_folds else folds):
        t0 = time.time()
        tr_users = [u for u in range(len(hist)) if fold[u] != k]
        print(f"[fold {k}] SASRec on {len(tr_users)} users outside fold {k}", flush=True)
        m, best, eps = train_sasrec([hist[u] for u in tr_users], [(hist[u], val_t[u]) for u in tr_users if u in val_t],
                                    n_items, dev, seed + k, epochs=max_epochs)
        meta["sasrec"][k] = {"valid_ndcg10": best, "epochs": eps, "seconds": time.time() - t0}
        U, T, I, S = event_states(m, hist, [u for u in range(len(hist)) if fold[u] == k])
        allU.append(U); allT.append(T); allI.append(I); allS.append(S); allF.append(np.full(len(U), k))
        emb[k] = m.item_weights().detach().cpu().numpy()
        counts[k] = np.bincount(np.concatenate([hist[u] for u in tr_users if len(hist[u])]), minlength=n_items)
    fr = sorted(emb)
    for k in fr[1:]:
        shared = (counts[k] >= 5) & (counts[fr[0]] >= 5)
        allS[fr.index(k)] = allS[fr.index(k)] @ procrustes(emb[k][shared], emb[fr[0]][shared])
        meta.setdefault("procrustes_items", {})[k] = int(shared.sum())
    U, T, I, S, Fo = map(np.concatenate, (allU, allT, allI, allS, allF))
    C, obj = spherical_kmeans(S, Z, seed, device=dev)
    inner = np.array([int(hashlib.md5(f"tau:{seed}:{users[u]}".encode()).hexdigest(), 16) % 5 == 0 for u in U])
    hard = (F.normalize(torch.as_tensor(S, dtype=torch.float32), dim=1) @ torch.as_tensor(C).T).argmax(1).numpy()
    ll = {}
    for tau in TAUS:
        th = soft_theta(S[~inner], C, tau, dev)
        g = item_null(th, I[~inner], n_items)
        ll[tau] = float(np.log(g[I[inner], hard[inner]]).mean())
    tau = max(ll, key=ll.get)
    theta = soft_theta(S, C, tau, dev)
    tr, te = ~inner, inner
    Pz = np.zeros((Z, n_items)); np.add.at(Pz.T, I[tr], theta[tr]); Pz = (Pz + 1e-9) / (Pz.sum(1, keepdims=True) + 1e-9 * n_items)
    pop = np.bincount(I[tr], minlength=n_items).astype(float)

    def nd_rc(scores, tgt):
        r = (scores > scores[np.arange(len(tgt)), tgt][:, None]).sum(1) + 1
        return float(np.where(r <= 10, 1 / np.log2(r + 1), 0).sum()), float((r <= 10).sum())
    te_idx = np.where(te)[0]; acc = np.zeros(4)
    for b in range(0, len(te_idx), 4096):
        ix = te_idx[b:b + 4096]
        acc += np.array(nd_rc(theta[ix] @ Pz, I[ix]) + nd_rc(np.broadcast_to(pop, (len(ix), n_items)), I[ix]))
    nd_t, rc_t, nd_p, rc_p = acc / max(len(te_idx), 1)
    meta.update({"tau": tau, "tau_loglik": ll, "kmeans_objective": obj, "n_events": int(len(U)),
                 "G1": {"theta_ndcg10": nd_t, "theta_recall10": rc_t, "pop_ndcg10": nd_p, "pop_recall10": rc_p,
                        "ndcg_gain": nd_t / max(nd_p, 1e-12) - 1, "pass": bool(nd_t >= 1.2 * nd_p)}})
    # events: user = row index into ds.all_item_seqs order, item = RPG item id (1-based)
    np.savez(out / "events.npz", user=U, fold=Fo, t=T, item=I + 1, theta=theta.astype(np.float16))
    np.save(out / "centers.npy", C)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps({k: meta[k] for k in ("tau", "n_events", "G1")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
