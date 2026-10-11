"""Step 8: depth by an information-exhaustion test (quantize until the next codeword stops carrying information).

For entity e (user or item), all aspects start at depth 1 (the first codeword is always used, as in FACE). At round h
(h = 1..H-1) every aspect a that is still active has state s = current prefixes of all aspects of e and a candidate
next codeword t = c_{e,a}^{(h+1)} (the residual quantization's next code). With the cross-fitted joint probe (s6):
    S_real   = log p(y_e | s + t) - log p(y_e | s)
    S_null,r = log p(y_e | s + t~_r) - log p(y_e | s),   t~_r ~ codewords used at (aspect a, depth h+1) on this side
    p        = (1 + #{r : S_null,r >= S_real}) / (R + 1)
Continue (quantize the next level) iff p <= alpha; otherwise the aspect stops: its next codeword is no more informative
about e's behaviour than an arbitrary codeword. All aspects of all entities are decided in parallel within a round and
the state is re-measured every round. No token budget.
Writes ids/depth_perm_{side}.npz (depth [n_side, A]) and results/perm_{side}_log.json.
"""
import argparse, json, os, sys
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('--side', default='item'); ap.add_argument('--R', type=int, default=19); ap.add_argument('--alpha', type=float, default=0.05)
ap.add_argument('--art', default=os.environ.get('V3F_ART', '')); ap.add_argument('--bs', type=int, default=512)
ap.add_argument('--out_dir', default='ids'); ap.add_argument('--probe', default='s6_joint_probe'); ap.add_argument('--log_tag', default='')
ap.add_argument('--group', default='entity')  # entity | prefix (nearest ancestor cell with >= min_cell candidates)   # entity | prefix (cell-level split test, pooled over entities sharing the prefix)
ap.add_argument('--min_cell', type=int, default=5); ap.add_argument('--H', type=int, default=8)
args = ap.parse_args()
sys.argv = [sys.argv[0], f'--side={args.side}', f'--art={args.art}', f'--H={args.H}']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib  # noqa: E402
J = importlib.import_module(args.probe)

dev = 'cuda'; A, H, NU, N = J.A, J.H, J.NU, J.N
rows = np.arange(NU, N) if args.side == 'item' else np.arange(NU)
y = J.y_all; rng = np.random.default_rng(2023)
models = {}
for f in (0, 1):
    m = J.V().to(dev); m.load_state_dict(torch.load(f'{args.art}/joint_models/{args.side}_fold{f}.pt')); m.eval(); models[f] = m
codes = J.tok_idx                                                     # [N, A, H] (shared with the probe's text builder)
pool = {(k, h): codes[rows, k, h] for k in range(A) for h in range(H)}


def gold_logp(ents, deps, override=None):
    """log p(y_e | state). override: (aspect idx, codeword idx) per row replacing the code at the extended level."""
    out = np.zeros(len(ents), np.float32)
    saved = None
    if override is not None:
        ka, lev, cw = override
        saved = codes[ents, ka, lev].copy(); codes[ents, ka, lev] = cw
    try:
        for f in (0, 1):
            sel = np.where(J.fold[ents] == f)[0]
            for s in range(0, len(sel), args.bs):
                b = sel[s:s + args.bs]
                lp = J.logits(models[f], ents[b], deps[b], bs=args.bs)
                out[b] = lp.gather(1, y[ents[b]][:, None]).squeeze(1).numpy()
    finally:
        if saved is not None:
            codes[ents, ka, lev] = saved
    return out


d = np.ones((N, A), np.int64); active = np.ones((N, A), bool); log = []
for h in range(1, H):                                                 # decide level h+1 (0-based index h)
    ci, ca = np.nonzero(active[rows] & (d[rows] == h))
    if len(ci) == 0:
        break
    ents = rows[ci]
    base = gold_logp(ents, d[ents])
    ext = d[ents].copy(); ext[np.arange(len(ci)), ca] += 1
    real = gold_logp(ents, ext) - base
    null = np.zeros((args.R, len(ci)), np.float32)
    for r in range(args.R):
        cw = np.array([pool[(k, h)][rng.integers(len(rows))] for k in ca])
        null[r] = gold_logp(ents, ext, override=(ca, h, cw)) - base
    p = (1 + (null >= real[None]).sum(0)) / (args.R + 1)
    go = p <= args.alpha
    if args.group == 'prefix':
        # A cell = (aspect, prefix c^(1..h)). Per-entity gains are too small to detect one by one (continue share = alpha),
        # so the split decision is made per cell: continue iff the mean excess gain x_e = S_real - mean_r S_null over the
        # cell's entities is > 0 by a one-sided t-test at alpha. Cells smaller than min_cell are pooled per (aspect).
        from scipy import stats
        x = real - null.mean(0)
        # Backoff (v3b): a cell with < min_cell candidates is tested together with its siblings, i.e. in its nearest
        # ancestor cell (aspect, c^(1..l)), l < h, that holds >= min_cell candidates (l = 0: the whole aspect). With
        # per-aspect pooling (v3a) every decision from level 4 on was aspect-wide because deep prefixes are near-unique.
        cid = np.full(len(ci), -1, np.int64); nxt = 0
        for l in range(h, -1, -1):
            todo = cid < 0
            if not todo.any():
                break
            key = np.concatenate([ca[:, None], codes[ents, ca, :l]], 1)
            _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel()
            # count only still-unassigned candidates so that an ancestor is not inflated by already-tested descendants
            cnt_todo = np.bincount(inv[todo], minlength=cnt.size)
            ok = todo & ((cnt_todo[inv] >= args.min_cell) | (l == 0))
            cid[ok] = nxt + inv[ok]; nxt += cnt.size
        go = np.zeros(len(ci), bool)
        for c in np.unique(cid):
            m = cid == c
            if m.sum() >= 2 and stats.ttest_1samp(x[m], 0, alternative='greater').pvalue <= args.alpha:
                go[m] = True
        np.savez(f'{args.out_dir}/perm_raw_{args.side}_L{h + 1}.npz', real=real, null=null, ents=ents, ca=ca)
    d[ents[go], ca[go]] += 1
    active[ents[~go], ca[~go]] = False
    rec = {'level': h + 1, 'candidates': int(len(ci)), 'continue': int(go.sum()), 'share_continue': float(go.mean()),
           'mean_S_real': float(real.mean()), 'mean_S_null': float(null.mean()), 'mean_depth': float(d[rows].mean())}
    log.append(rec); print(rec, flush=True)
os.makedirs(args.out_dir, exist_ok=True)
np.savez(f'{args.out_dir}/depth_perm_{args.side}.npz', depth=d[rows], rows=rows)
json.dump(log, open(f'results/perm_{args.side}{args.log_tag}_log.json', 'w'), indent=1)
print(f'{args.side} depth hist', np.bincount(d[rows].ravel(), minlength=H + 1)[1:].tolist(), 'mean', round(float(d[rows].mean()), 3))
