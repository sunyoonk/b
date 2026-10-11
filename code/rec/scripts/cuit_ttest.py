"""Paired t-test over test users (same users, same targets) for per-user Recall@5, Recall@10, NDCG@10.

Reads <dir>/<name><suffix>_peruser.npz (recall5, recall10, ndcg10, target) or an explicit npz path.
    .venv/bin/python scripts/cuit_ttest.py --pairs=A:B,C:D --dir=results/... --suffix=_final
    .venv/bin/python scripts/cuit_ttest.py --files=a.npz:b.npz --names=A:B
Reports mean A, mean B, diff, 95% CI of the mean difference (t distribution), t, two-sided p, and n.
"""
import argparse
import json
import os

import numpy as np
from scipy import stats


def load(path):
    z = np.load(path)
    return {k: z[k].astype(float) for k in ('recall5', 'recall10', 'ndcg10')}, z['target']


def ttest(a, b):
    d = a - b; n = len(d)
    t, p = stats.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n); q = stats.t.ppf(0.975, n - 1)
    return {'mean_a': a.mean(), 'mean_b': b.mean(), 'diff': d.mean(), 'ci95': [d.mean() - q * se, d.mean() + q * se],
            't': float(t), 'p': float(p), 'n': int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default=None); ap.add_argument('--dir', default='.'); ap.add_argument('--suffix', default='_final')
    ap.add_argument('--files', default=None); ap.add_argument('--names', default=None); ap.add_argument('--out', default=None)
    args = ap.parse_args()
    jobs = []
    if args.files:
        fa, fb = args.files.split(':'); na, nb = (args.names or 'A:B').split(':'); jobs.append((na, nb, fa, fb))
    for pr in (args.pairs.split(',') if args.pairs else []):
        a, b = pr.split(':')
        jobs.append((a, b, os.path.join(args.dir, f'{a}{args.suffix}_peruser.npz'), os.path.join(args.dir, f'{b}{args.suffix}_peruser.npz')))
    res = {}
    for a, b, fa, fb in jobs:
        ma, ta = load(fa); mb, tb = load(fb)
        assert (ta == tb).all(), 'different users / targets'
        res[f'{a}:{b}'] = {}
        for k in ('recall5', 'recall10', 'ndcg10'):
            r = ttest(ma[k], mb[k]); res[f'{a}:{b}'][k] = r
            sig = '*' if r['p'] < 0.05 else ' '
            print(f"{a} vs {b} {k:9s} {r['mean_a']:.4f} vs {r['mean_b']:.4f}  diff {r['diff']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
                  f"  t {r['t']:+.2f}  p {r['p']:.3f} {sig}")
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
