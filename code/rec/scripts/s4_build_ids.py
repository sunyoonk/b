"""Step 4a: adaptive textual IDs -> RPG semantic-ID files for the parallel recommender.

Slots: s = a * H + (h - 1) for aspect a (16) and depth h (1..H), plus one dedup slot (index A*H). Every slot has a local
vocabulary: the FACE words that occur at that slot over all users and items (depth H codes), remapped to 0..K_s-1;
codebook_size K = max_s K_s (shared by every run so that runs differ only in their masks).
Kept slots of entity e: aspect a, depths 1..d_{e,a}; dedup slot only for items whose kept tuple collides.

Inputs: <art>/rq_H{H}.pt, depth file (npz: depth [N_entities, A], rows = users then items) or --fixed=h.
Writes  cache/.../flan-t5-xl_pca0/rqk_L{A*H+1}_seed{tag}.sem_ids  (item ASIN -> codes),
        <out>/mask_{name}.npz (items, mask), <out>/users_{name}.npz (users, tokens, mask), <out>/ids_{name}.json (stats).
"""
import argparse, json, os
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--art', default=os.environ.get('V3F_ART', 'artifacts')); ap.add_argument('--H', type=int, default=4)
ap.add_argument('--depth', default=None); ap.add_argument('--fixed', type=int, default=None)
ap.add_argument('--name', required=True); ap.add_argument('--tag', type=int, required=True); ap.add_argument('--out', required=True)
ap.add_argument('--cache', default=f"cache/AmazonReviews2014/{os.environ.get('CATEGORY', '')}/processed/flan-t5-xl_pca0")
ap.add_argument('--cf2_from', default=None)
ap.add_argument('--content_from', default=None)   # sem_ids of the B-family tokenizer: its first --content_L content codes are prepended as item slots
ap.add_argument('--content_L', type=int, default=0)
ap.add_argument('--items_vocab', type=int, default=0)   # sem_ids file whose last 2 codes are SASRec CF codes (B family): appended as 2 item slots
a = ap.parse_args()
R = torch.load(f'{a.art}/rq_H{a.H}.pt', weights_only=False)
tok = R['token_indices'].numpy(); ids, NU = R['ids'], R['n_user']
N, A, H = tok.shape
S = A * H
codes = tok.reshape(N, S)                                            # slot s = a*H + h-1
local = np.zeros_like(codes)
sizes, uniq = [], []
for s in range(S):
    if a.items_vocab:   # slot vocabulary from items only (user codes outside it -> code 0; users unused by the recommender)
        u = np.unique(codes[NU:, s]); pos = np.searchsorted(u, codes[:, s]).clip(0, len(u) - 1)
        local[:, s] = np.where(u[pos] == codes[:, s], pos, 0)
    else:
        u, inv = np.unique(codes[:, s], return_inverse=True); local[:, s] = inv
    sizes.append(len(u)); uniq.append(u)
K = 1 << int(np.ceil(np.log2(max(sizes))))   # power of two (RPG tokenizer requirement); unused codes stay idle
if a.fixed:
    dep = np.full((N, A), a.fixed)
else:
    dep = np.load(a.depth)['depth']
assert dep.shape == (N, A) and dep.min() >= 1 and dep.max() <= H
keep = (np.arange(H)[None, None, :] < dep[:, :, None]).reshape(N, S)  # [N, S]
# items: dedup over the kept tuple
items = list(range(NU, N))
key = {r: tuple(np.where(keep[r], local[r], -1).tolist()) for r in items}
seen, dd = {}, {}
for r in items:
    dd[r] = seen.get(key[r], 0); seen[key[r]] = dd[r] + 1
assert max(dd.values()) < K
cf = None; P = a.content_L if a.content_from else 0
pre = {}
if P:
    src_c = json.load(open(a.content_from)); pre = {x: list(map(int, src_c[x][:P])) for x in src_c}
if a.cf2_from:
    src = json.load(open(a.cf2_from)); cf = {x: src[x][-2:] for x in src}
sem = {ids[r]: (pre[ids[r]] if P else []) + [int(v) if k else 0 for v, k in zip(local[r], keep[r])] + [int(dd[r])] + (list(map(int, cf[ids[r]])) if cf else []) for r in items}
os.makedirs(a.out, exist_ok=True)
E = 2 if cf else 0                                                    # extra CF slots
json.dump(sem, open(f'{a.cache}/rqk_L{P + S + 1 + E}_seed{a.tag}.sem_ids', 'w'))
im = np.concatenate([np.ones((len(items), P), bool), keep[NU:], np.array([[seen[key[r]] > 1] for r in items]), np.ones((len(items), E), bool)], 1)
np.savez(f'{a.out}/mask_{a.name}.npz', items=np.asarray([ids[r] for r in items]), mask=im)
um = np.concatenate([np.zeros((NU, P), bool), keep[:NU], np.zeros((NU, 1 + E), bool)], 1)
ut = np.concatenate([np.zeros((NU, P), np.int64), np.where(keep[:NU], local[:NU], 0), np.zeros((NU, 1 + E), np.int64)], 1)
gw = np.concatenate([np.zeros((NU, P), np.int64), np.where(keep[:NU], codes[:NU] + 1, 0), np.zeros((NU, 1 + E), np.int64)], 1)
np.savez(f'{a.out}/users_{a.name}.npz', users=np.asarray(ids[:NU]), tokens=ut, mask=um, gwords=gw)
# shared-vocabulary map (RPG token -> one global row): word slots -> FACE word index + 1 (same word, same row at every
# aspect / depth); dedup slot and CF slots get their own rows after the vocabulary; unused codes -> row 0
V = int(codes.max()) + 1
gmap = np.zeros((P + S + 1 + E, K), np.int64)
base = V + 1 + K * (1 + E)
for j in range(P):                                                   # content levels: own rows per level
    gmap[j] = base + K * j + np.arange(K)
for s_ in range(S):
    gmap[P + s_, :len(uniq[s_])] = uniq[s_] + 1
gmap[P + S] = V + 1 + np.arange(K)
for j in range(E):
    gmap[P + S + 1 + j] = V + 1 + K * (1 + j) + np.arange(K)
np.savez(f'{a.out}/gmap_{a.name}.npz', gmap=gmap)
st = {'name': a.name, 'tag': a.tag, 'n_digit': P + S + 1 + E, 'codebook_size': K, 'slot_vocab': sizes,
      'item_tokens_mean': float(im.sum(1).mean()), 'user_tokens_mean': float(um.sum(1).mean()),
      'item_depth_mean': float(dep[NU:].mean()), 'user_depth_mean': float(dep[:NU].mean()),
      'item_depth_hist': np.bincount(dep[NU:].ravel(), minlength=H + 1)[1:].tolist(),
      'user_depth_hist': np.bincount(dep[:NU].ravel(), minlength=H + 1)[1:].tolist(),
      'colliding_items': int(sum(seen[key[r]] > 1 for r in items))}
json.dump(st, open(f'{a.out}/ids_{a.name}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in st.items() if k != 'slot_vocab'}))
