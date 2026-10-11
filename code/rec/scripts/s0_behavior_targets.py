"""Step 0 (v2 target): behavioural collaborative targets y_e that do NOT depend on the FACE aspects z.

Cross-fitted SASRec (user folds md5(seed:user) mod 5, seed 2024; training parts s[:-2] only):
  model k is trained on users outside fold k and encodes the users of fold k.
  - purchase-time state of event (u, t): SASRec state of the history i_{<t} that precedes the target i_t (t >= 2)
  - user state: SASRec state of the whole training history s[:-2] (the state that predicts the validation item;
    the validation item itself is never used)
  Fold spaces are aligned to fold 0 by orthogonal Procrustes on the item embeddings of items seen >= 5 times in both.
Targets (K = 64, spherical k-means, fitted on training entities = cuit1 split_entities 90%, seed 2023):
  item y_i = cluster of the mean (L2-normalised) purchase-time state over the events whose target is i
  user y_u = cluster of the user's state
The same y_e is used for all 16 aspects. Writes <art>/labels_behavior.npy [N, 16] int64 (rows users then items, maps.pkl
order) and labels_behavior.json (stats).
"""
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuit_context import train_sasrec, event_states, procrustes, spherical_kmeans, _left_pad  # noqa: E402

ART = os.environ.get('S0_ART', 'artifacts/behavior'); K = 64; FOLDS = 5; SEED = 2024


def main():
    from accelerate import Accelerator
    from genrec.utils import get_config, get_dataset
    from genrec.dataset import user_fold_index
    sys.argv = ['s0']
    cfg = get_config('RPG', 'AmazonReviews2014', None, {'category': os.environ['CATEGORY'], 'log_dir': '/tmp/s0_logs/'})
    cfg['accelerator'] = Accelerator()
    ds = get_dataset('AmazonReviews2014')(cfg)
    dev = torch.device('cuda')
    users = list(ds.all_item_seqs)
    hist = [np.asarray([ds.item2id[x] - 1 for x in ds.all_item_seqs[u][:-2]], dtype=np.int64) for u in users]
    val_t = {i: ds.item2id[ds.all_item_seqs[u][-2]] - 1 for i, u in enumerate(users) if len(ds.all_item_seqs[u]) > 2}
    n_items = ds.n_items - 1
    fold = np.array([user_fold_index(u, FOLDS, SEED) for u in users])
    ev_I, ev_S, us_S, emb, counts, meta = [], [], np.zeros((len(users), 64), np.float32), {}, {}, {'sasrec': {}}
    for k in range(FOLDS):
        t0 = time.time()
        tr = [u for u in range(len(users)) if fold[u] != k]; te = [u for u in range(len(users)) if fold[u] == k]
        m, best, eps = train_sasrec([hist[u] for u in tr], [(hist[u], val_t[u]) for u in tr if u in val_t], n_items, dev, SEED + k)
        U, T, I, S = event_states(m, hist, te)
        with torch.no_grad():
            X = _left_pad([hist[u] + 1 for u in te], 50)
            st = torch.cat([m(torch.as_tensor(X[b:b + 2048], device=dev))[:, -1].cpu() for b in range(0, len(X), 2048)]).numpy()
        emb[k] = m.item_weights().detach().cpu().numpy()
        counts[k] = np.bincount(np.concatenate([hist[u] for u in tr if len(hist[u])]), minlength=n_items)
        if k > 0:   # align fold k to fold 0
            sh = (counts[k] >= 5) & (counts[0] >= 5); Rk = procrustes(emb[k][sh], emb[0][sh])
            S, st = S @ Rk, st @ Rk
            meta.setdefault('procrustes_items', {})[k] = int(sh.sum())
        ev_I.append(I); ev_S.append(S); us_S[te] = st
        meta['sasrec'][k] = {'valid_ndcg10': best, 'epochs': eps, 'seconds': round(time.time() - t0)}
        print(f'[fold {k}] SASRec valid N@10 {best:.4f}, {len(I)} events', flush=True)
    I = np.concatenate(ev_I); S = F.normalize(torch.as_tensor(np.concatenate(ev_S)), dim=1).numpy()
    it_S = np.zeros((n_items, S.shape[1]), np.float32); np.add.at(it_S, I, S)
    n_ev = np.bincount(I, minlength=n_items)
    # rows of the entity table (cuit1 maps.pkl order): users sorted raw ids, then items sorted ASINs
    import pickle
    maps = pickle.load(open(os.environ['MAPS_PKL'], 'rb'))
    NU, NI = len(maps['users']), len(maps['items'])
    urow = np.array([users.index(u) for u in maps['users']]) if False else None
    upos = {u: i for i, u in enumerate(users)}
    U_vec = us_S[[upos[u] for u in maps['users']]]
    I_vec = it_S[[ds.item2id[a] - 1 for a in maps['items']]]
    I_n = n_ev[[ds.item2id[a] - 1 for a in maps['items']]]
    meta['items_without_events'] = int((I_n == 0).sum())
    # items never seen as a training target: fall back to their (aligned fold-0) SASRec item embedding
    if (I_n == 0).any():
        I_vec[I_n == 0] = emb[0][[ds.item2id[a] - 1 for a in np.asarray(maps['items'])[I_n == 0]]]
    if os.environ.get('VEC_OUT'):   # continuous behavioural targets v_e (L2-normalised), rows users then items (maps.pkl order)
        V = np.concatenate([U_vec, I_vec]).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True).clip(1e-12)
        np.save(os.environ['VEC_OUT'], V); print('saved behaviour vectors', V.shape, flush=True); return
    rng = np.random.default_rng(2023); va = []                     # cuit1 split_entities(val_frac 0.1, seed 2023)
    for lo, n in ((0, NU), (NU, NI)):
        p = rng.permutation(n) + lo; va.append(p[:int(round(n * 0.1))])
    is_val = np.zeros(NU + NI, bool); is_val[np.concatenate(va)] = True
    y = np.zeros(NU + NI, np.int64)
    for side, vec, lo in (('user', U_vec, 0), ('item', I_vec, NU)):
        rows = np.arange(lo, lo + len(vec)); trm = ~is_val[rows]
        C, obj = spherical_kmeans(vec[trm], K, 2023, device=dev)
        Xn = F.normalize(torch.as_tensor(vec, dtype=torch.float32), dim=1).numpy()
        y[rows] = (Xn @ C.T).argmax(1)
        sz = np.bincount(y[rows], minlength=K)
        meta[side] = {'cluster_size_min': int(sz.min()), 'cluster_size_median': float(np.median(sz)), 'cluster_size_max': int(sz.max()),
                      'entropy_bits': float(-(sz / sz.sum() * np.log2(np.clip(sz / sz.sum(), 1e-12, 1))).sum())}
    # agreement with the old (z-derived) labels, aspect 0, for reference
    if os.environ.get('OLD_Z_LABELS'):   # optional reference only (not used by V3F, which runs with VEC_OUT)
        old = np.load(os.environ['OLD_Z_LABELS']).astype(np.int64)
        from sklearn.metrics import normalized_mutual_info_score as nmi
        meta['nmi_with_z_labels_aspect_mean'] = {s: float(np.mean([nmi(y[r], old[r, a]) for a in range(16)]))
                                                 for s, r in (('user', slice(0, NU)), ('item', slice(NU, NU + NI)))}
    np.save(f'{ART}/labels_behavior.npy', np.repeat(y[:, None], 16, 1))
    json.dump(meta, open(f'{ART}/labels_behavior.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in meta.items() if k != 'sasrec'}, indent=1, default=float))


if __name__ == '__main__':
    main()
