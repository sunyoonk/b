"""AdaRQ §3: per-item stopping depth from level-wise usable information (V-information, Ethayarajh et al. 2022, Alg. 1).

g' = B2 (RPG on RQ-KMeans L_max, no depth mask, full training data); g = null model = training-target token
frequency per level (Laplace +1). Held-out events = validation events: per user (history i_{<T-1}, target i_{T-1}).
    v_{e,l} = ln p_l(c_{i,l} | h; T*) - ln q_l(c_{i,l})   (nats; PVI, scores, δ and β all in nats),
T* = argmax over {0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3} of the mean log p of the validation targets (the output
temperature is part of the probe family V). Test events are never read (the test target i_T is not even loaded into a
window: validation windows are built from i_1..i_{T-1}). The validation set is also g''s early-stopping set.
Item x level EB shrinkage (scripts/slot_pvi.eb_aggregate) -> U_tilde; depth rules (§3.4) with prefix-uniqueness
extension. Sensitivity (reported only): the same computation on in-sample training events.

    python scripts/adarq_depth.py --config=experiments/adarq_<dataset>/B2.yaml --ckpt=<B2 best.pth> --out=<dir> \
        [--d_min=4] [--deltas=0] [--seed=2024]
    python scripts/adarq_depth.py --from_pvi=<dir>/adarq_pvi.npz --out=<dir> --deltas=0.01,0.02  (info rule for new δ only)
Writes <out>/depth_<rule>.npz (items, depth) and .npy (depth, rows = item ids 1..N-1), <out>/adarq_pvi.npz,
<out>/adarq_depth.md.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_pvi import eb_aggregate, token_log2_prior  # noqa: E402

LN2 = float(np.log(2))


def token_logq(codes_all, target_items, codebook_size):
    """ln q_l(c) [L, K]: per-level frequency of the next-item target tokens (Laplace +1, normalised over the 256 codes)."""
    return token_log2_prior(codes_all, target_items, codebook_size) * LN2

T_GRID = (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)


# ----------------------------------------------------------------------------------------------------------------------
# events
def val_events(dataset):
    """[(user, position, history, target)] — one per user, position T-2 (0-based) = i_{T-1}; history i_1..i_{T-2}."""
    out = []
    for u, s in dataset.all_item_seqs.items():
        if len(s) > 2:
            out.append((u, len(s) - 2, s[:-2], s[-2]))
    return out


def train_events(dataset):
    """In-sample events of the training parts: (user, position t, history i_{<t}, target i_t), t = 1..T-3."""
    out = []
    for u, s in dataset.all_item_seqs.items():
        tr = s[:-2]
        out += [(u, t, tr[:t], tr[t]) for t in range(1, len(tr))]
    return out


def _windows(tokenizer, dataset, events):
    """One window per event: last <= max_item_seq_len history items, label at the last position."""
    L = tokenizer.config['max_item_seq_len']
    rows = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for _, _, h, t in events:
        ids, att, lab, _ = tokenizer._tokenize_later_items(list(h[-L:]) + [t])
        rows['input_ids'].append(ids); rows['attention_mask'].append(att); rows['labels'].append(lab)
    return rows


def _train_windows(tokenizer, dataset, users=None):
    """RPG's own training windows (every training event once, causal: one forward per window), optionally only for
    the given users."""
    rows = {'input_ids': [], 'attention_mask': [], 'labels': []}
    keep = None if users is None else set(users)
    for u, s in dataset.all_item_seqs.items():
        if keep is not None and u not in keep:
            continue
        t = tokenizer.tokenize_function({'item_seq': [s[:-2]]}, 'train')
        for k in rows:
            rows[k] += t[k]
    return rows


@torch.no_grad()
def cosine_and_codes(model, rows, device, batch_size=256):
    """Yields (target ids [E_b], cos(g_l(h), e_{l,c}) [E_b, L, K], 0-based target codes [E_b, L])."""
    model.eval()
    L, K = model.n_pred_head, model.tokenizer.codebook_size
    emb = F.normalize(model.gpt2.wte.weight[1:-1], dim=-1).view(L, K, -1)
    for b in range(0, len(rows['input_ids']), batch_size):
        batch = {k: torch.tensor(v[b:b + batch_size], device=device) for k, v in rows.items()}
        st = model.forward(batch, return_loss=False).final_states
        lab = batch['labels'].view(-1)
        keep = lab != -100
        st = F.normalize(st.reshape(-1, L, st.shape[-1])[keep], dim=-1)
        tgt = lab[keep]
        yield tgt, torch.einsum('eld,lkd->elk', st, emb).float(), model.item_id2tokens[tgt] - 1 - torch.arange(L, device=device) * K


def select_temperature(model, rows, device, masked=False):
    """T* = argmin_T validation NLL = argmax_T mean ln p_l(c | h; T) over the given (validation) events and levels,
    T in T_GRID (absolute temperature applied to the raw cosine: logits = cos / T; the grid contains τ_train = 0.03). masked: only the target item's kept levels (depth / slot mask) count."""
    tot = {T: 0.0 for T in T_GRID}
    cnt = 0
    for tgt, cos, codes in cosine_and_codes(model, rows, device):
        w = model.item_slot_mask[tgt].float() if masked and model.use_slot_mask else torch.ones_like(codes, dtype=torch.float)
        for T in T_GRID:
            tot[T] += (F.log_softmax(cos / T, -1).gather(-1, codes.unsqueeze(-1)).squeeze(-1) * w).sum().item()
        cnt += w.sum().item()
    curve = {T: tot[T] / cnt for T in T_GRID}
    return max(curve, key=curve.get), curve


def event_logp(model, rows, device, T):
    """(target ids [E], ln p_l(c_il | h; T) [E, L])."""
    items, out = [], []
    for tgt, cos, codes in cosine_and_codes(model, rows, device):
        out.append(F.log_softmax(cos / T, -1).gather(-1, codes.unsqueeze(-1)).squeeze(-1).cpu().numpy())
        items.append(tgt.cpu().numpy())
    return np.concatenate(items), np.concatenate(out).astype(np.float64)


# ----------------------------------------------------------------------------------------------------------------------
# Empirical Bayes with item-specific, user-clustered standard errors (main; slot_pvi.eb_aggregate = pooled-σ² ablation)
def eb_cluster(items, v, users, n_items):
    """Per level l, items with events:
      Û_il = mean of v_el over the item's events;
      SE²_il = max(cluster-robust variance of the mean with users as clusters (small-sample factor G/(G-1); needs G ≥ 2),
                   σ²_l / n_i), σ²_l = pooled within-item variance (the iid floor; also used alone when G_i = 1);
      τ²_l   = DerSimonian–Laird method of moments with w = 1/SE², truncated at 0;
      μ_l    = Σ w* Û / Σ w*, w* = 1/(SE² + τ²);
      Ũ_il   = μ_l + B (Û_il − μ_l), B = τ²/(τ² + SE²_il);  posterior SD = sqrt(τ² SE² / (τ² + SE²)).
    τ²_l = 0 (no detectable between-item variation): B = 0, Ũ_il = μ_l for every item of that level (reported).
    n_i = 0: Ũ_il = μ_l, SD = sqrt(τ²_l). Row 0 (padding) has n = 0."""
    items = np.asarray(items); v = np.asarray(v, dtype=np.float64)
    E, L = v.shape
    n = np.bincount(items, minlength=n_items).astype(np.float64)
    s1 = np.zeros((n_items, L)); np.add.at(s1, items, v)
    U_hat = np.divide(s1, n[:, None], out=np.zeros_like(s1), where=n[:, None] > 0)
    resid = v - U_hat[items]
    has = n > 0
    sigma2 = (resid ** 2).sum(0) / max(n.sum() - has.sum(), 1)                      # pooled within-item variance
    # user clusters within items
    uid = np.unique(np.asarray(users, dtype=object), return_inverse=True)[1].ravel()
    pair = items.astype(np.int64) * (uid.max() + 1) + uid
    pk, pinv = np.unique(pair, return_inverse=True)
    pinv = pinv.ravel()
    csum = np.zeros((len(pk), L)); np.add.at(csum, pinv, resid)                      # per (item, user) residual sums
    c_item = (pk // (uid.max() + 1)).astype(np.int64)
    G = np.bincount(c_item, minlength=n_items).astype(np.float64)
    rob = np.zeros((n_items, L)); np.add.at(rob, c_item, csum ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rob = np.where((G[:, None] >= 2) & has[:, None], G[:, None] / np.maximum(G[:, None] - 1, 1) * rob / n[:, None] ** 2, 0.0)
        iid = np.where(has[:, None], sigma2[None, :] / np.maximum(n[:, None], 1), np.inf)
    SE2 = np.maximum(rob, iid)
    w = np.where(has[:, None], 1.0 / SE2, 0.0)
    mu_fe = (w * U_hat).sum(0) / w.sum(0)
    Q = (w * (U_hat - mu_fe) ** 2).sum(0)
    k = has.sum()
    c = w.sum(0) - (w ** 2).sum(0) / w.sum(0)
    tau2 = np.maximum(0.0, (Q - (k - 1)) / c)
    ws = np.where(has[:, None], 1.0 / (SE2 + tau2[None, :]), 0.0)
    mu = (ws * U_hat).sum(0) / ws.sum(0)
    B = np.where(has[:, None], tau2[None, :] / (tau2[None, :] + SE2), 0.0)
    U_tilde = mu[None, :] + B * (U_hat - mu[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        SD = np.where(has[:, None], np.sqrt(tau2[None, :] * SE2 / (tau2[None, :] + SE2)), np.sqrt(tau2)[None, :])
    SE2_out = np.where(has[:, None], SE2, np.nan)
    return dict(n=n, U_hat=U_hat, mu=mu, sigma2=sigma2, tau2=tau2, U_tilde=U_tilde, SE=SD, SE2=SE2_out, B=B,
                clusters=G, kappa=np.nanmedian(np.where(has[:, None], SE2 / np.maximum(tau2[None, :], 1e-300), np.nan), axis=0),
                tau2_zero_levels=int((tau2 == 0).sum()), robust_used_share=float(((rob > iid) & has[:, None]).sum() / max(has.sum() * L, 1)))


# ----------------------------------------------------------------------------------------------------------------------
# cross-fitted PVI (K user folds): probe k is trained without fold k (holdout_fold = k) and scores every training-part
# event of the fold-k users; selection users (early stopping / δ selection) are excluded from the PVI events.
def fold_users(dataset, k, n_folds, seed=2024, selection_frac=0.0):
    from genrec.dataset import user_fold_index, is_selection_user
    return [u for u, s in dataset.all_item_seqs.items()
            if len(s) > 2 and user_fold_index(u, n_folds, seed) == k and not is_selection_user(u, selection_frac, seed)]


def probe_users(dataset, k, n_folds, seed=2024):
    """Users a probe with holdout_fold = k trains and early-stops on."""
    from genrec.dataset import user_fold_index
    return [u for u, s in dataset.all_item_seqs.items() if len(s) > 2 and user_fold_index(u, n_folds, seed) != k]


def compute_pvi_crossfit(model, dataset, device, probes, n_folds=5, seed=2024, selection_frac=0.0):
    """Returns (target ids [E], PVI [E, L], {fold: T*}, {fold: T curve}, PVI event keys [(user, position)], fold of each
    event [E])."""
    codes_all = np.vstack([np.zeros((1, model.n_pred_head), np.int64), item_codes(model)])
    items, vs, Ts, curves, keys, folds = [], [], {}, {}, [], []
    for k, ckpt in enumerate(probes):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        own = probe_users(dataset, k, n_folds, seed)
        tt = np.asarray([dataset.item2id[it] for u in own for it in dataset.all_item_seqs[u][:-2][1:]], dtype=np.int64)
        logq = token_logq(codes_all, tt, model.tokenizer.codebook_size)
        own_set = set(own)
        val_rows = _windows(model.tokenizer, dataset, [e for e in val_events(dataset) if e[0] in own_set])
        Ts[k], curves[k] = select_temperature(model, val_rows, device)
        held = fold_users(dataset, k, n_folds, seed, selection_frac)
        it, lp = event_logp(model, _train_windows(model.tokenizer, dataset, held), device, Ts[k])
        vs.append(lp - logq[np.arange(model.n_pred_head), codes_all[it]])
        items.append(it)
        folds.append(np.full(len(it), k))
        held_set = set(held)
        keys += [(u, t) for u, t, _, _ in train_events(dataset) if u in held_set]
    return np.concatenate(items), np.concatenate(vs), Ts, curves, keys, np.concatenate(folds)


def event_count_report(dataset, n_folds=5, seed=2024, selection_frac=0.0):
    """Per-item event counts n_i for the validation-event PVI and for the cross-fitted training-event PVI."""
    from genrec.dataset import is_selection_user
    n_items = dataset.n_items
    val = np.bincount([dataset.item2id[t] for _, _, _, t in val_events(dataset)], minlength=n_items)[1:]
    tr = [(u, dataset.item2id[t]) for u, _, _, t in train_events(dataset)]
    cf = np.bincount([i for u, i in tr if not is_selection_user(u, selection_frac, seed)], minlength=n_items)[1:]
    rows = ['| events | total | items n_i = 0 | n_i = 1 | n_i = 2–4 | n_i = 5–9 | n_i ≥ 10 | median | mean | 90% |',
            '|---|---|---|---|---|---|---|---|---|---|']
    for lab, n in (('validation (one per user)', val),
                   (f'cross-fitted training-part events ({n_folds} folds, selection users {selection_frac:.0%} excluded)', cf)):
        rows.append(f'| {lab} | {int(n.sum())} | {int((n == 0).sum())} | {int((n == 1).sum())} | {int(((n >= 2) & (n <= 4)).sum())} | '
                    f'{int(((n >= 5) & (n <= 9)).sum())} | {int((n >= 10).sum())} | {np.median(n):.0f} | {n.mean():.2f} | '
                    f'{np.percentile(n, 90):.0f} |')
    return rows, val, cf


def check_disjoint(dataset, pvi_keys, selection_frac=0.0, seed=2024, val_user_set='all'):
    """PVI events vs early-stopping / δ-selection events (validation targets of the validation users)."""
    from genrec.dataset import is_selection_user
    sel = [(u, len(s) - 2) for u, s in dataset.all_item_seqs.items()
           if len(s) > 2 and (val_user_set == 'all' or is_selection_user(u, selection_frac, seed))]
    ev_overlap = len(set(pvi_keys) & set(sel))
    user_overlap = len({u for u, _ in pvi_keys} & {u for u, _ in sel})
    return ev_overlap, user_overlap, len(sel)


# ----------------------------------------------------------------------------------------------------------------------
# depth rules (pure numpy; codes [N, L] 0-based for items 1..N-1, depths 1-based)
def info_depth(U, d_min, delta=0.0, lower=None):
    """argmax_{d in [max(d_min, lower_i), L]} sum_{l <= d} (U_l - delta); ties -> smallest d.
    lower = minimal unique prefix length q_i makes the uniqueness a constraint of the argmax (AdaRQ main rule)."""
    cs = np.cumsum(U - delta, axis=1)
    lo = np.full(len(U), d_min) if lower is None else np.maximum(d_min, lower)
    cs = np.where(np.arange(1, U.shape[1] + 1)[None, :] >= lo[:, None], cs, -np.inf)
    return np.argmax(cs, axis=1) + 1


def prefix_conflicts(codes, d):
    """Items whose SID (prefix of length d_i) equals another item's SID or is a prefix of another item's SID."""
    flag = np.zeros(len(d), dtype=bool)
    for t in np.unique(d):
        cand = np.where(d >= t)[0]
        _, inv, cnt = np.unique(codes[cand, :t], axis=0, return_inverse=True, return_counts=True)
        shared = cnt[inv.ravel()] > 1
        flag[cand[shared & (d[cand] == t)]] = True
    return flag


def extend_unique(codes, d):
    """Add one level at a time to every conflicting item until no conflicts remain (items at L_max stay tied)."""
    L = codes.shape[1]
    d = d.copy()
    for _ in range(L):
        c = prefix_conflicts(codes, d) & (d < L)
        if not c.any():
            break
        d[c] += 1
    return d


def min_unique_prefix(codes):
    L = codes.shape[1]
    out = np.full(len(codes), L)
    done = np.zeros(len(codes), dtype=bool)
    for t in range(1, L + 1):
        _, inv, cnt = np.unique(codes[:, :t], axis=0, return_inverse=True, return_counts=True)
        new = (cnt[inv.ravel()] == 1) & ~done
        out[new] = t
        done |= new
    return out


def depth_rules(U, mu, codes, popularity, d_min, deltas=(0.0,), seed=2024):
    """Returns {rule: (reference d, final d)}.
    Main rule (uniqueness as a constraint, q_i = minimal unique prefix length, lo_i = max(d_min, q_i)):
      info:          argmax_{d >= lo_i} sum_{l<=d} (U_il - δ)          (reference d = unconstrained argmax over d >= d_min)
      random:        info's final depths permuted across items, then d = max(d, lo_i)   (reference = permuted d)
      popularity:    info's final depths re-assigned by training frequency rank, then max(d, lo_i)
      unique:        lo_i
      global:        argmax_{d >= lo_i} sum_{l<=d} mu_l
      fixed4 / fixed<L> / fixedmean (round(mean final info d)): no uniqueness (ties allowed, as RPG)
      info_delta<δ>: info with the other deltas
    Ablation (previous rule): info_ext / random_ext / popularity_ext / global_ext = unconstrained argmax followed by
    one-level-at-a-time uniqueness extension."""
    L = codes.shape[1]
    lo = np.maximum(d_min, min_unique_prefix(codes))
    rng = np.random.default_rng(seed)
    out = {}
    ref = info_depth(U, d_min, deltas[0])
    info = info_depth(U, d_min, deltas[0], lower=lo)
    out['info'] = (ref, info)
    r = rng.permutation(info)
    out['random'] = (r, np.maximum(r, lo))
    pop = np.empty_like(info)
    pop[np.argsort(-popularity, kind='stable')] = np.sort(info)[::-1]
    out['popularity'] = (pop, np.maximum(pop, lo))
    out['unique'] = (lo, lo)
    g = np.full(len(codes), int(np.argmax(np.cumsum(mu)[d_min - 1:]) + d_min))
    out['global'] = (g, info_depth(np.broadcast_to(mu, U.shape), d_min, 0.0, lower=lo))
    k_mean = int(round(info.mean()))
    for name, k in (('fixed4', d_min), (f'fixed{L}', L), ('fixedmean', k_mean)):
        out[name] = (np.full(len(codes), k), np.full(len(codes), k))
    for delta in deltas[1:]:
        out[f'info_delta{delta:g}'] = (info_depth(U, d_min, delta), info_depth(U, d_min, delta, lower=lo))
    # ablation: previous rule (post-hoc extension)
    rng = np.random.default_rng(seed)
    out['info_ext'] = (ref, extend_unique(codes, ref))
    r = rng.permutation(ref)
    out['random_ext'] = (r, extend_unique(codes, r))
    pe = np.empty_like(ref)
    pe[np.argsort(-popularity, kind='stable')] = np.sort(ref)[::-1]
    out['popularity_ext'] = (pe, extend_unique(codes, pe))
    out['global_ext'] = (g, extend_unique(codes, g))
    return out


def save_depths(out_dir, items, rules):
    os.makedirs(out_dir, exist_ok=True)
    for name, (_, d) in rules.items():
        np.savez(f'{out_dir}/depth_{name}.npz', items=items, depth=d.astype(np.int64))
        np.save(f'{out_dir}/depth_{name}.npy', d.astype(np.int64))


def rules_table(rules, codes, L):
    md = ['| rule | mean d | min | median | max | items raised by uniqueness (final > reference) | mean raise | prefix conflicts | distinct SIDs |',
          '|---|---|---|---|---|---|---|---|---|']
    for name, (pre, d) in rules.items():
        ext = d > pre
        conf = int(prefix_conflicts(codes, d).sum())
        sids = len({tuple(c[:k]) for c, k in zip(codes, d)})
        md.append(f'| {name} | {d.mean():.2f} | {d.min()} | {int(np.median(d))} | {d.max()} | {int(ext.sum())} | '
                  f'{(d - pre)[ext].mean() if ext.any() else 0:.2f} | {conf} | {sids} |')
    return md


# ----------------------------------------------------------------------------------------------------------------------
def build(config_file, overrides):
    from accelerate import Accelerator
    from genrec.utils import get_config, get_dataset, init_device, init_logger
    from genrec.models.RPG.model import RPG
    from genrec.models.RPG.tokenizer import RPGTokenizer
    cfg = get_config('RPG', 'AmazonReviews2014', config_file, dict(overrides, depth_path=None, user_fold=None))
    cfg['device'], cfg['use_ddp'] = init_device()
    cfg['accelerator'] = Accelerator()
    init_logger(cfg)
    ds = get_dataset('AmazonReviews2014')(cfg)
    ds.split()
    tok = RPGTokenizer(cfg, ds)
    return cfg, ds, tok, RPG(cfg, ds, tok).to(cfg['device'])


def item_codes(model):
    L, K = model.n_pred_head, model.tokenizer.codebook_size
    return (model.item_id2tokens.cpu().numpy() - 1 - np.arange(L) * K)[1:]          # rows = item ids 1..N-1


def train_target_ids(dataset):
    return np.asarray([dataset.item2id[it] for s in dataset.all_item_seqs.values() for it in s[:-2][1:]], dtype=np.int64)


def compute_pvi(model, dataset, device, events='val'):
    """Event PVI matrix for 'val' (held-out) or 'train' (in-sample) events; T* always from the validation events."""
    codes_all = np.vstack([np.zeros((1, model.n_pred_head), np.int64), item_codes(model)])
    logq = token_logq(codes_all, train_target_ids(dataset), model.tokenizer.codebook_size)
    val_ev = val_events(dataset)
    val_rows = _windows(model.tokenizer, dataset, val_ev)
    T, curve = select_temperature(model, val_rows, device)
    rows = val_rows if events == 'val' else _train_windows(model.tokenizer, dataset)
    items, lp = event_logp(model, rows, device, T)
    v = lp - logq[np.arange(model.n_pred_head), codes_all[items]]
    ev = val_ev if events == 'val' else train_events(dataset)
    assert len(ev) == len(items) and all(dataset.item2id[t] == i for (_, _, _, t), i in zip(ev, items)), 'event order'
    return items, v, T, curve, [u for u, _, _, _ in ev]


def main():
    from genrec.utils import parse_command_line_args
    args = parse_command_line_args(sys.argv[1:])
    sys.argv = ['adarq_depth']
    out = args.pop('out')
    d_min = int(args.pop('d_min', 4))
    seed = int(args.pop('seed', 2024))
    deltas = args.pop('deltas', 0.0)
    deltas = [float(x) for x in (deltas if isinstance(deltas, (tuple, list)) else [deltas])]
    os.makedirs(out, exist_ok=True)

    if 'from_pvi' in args:  # stage 4: info rule for new δ from the saved statistics
        z = np.load(args.pop('from_pvi'), allow_pickle=True)
        U, mu, codes, pop, items = z['U_tilde'], z['mu'], z['codes'], z['popularity'], z['items']
        rules = {}
        lo = np.maximum(int(z['d_min']), min_unique_prefix(codes))
        for delta in deltas:
            rules[f'info_delta{delta:g}'] = (info_depth(U, int(z['d_min']), delta), info_depth(U, int(z['d_min']), delta, lower=lo))
        save_depths(out, items, rules)
        print('\n'.join(rules_table(rules, codes, codes.shape[1])))
        return

    mode = args.pop('events', 'val')                 # val (held-out validation events) | crossfit (K user folds)
    n_folds = int(args.pop('n_folds', 5))
    sel_frac = float(args.pop('selection_user_frac', 0.0))
    if args.pop('count_only', False):                # event-count distribution only (CPU, no model)
        from accelerate import Accelerator
        from genrec.utils import get_config, get_dataset, init_logger
        cfg = get_config('RPG', 'AmazonReviews2014', args.pop('config'), args)
        cfg['accelerator'] = Accelerator()
        init_logger(cfg)
        ds = get_dataset('AmazonReviews2014')(cfg)
        rows, _, _ = event_count_report(ds, n_folds, seed, sel_frac)
        md = [f'# AdaRQ PVI event counts per item — {cfg["category"]}', ''] + rows
        open(f'{out}/event_counts.md', 'w').write('\n'.join(md) + '\n')
        print('\n'.join(md))
        return

    t_start = time.time()
    config_file, ckpt = args.pop('config'), args.pop('ckpt', None)
    probes = args.pop('probes', None)
    cfg, ds, tok, model = build(config_file, args)
    codes = item_codes(model)
    L = codes.shape[1]
    items_raw = np.array([ds.id_mapping['id2item'][i] for i in range(1, ds.n_items)])
    pop = np.bincount(train_target_ids(ds), minlength=ds.n_items)[1:]
    from scipy.stats import spearmanr

    if mode == 'crossfit':
        probes = [probes] if isinstance(probes, str) else list(probes)
        probes = [x for p_ in probes for x in str(p_).split(',')]
        assert len(probes) == n_folds, f'{n_folds} probe checkpoints needed'
        ev_items, v, Ts, curves, keys, ev_folds = compute_pvi_crossfit(model, ds, cfg['device'], probes, n_folds, seed, sel_frac)
        assert all(ds.item2id[ds.all_item_seqs[u][t]] == i for (u, t), i in zip(keys, ev_items)), 'event / key order'
        ev_users = [u for u, _ in keys]
        np.savez(f'{out}/crossfit_events.npz', items=ev_items, v=v.astype(np.float32), fold=ev_folds,
                 users=np.asarray(ev_users, dtype=object))  # Ũ reproducibility
        T, curve = Ts, curves[0]
        ev_overlap, user_overlap, n_sel = check_disjoint(ds, keys, sel_frac, seed, 'selection' if sel_frac > 0 else 'all')
        count_rows, _, _ = event_count_report(ds, n_folds, seed, sel_frac)
    else:
        model.load_state_dict(torch.load(ckpt, map_location=cfg['device']))
        ev_items, v, T, curve, ev_users = compute_pvi(model, ds, cfg['device'], 'val')
    st = eb_cluster(ev_items, v, ev_users, ds.n_items)
    U = st['U_tilde'][1:]
    rules = depth_rules(U, st['mu'], codes, pop, d_min, deltas, seed)
    save_depths(out, items_raw, rules)

    # sensitivity (reported only): val mode -> the same g' on in-sample training events;
    # crossfit mode -> validation-event PVI of the full-data g' (--ckpt), if given
    st_tr, other_lab = None, None
    if mode == 'val':
        tr_items, v_tr, _, _, tr_users = compute_pvi(model, ds, cfg['device'], 'train')
        other_lab = 'in-sample'
    elif ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=cfg['device']))
        tr_items, v_tr, _, _, tr_users = compute_pvi(model, ds, cfg['device'], 'val')
        other_lab = 'validation (full-data g\')'
    if other_lab:
        st_tr = eb_cluster(tr_items, v_tr, tr_users, ds.n_items)
        have = (st['n'][1:] > 0) & (st_tr['n'][1:] > 0)
        per_item = np.array([spearmanr(a_, b_)[0] for a_, b_ in zip(U[have], st_tr['U_tilde'][1:][have])])
        d_tr = info_depth(st_tr['U_tilde'][1:], d_min, deltas[0], lower=np.maximum(d_min, min_unique_prefix(codes)))
    np.savez(f'{out}/adarq_pvi.npz', items=items_raw, codes=codes, popularity=pop, d_min=d_min, events=mode,
             T_star=np.array(list(T.values()) if isinstance(T, dict) else T),
             T_curve=np.array([[t, c] for t, c in curve.items()]), U_tilde=U, U_hat=st['U_hat'][1:], n=st['n'][1:],
             mu=st['mu'], sigma2=st['sigma2'], tau2=st['tau2'], kappa=st['kappa'],
             **({} if st_tr is None else {'U_tilde_other': st_tr['U_tilde'][1:], 'n_other': st_tr['n'][1:], 'mu_other': st_tr['mu']}))

    n = st['n'][1:]
    if mode == 'crossfit':
        head = [f'Cross-fitted PVI: {n_folds} user folds (md5 user hash, seed {seed}); probe k = RPG-RQ L_max = {L} trained and '
                f'early-stopped without the users of fold k; PVI on every training-part event of the fold-k users '
                f'({len(ev_items)} events); selection users ({sel_frac:.0%}) excluded. g = per-level training-target token '
                'frequency of the probe\'s own users (Laplace +1). Test events are never read.',
                '', 'T* per probe (own users\' validation events): ' + ', '.join(f'fold {k}: {t}' for k, t in T.items()),
                f'Overlap of PVI events with early-stopping / δ-selection events: {ev_overlap} events, {user_overlap} users '
                f'(selection events: {n_sel}).', '', '### Events per item', ''] + count_rows + ['']
    else:
        head = [f'g\' = {ckpt} (RPG, RQ-KMeans L_max = {L}, no depth mask, full training data); g = per-level training-target '
                f'token frequency (Laplace +1). PVI on validation events (one per user, {len(ev_items)} events); test events are '
                'never read. **The validation set is also g\'\'s early-stopping set and the T* selection set.**', '',
                f'T* = {T} (grid {", ".join(f"{t}: {c:.3f}" for t, c in curve.items())} — mean ln p on validation events; T* = argmin validation NLL).']
    md = [f'# AdaRQ depths — {cfg["category"]}', ''] + head + [
          f'Mean event PVI {v.mean():+.3f} nats; levels with μ_l > 0: {int((st["mu"] > 0).sum())}/{L}.',
          f'n_i (PVI events per item): items with n_i = 0: {int((n == 0).sum())}; median {np.median(n):.0f}, '
          f'mean {n.mean():.2f}, max {int(n.max())}. EB: item-specific user-clustered SE (DerSimonian–Laird τ²); levels with '
          f'τ² = 0 (Ũ = μ for all items): {st["tau2_zero_levels"]}/{L}; cluster-robust SE above the iid floor for '
          f'{st["robust_used_share"]:.1%} of (item, level); median B over (item, level): {np.median(st["B"][1:][n > 0]):.3f}.', '',
          '## Depth rules (d_min = %d, δ = %g)' % (d_min, deltas[0]), ''] + rules_table(rules, codes, L)
    md += ['', '## Per-level statistics', '', f'| level | μ_l (nats) | σ²_l | τ²_l | median B_l | μ_l ({other_lab or "–"}) | share of items with d_info ≥ l |',
           '|---|---|---|---|---|---|---|']
    for l in range(L):
        md.append(f'| {l + 1} | {st["mu"][l]:+.4f} | {st["sigma2"][l]:.3f} | {st["tau2"][l]:.2e} | {np.median(st["B"][1:][st["n"][1:] > 0, l]):.3f} | ' +
                  (f'{st_tr["mu"][l]:+.4f}' if st_tr is not None else '–') + f' | {(rules["info"][1] > l).mean():.1%} |')
    hist = np.bincount(rules['info'][1], minlength=L + 1)
    md += ['', '## d_info distribution (after extension)', '', '| d | items |', '|---|---|'] + \
          [f'| {k} | {hist[k]} |' for k in range(1, L + 1) if hist[k]]
    pre, fin = rules['info']
    ext = fin > pre
    # headline depth statistics
    md += ['', '## info depth: summary', '', f'- mean d {fin.mean():.2f} (unconstrained argmax {pre.mean():.2f}); '
           f'd = {d_min}: {(fin == d_min).mean():.1%}; d = {L}: {(fin == L).mean():.1%}',
           f'- Spearman(d, log(1 + popularity)) across items: {spearmanr(fin, np.log1p(pop))[0]:.3f} '
           f'(popularity = training-target count); Spearman(d, n_i): {spearmanr(fin, n)[0]:.3f}']
    # where do the final info depths come from: the info rule itself or the uniqueness extension
    pre, fin = rules['info']
    ext = fin > pre
    md += ['', '## info rule: unconstrained argmax vs raised by the uniqueness constraint (d ≥ q_i)', '',
           '| | items | share | mean d (final) | mean extension (levels) |', '|---|---|---|---|---|',
           f'| final d = unconstrained argmax | {int((~ext).sum())} | {(~ext).mean():.1%} | {fin[~ext].mean():.2f} | 0 |',
           f'| raised by the uniqueness constraint | {int(ext.sum())} | {ext.mean():.1%} | '
           f'{fin[ext].mean() if ext.any() else float("nan"):.2f} | {(fin - pre)[ext].mean() if ext.any() else 0:.2f} |',
           f'| all | {len(fin)} | 100% | {fin.mean():.2f} | {(fin - pre).mean():.2f} (over all items) |',
           '', f'Mean unconstrained d {pre.mean():.2f}, constrained {fin.mean():.2f}; the constraint adds '
           f'{(fin - pre).sum() / max(fin.sum(), 1):.1%} of all kept levels.']
    # depth by number of validation events
    md += ['', '## d_info by n_i (validation events of the item)', '',
           '| n_i | items | mean d (unconstrained) | mean d (final) | median | 90% | max | raised share | d = d_min (final) |',
           '|---|---|---|---|---|---|---|---|---|']
    for lab, sel in (('0', n == 0), ('1', n == 1), ('2–4', (n >= 2) & (n <= 4)), ('5–9', (n >= 5) & (n <= 9)),
                     ('≥ 10', n >= 10)):
        if sel.any():
            md.append(f'| {lab} | {int(sel.sum())} | {pre[sel].mean():.2f} | {fin[sel].mean():.2f} | {int(np.median(fin[sel]))} | '
                      f'{int(np.percentile(fin[sel], 90))} | {fin[sel].max()} | {ext[sel].mean():.1%} | {(fin[sel] == d_min).mean():.1%} |')
    # sensitivity
    if other_lab:
      pooled = spearmanr(U[have].ravel(), st_tr['U_tilde'][1:][have].ravel())[0]
      per_level = [spearmanr(U[have][:, l], st_tr['U_tilde'][1:][have][:, l])[0] for l in range(L)]
      md += ['', f'## Sensitivity: {other_lab} events (not used for depths)', '',
           f'- events: {len(tr_items)}; mean PVI {v_tr.mean():+.3f} nats (depth events: {v.mean():+.3f})',
           f'- per-item Spearman over levels of Ũ (depth events) vs Ũ ({other_lab}), items with n_i > 0: median '
           f'{np.nanmedian(per_item):.3f}, IQR [{np.nanpercentile(per_item, 25):.3f}, {np.nanpercentile(per_item, 75):.3f}]',
           f'- pooled Spearman over (item, level), items with n_i > 0: {pooled:.3f}',
           f'- per-level Spearman across items: median {np.nanmedian(per_level):.3f}, min {np.nanmin(per_level):.3f}, '
           f'max {np.nanmax(per_level):.3f}',
           f'- Spearman across items of d_info (depth events) vs d_info ({other_lab}): {spearmanr(rules["info"][1], d_tr)[0]:.3f}; '
           f'mean d_info ({other_lab}) {d_tr.mean():.2f}']
    md += ['', f'Total time {time.time() - t_start:.0f} s.']
    open(f'{out}/adarq_depth.md', 'w').write('\n'.join(md) + '\n')
    json.dump({'T_star': {str(k): v_ for k, v_ in T.items()} if isinstance(T, dict) else T, 'mean_d_info': float(rules['info'][1].mean()),
               'fixedmean_k': int(rules['fixedmean'][1][0])}, open(f'{out}/summary.json', 'w'))
    print('\n'.join(md[:12]))


if __name__ == '__main__':
    main()
