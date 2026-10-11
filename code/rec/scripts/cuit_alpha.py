"""Score-side popularity prior, independent of the ID length:
score_i = sum_{l<d_i} [log p_l(c|h; T*) - log q_l(c)] + beta * log(n_i + 1)  on a trained checkpoint.
(The alpha-PMI family log p - alpha log q is length-coupled: alpha = 0 gave val N@10 0.0114 vs 0.0437 for CUIT, because a sum
of negative log p penalises long IDs — so popularity is added as an item-level term instead.)
beta is chosen on validation (exact enumeration) only; test (exact) is run once with beta*.

    .venv/bin/python scripts/cuit_alpha.py --config=experiments/cuit_rq/CUIT.yaml --depth_dir=results/cuit_rq/depth \
        --eval_only=runs/cuit_rq/seed2024/CUIT/ckpt/best.pth --results_path=results/cuit_rq/alpha/CUIT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from genrec.pipeline import Pipeline  # noqa: E402
from genrec.utils import parse_command_line_args  # noqa: E402

BETAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)   # extended twice: β* hit 3.0, then 6.0
TEMPS = (0.01, 0.015, 0.02)


def main():
    args = parse_command_line_args(sys.argv[1:])
    config_file, ckpt, out = args.pop('config'), args.pop('eval_only'), args.pop('results_path')
    val_only = bool(args.pop('val_only', False))   # selection among a method's own variants: no test here
    buckets = bool(args.pop('buckets', False))
    betas = args.pop('betas', None)                     # override the β grid (e.g. '0' for the no-popularity-prior ablation)
    deep_ws = args.pop('deep_ws', None)                 # score weight w on semantic levels >= deep_from (variable depth)
    deep_from = int(args.pop('deep_from', 0))
    DWS_ = tuple(float(x) for x in str(deep_ws).strip('()[] ').split(',') if x.strip()) if deep_ws is not None else (None,)
    gammas = args.pop('gammas', None)                   # length-bias correction grid (variable-depth runs), selected on val
    GAMMAS_ = tuple(float(x) for x in str(gammas).strip('()[] ').split(',') if x.strip()) if gammas is not None else (0.0,)
    BETAS_ = tuple(float(x) for x in str(betas).strip('()[] ').split(',') if x.strip()) if betas is not None else BETAS     # also test R@10 / N@10 by target-popularity tertile (test users split in thirds)
    import yaml
    run_cfg = yaml.safe_load(open(config_file))
    if run_cfg.get('depth_file') and 'depth_path' not in args:
        args['depth_path'] = os.path.join(args.pop('depth_dir'), run_cfg['depth_file'])
    args.setdefault('ckpt_dir', '/tmp/cuit_alpha_ckpt/'); args.setdefault('log_dir', '/tmp/cuit_alpha_logs/')
    sys.argv = [os.path.splitext(os.path.basename(config_file))[0] + '_alpha']
    p = Pipeline(model_name='RPG', dataset_name='AmazonReviews2014', config_file=config_file, config_dict=args)
    cfg = p.config
    assert cfg.get('score') == 'rel_logq', 'alpha family needs the rel_logq score (depth-masked model)'
    loaders = {s: DataLoader(p.tokenized_datasets[s], batch_size=cfg['eval_batch_size'], shuffle=False,
                             collate_fn=p.tokenizer.collate_fn[s]) for s in ('val', 'test')}
    model = p.accelerator.unwrap_model(p.model)
    model.load_state_dict(torch.load(ckpt, map_location=cfg['device']))
    model, val_loader, test_loader = p.accelerator.prepare(model, loaders['val'], loaders['test'])
    p.trainer.model = model
    m = p.accelerator.unwrap_model(model)
    m.eval()
    if not cfg.get('no_graph'):
        m.init_graph(); m.init_flag = True
    m.generate_w_decoding_graph, m.decode_mode = False, None
    m.item_logpop = m.item_log_popularity().to(cfg['device'])
    res = {'checkpoint': ckpt, 'val_exact': {}}
    n_sem = m.n_pred_head - 3

    def set_deep_w(model, w):
        """Score weights: kept slots weigh 1, kept semantic slots >= deep_from weigh w (None = plain mask)."""
        if w is None:
            model.item_score_weight = None
            return
        sw = model.item_slot_mask.float().clone()
        sw[:, deep_from:n_sem] *= w
        model.item_score_weight = sw
    # read-out temperature selected on validation ranking together with beta: the LL-optimal T* (0.05 for most runs,
    # 0.03 for POP) ranks worse on validation than T = 0.03 and differed across conditions
    for T in TEMPS:
        m.score_temperature = T
        for b in BETAS_:
            for g in GAMMAS_:
              for w in DWS_:
                m.pop_beta, m.len_gamma = b, g
                set_deep_w(m, w)
                k = f'T{T}_b{b}' + (f'_g{g}' if gammas is not None else '') + (f'_w{w}' if w is not None else '')
                res['val_exact'][k] = dict(p.trainer.evaluate(val_loader, split='val'))
                p.log(f'[beta] T {T} beta {b} gamma {g}: val N@10 {res["val_exact"][k]["ndcg@10"]:.4f}')
    key = max(res['val_exact'], key=lambda k: res['val_exact'][k]['ndcg@10'])
    w = float(key.split('_w')[1]) if '_w' in key else None
    key0 = key.split('_w')[0]
    T, b = float(key0.split('_b')[0][1:]), float(key0.split('_b')[1].split('_g')[0])
    g = float(key0.split('_g')[1]) if '_g' in key0 else 0.0
    m.score_temperature, m.pop_beta, m.len_gamma = T, b, g
    set_deep_w(m, w)
    res.update({'T_star': T, 'beta_star': b, 'gamma_star': g, 'deep_w_star': w, 'val_ndcg@10_star': res['val_exact'][key]['ndcg@10']})
    if val_only:   # item statistics for the sum/mean selection rule (scripts/score_norm_select.py)
        import numpy as np
        np.savez(out.replace('.json', '_items.npz'), active=m.item_slot_mask.float().sum(-1).cpu().numpy(),
                 pop=(m.item_logpop.exp() - 1).cpu().numpy())
    if not val_only:
        res['test_exact'] = dict(p.trainer.evaluate(test_loader))
        if buckets:
            ev = p.trainer.evaluator
            rec, ndcg, tgt, rec5 = [], [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(cfg['device']) for k, v in batch.items()}
                    preds = m.generate(batch, n_return_sequences=ev.maxk)
                    r = ev.calculate_metrics(preds, batch['labels'])
                    rec.append(r['recall@10']); ndcg.append(r['ndcg@10']); tgt.append(batch['labels'][:, 0].cpu()); rec5.append(r['recall@5'])
            rec, ndcg, tgt, rec5 = torch.cat(rec).numpy(), torch.cat(ndcg).numpy(), torch.cat(tgt).numpy(), torch.cat(rec5).numpy()
            import numpy as _np
            _np.savez(out.replace('.json', '_peruser.npz'), recall5=rec5, recall10=rec, ndcg10=ndcg, target=tgt)   # paired bootstrap
            pop = (m.item_logpop.exp() - 1).cpu().numpy()[tgt]                 # training-target count of the test target
            import numpy as np
            q1, q2 = np.quantile(pop, [1 / 3, 2 / 3])
            grp = np.where(pop <= q1, 0, np.where(pop <= q2, 1, 2))
            res['test_by_popularity'] = {name: {'n_users': int((grp == g).sum()), 'R@10': float(rec[grp == g].mean()),
                                                'N@10': float(ndcg[grp == g].mean())}
                                         for g, name in enumerate(('tail', 'mid', 'head'))}
            res['popularity_thresholds'] = [float(q1), float(q2)]
            res['test_exact_recheck_R@10'] = float(rec.mean())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1, default=float)
    p.log(f'[beta] T* {T} beta* {b}' + ('' if val_only else f' -> test exact N@10 {res["test_exact"]["ndcg@10"]:.4f}'))


if __name__ == '__main__':
    main()
