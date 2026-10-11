"""AdaRQ run: Pipeline.run() with timings and the extra outputs of §5, written to results/<exp>/<run>.json.

    python scripts/run_adarq.py --config=experiments/adarq_<dataset>/M.yaml [--key=value ...]

Steps (as genrec.pipeline.Pipeline.run): train with early stopping on val (exact enumeration, the model's score),
load the best checkpoint, then test with graph-constrained decoding; with `also_exact_test: True` also test with exact
enumeration. The best checkpoint is copied to <ckpt_dir>/best.pth (stage 2 reads B2's from there).
Extra config keys read here (ignored by genrec): results_path; depth_file (+ --depth_dir) -> depth_path.
"""

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from genrec.pipeline import Pipeline  # noqa: E402
from genrec.utils import parse_command_line_args  # noqa: E402


def main():
    args = parse_command_line_args(sys.argv[1:])
    config_file = args.pop('config')
    eval_only = args.pop('eval_only', None)  # checkpoint path: no training, test only (solo efficiency re-measurement)
    import yaml
    run_cfg = yaml.safe_load(open(config_file))
    if run_cfg.get('depth_file') and 'depth_path' not in args:  # depth files live in a per-seed directory
        args['depth_path'] = os.path.join(args.pop('depth_dir'), run_cfg['depth_file'])
    sys.argv = [os.path.splitext(os.path.basename(config_file))[0]]  # genrec builds file names from argv
    p = Pipeline(model_name='RPG', dataset_name='AmazonReviews2014', config_file=config_file, config_dict=args)
    cfg = p.config
    loaders = {s: DataLoader(p.tokenized_datasets[s], batch_size=cfg['train_batch_size' if s == 'train' else 'eval_batch_size'],
                             shuffle=(s == 'train'), collate_fn=p.tokenizer.collate_fn[s]) for s in ('train', 'val', 'test')}
    n_val = {'epochs': 0}
    evaluate = p.trainer.evaluate

    def counting_evaluate(dataloader, split='test'):  # one val evaluation per epoch (eval_interval 1)
        n_val['epochs'] += split == 'val'
        return evaluate(dataloader, split=split)
    p.trainer.evaluate = counting_evaluate
    t0 = time.time()
    if eval_only:
        best_epoch, best_val = None, None
        model = p.accelerator.unwrap_model(p.model)
        model.load_state_dict(torch.load(eval_only, map_location=cfg['device']))
    else:
        if cfg.get('val_T') is not None:
            # early stopping on the deployed score form: read-out temperature val_T and popularity prior val_beta.
            # The raw training score (T = 0.03, no prior) is length-biased for variable-depth SIDs.
            mm = p.accelerator.unwrap_model(p.model)
            mm.score_temperature = float(cfg['val_T']); mm.pop_beta = float(cfg.get('val_beta', 0.0))
            mm.item_logpop = mm.item_log_popularity().to(cfg['device'])
            p.log(f'[AdaRQ] validation score: T {mm.score_temperature}, beta {mm.pop_beta}')
        if cfg.get('init_ckpt'):
            mm = p.accelerator.unwrap_model(p.model)
            mm.load_state_dict(torch.load(cfg['init_ckpt'], map_location=cfg['device']))
            p.log(f"[AdaRQ] initialised from {cfg['init_ckpt']}")
        if cfg.get('residual_from') is not None:
            # residual deepening: the fixed-depth model stays frozen (backbone, heads and token embeddings of levels
            # < residual_from, dedup/CF heads); only the heads and token embeddings of the deeper semantic levels learn.
            mm = p.accelerator.unwrap_model(p.model)
            rf, K, n_sem = int(cfg['residual_from']), mm.tokenizer.codebook_size, mm.n_pred_head - 3
            for prm in mm.parameters():
                prm.requires_grad_(False)
            for i in range(rf, n_sem):
                for prm in mm.pred_heads[i].parameters():
                    prm.requires_grad_(True)
            rows = torch.zeros(mm.gpt2.wte.weight.shape[0], 1, device=mm.gpt2.wte.weight.device)
            rows[1 + rf * K: 1 + n_sem * K] = 1.0
            mm.gpt2.wte.weight.requires_grad_(True)
            mm.gpt2.wte.weight.register_hook(lambda g: g * rows.to(g.device, g.dtype))
            p.log(f'[AdaRQ] residual deepening: training heads/tokens of levels {rf + 1}..{n_sem} only')
        best_epoch, best_val = p.trainer.fit(loaders['train'], loaders['val'])
        p.accelerator.wait_for_everyone()
        model = p.accelerator.unwrap_model(p.model)
        model.load_state_dict(torch.load(p.trainer.saved_model_ckpt))
        shutil.copy(p.trainer.saved_model_ckpt, os.path.join(cfg['ckpt_dir'], 'best.pth'))
    fit_seconds = time.time() - t0
    model, test_loader = p.accelerator.prepare(model, loaders['test'])
    p.trainer.model = model
    m = p.accelerator.unwrap_model(model)

    if cfg.get('val_only'):  # selection among eval-only variants: validation (exact enumeration) only, no test
        m.generate_w_decoding_graph = False
        res = {'run': sys.argv[0], 'val_exact': dict(p.trainer.evaluate(p.accelerator.prepare(loaders['val']), split='val')),
               'eval_only_checkpoint': eval_only}
        out = cfg.get('results_path') or f'results/{sys.argv[0]}.json'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1, default=float)
        p.log(f'[AdaRQ] val only -> {out}: N@10 {res["val_exact"]["ndcg@10"]:.4f}')
        return

    if cfg.get('fit_score_temperature'):  # T* = argmax held-out (validation) log-likelihood of the kept levels
        import adarq_depth as ad
        m.score_temperature, curve = ad.select_temperature(
            m, ad._windows(m.tokenizer, m.dataset, ad.val_events(m.dataset)), cfg['device'], masked=True)
        p.log(f'[AdaRQ] score temperature T* = {m.score_temperature} (validation log-likelihood {curve})')

    t0 = time.time()
    if cfg.get('no_graph'):   # large vocabularies: RPG decoding graph not built, exact enumeration only
        test_graph, graph_seconds, test_graph_seconds = {'ndcg@10': float('nan')}, 0.0, 0.0
        cfg['also_exact_test'] = True
    else:
        m.init_graph()
        m.init_flag = True
        graph_seconds = time.time() - t0
        m.generate_w_decoding_graph = True
        t0 = time.time()
        test_graph = p.trainer.evaluate(test_loader)
        test_graph_seconds = time.time() - t0
    res = {'run': sys.argv[0], 'best_epoch': best_epoch, 'best_val_ndcg@10': best_val,
           'test_graph': dict(test_graph), 'graph_build_seconds': graph_seconds, 'test_graph_decode_seconds': test_graph_seconds}
    if cfg.get('also_exact_test'):
        m.generate_w_decoding_graph = False
        t0 = time.time()
        res['test_exact'] = dict(p.trainer.evaluate(test_loader))
        res['test_exact_decode_seconds'] = time.time() - t0
    if cfg.get('trie_astar_test'):  # exact top-k by A* over the SID prefix trie (RQ runs)
        m.generate_w_decoding_graph, m.decode_mode, m.astar_stats = False, 'trie_astar', []
        big = p.accelerator.prepare(DataLoader(p.tokenized_datasets['test'], batch_size=512, shuffle=False,
                                               collate_fn=p.tokenizer.collate_fn['test']))
        t0 = time.time()
        res['test_trie_astar'] = dict(p.trainer.evaluate(big))
        res['test_trie_astar_decode_seconds'] = time.time() - t0
        res['trie_astar_workers'] = int(cfg.get('astar_workers', 1) or 1)
        res['trie_astar_popped_per_user'] = {'mean': float(np.mean(m.astar_stats)), 'median': float(np.median(m.astar_stats)),
                                             'max': int(np.max(m.astar_stats))}
        res['trie_nodes'] = len(m._trie[0])
        m.decode_mode = None
    if cfg.get('fit_score_temperature'):
        res['score_temperature'] = m.score_temperature
    res['score'] = getattr(m, 'score_mode', 'rpg_mean')
    for alt in (cfg.get('ablation_scores') or []):  # other scores on the same checkpoint: exact (+ trie_astar if additive)
        main_mode, main_T = m.score_mode, m.score_temperature
        m.score_mode = 'centered_impute' if alt == 'centered' else alt
        m.score_temperature = None if m.score_mode == 'centered_impute' else main_T
        m.generate_w_decoding_graph, m.decode_mode = False, None
        r = {'test_exact': dict(p.trainer.evaluate(test_loader))}
        if cfg.get('trie_astar_test') and m.score_mode in ('centered_impute', 'rel_logq', 'sum_kept'):
            m.decode_mode, m.astar_stats = 'trie_astar', []
            r['test_trie_astar'] = dict(p.trainer.evaluate(big))
            m.decode_mode = None
        res.setdefault('ablation', {})[alt] = r
        m.score_mode, m.score_temperature = main_mode, main_T
    if cfg.get('eval_val'):  # validation with exact enumeration (selection among eval-only variants)
        m.generate_w_decoding_graph = False
        res['val_exact'] = dict(p.trainer.evaluate(p.accelerator.prepare(loaders['val']), split='val'))

    if eval_only:  # one training epoch (forward + backward, no optimizer step) timed alone
        m.train()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for b in loaders['train']:
            m.zero_grad()
            m.forward({k: v.to(cfg['device']) for k, v in b.items()}).loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        res['train_epoch_fwd_bwd_seconds'] = time.time() - t0
        m.zero_grad()
        m.eval()
        res['eval_only_checkpoint'] = eval_only

    L, K = m.tokenizer.n_digit, m.tokenizer.codebook_size
    if m.use_slot_mask:
        keep = m.item_slot_mask[1:].cpu()
    else:
        keep = torch.ones(m.dataset.n_items - 1, L, dtype=torch.bool)
    toks = m.item_id2tokens[1:].cpu()[keep]
    res.update({
        'mean_depth': float(keep.sum(1).float().mean()), 'L_max': L,
        'token_table_rows': int(m.gpt2.wte.weight.shape[0]),                 # embedding rows incl. pad / eos
        'distinct_tokens_used': int(torch.unique(toks).numel()),             # (level, code) pairs used by kept levels
        'fit_seconds': None if eval_only else fit_seconds, 'epochs_run': n_val['epochs'],
        'seconds_per_epoch': None if eval_only else fit_seconds / max(n_val['epochs'], 1),
        'config': {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, type(None), list))},
    })
    out = cfg.get('results_path') or f'results/{sys.argv[0]}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1, default=float)
    p.log(f'[AdaRQ] results -> {out}: graph N@10 {res["test_graph"]["ndcg@10"]:.4f}'
          + (f', exact N@10 {res["test_exact"]["ndcg@10"]:.4f}' if 'test_exact' in res else ''))
    p.trainer.end()


if __name__ == '__main__':
    main()
