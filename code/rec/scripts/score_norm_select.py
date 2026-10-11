"""Pre-registered sum/mean score selection (2026-10-10), validation only, decided before the single test run.
Inputs: two val-only cuit_alpha results (score_norm=mean and score_norm=sum, beta = 0, T grid) and the item stats.
Rule: rho = Spearman(|A_i|, n_i) over catalog items (|A_i| = active slots, n_i = training-target count).
  - |A_i| constant (fixed length): sum and mean give the same ranking -> mean.
  - |rho| >= 0.10: the ID length tracks popularity, so a sum score would let length act as a popularity prior -> mean.
  - otherwise: the norm with the higher validation N@10.
Prints the chosen norm on the last line."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
mean_json, sum_json, items_npz, out = sys.argv[1:5]
vm, vs = json.load(open(mean_json))['val_ndcg@10_star'], json.load(open(sum_json))['val_ndcg@10_star']
z = np.load(items_npz); act, pop = z['active'][1:], z['pop'][1:]   # row 0 = padding item
rho = float('nan') if act.std() == 0 else float(spearmanr(act, pop).correlation)
if act.std() == 0: choice, why = 'mean', 'constant length: identical ranking'
elif abs(rho) >= 0.10: choice, why = 'mean', f'|rho|={abs(rho):.3f}>=0.10: length tracks popularity'
else: choice, why = ('sum' if vs > vm else 'mean'), 'validation N@10'
rec = {'spearman_active_pop': rho, 'val_ndcg10_mean': vm, 'val_ndcg10_sum': vs, 'choice': choice, 'reason': why,
       'active_mean': float(act.mean()), 'active_min': float(act.min()), 'active_max': float(act.max())}
json.dump(rec, open(out, 'w'), indent=1)
print(json.dumps(rec)); print(choice)
