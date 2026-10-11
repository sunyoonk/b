"""The FACE side (GRAM-format split, prep/01) and the recommender side (genrec AmazonReviews2014 5-core, RPG split) must
cover exactly the same users and items: s0 / s4 / the recommender index one table by the other. Run from code/rec with
REC_PY; downloads + processes the category on first use (cache_dir = ./cache). usage: python check_rpg_consistency.py"""
import os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/rec')
from accelerate import Accelerator
from genrec.utils import get_config, get_dataset
sys.argv = ['check']
cfg = get_config('RPG', 'AmazonReviews2014', None, {'category': os.environ['CATEGORY'], 'log_dir': '/tmp/check_logs/'})
cfg['accelerator'] = Accelerator()
ds = get_dataset('AmazonReviews2014')(cfg)
maps = pickle.load(open(os.environ['MAPS_PKL'], 'rb'))
ru, ri = set(ds.all_item_seqs), set(ds.item2id) - {'[PAD]'}
mu, mi = set(maps['users']), set(maps['items'])
print(f'genrec users {len(ru)} items {len(ri)} | maps users {len(mu)} items {len(mi)}')
print(f'users only in genrec {len(ru - mu)}, only in maps {len(mu - ru)}; items only in genrec {len(ri - mi)}, only in maps {len(mi - ri)}')
sys.exit(0 if (ru == mu and ri == mi) else 1)
