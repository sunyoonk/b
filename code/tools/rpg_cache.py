"""Download + process AmazonReviews2014/<CATEGORY> into the genrec cache (cwd = code/rec, cache/ -> RPG_CACHE)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/rec')
from accelerate import Accelerator
from genrec.utils import get_config, get_dataset
sys.argv = ['rpg_cache']
cfg = get_config('RPG', 'AmazonReviews2014', None, {'category': os.environ['CATEGORY'], 'log_dir': '/tmp/rpg_cache_logs/'})
cfg['accelerator'] = Accelerator()
ds = get_dataset('AmazonReviews2014')(cfg)
print(f"{os.environ['CATEGORY']}: {len(ds.all_item_seqs)} users, {ds.n_items - 1} items, cache {ds.cache_dir}")
