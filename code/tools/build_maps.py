"""maps.pkl (entity table order used by every CUIT script): users = sorted raw user ids, items = sorted ASINs,
uid / iid = their positions. Identical to the Beauty maps.pkl (checked: maps == sorted(user2id) / sorted(item2id)).
usage: python build_maps.py <prep data dir of the dataset (user2id.pkl, item2id.pkl)> <out maps.pkl>"""
import os, pickle, sys
src, out = sys.argv[1], sys.argv[2]
u2id = pickle.load(open(f'{src}/user2id.pkl', 'rb')); i2id = pickle.load(open(f'{src}/item2id.pkl', 'rb'))
users, items = sorted(u2id), sorted(i2id)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
pickle.dump({'users': users, 'items': items, 'uid': {k: n for n, k in enumerate(users)}, 'iid': {k: n for n, k in enumerate(items)}},
            open(out, 'wb'))
print(f'maps: {len(users)} users, {len(items)} items -> {out}')
