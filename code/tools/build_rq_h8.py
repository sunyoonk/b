"""rq_H8.pt for the probes / depth test / ID builder from the dumped FACE codes (as in the Beauty V3F driver):
FACE id order (prep user2id / item2id) -> maps.pkl order (users then items); vocabulary = FACE's filtered vocabulary csv.
usage: python build_rq_h8.py <align_codes.npz> <maps.pkl> <prep data dir> <vocabulary csv> <out rq_H8.pt> [source tag]"""
import os, pickle, sys
import numpy as np, pandas as pd, torch
codes, maps_p, prep, voc, out = sys.argv[1:6]; tag = sys.argv[6] if len(sys.argv) > 6 else 'FACE align v3f full depth (RQ)'
z = np.load(codes); maps = pickle.load(open(maps_p, 'rb'))
u2id = pickle.load(open(f'{prep}/user2id.pkl', 'rb')); i2id = pickle.load(open(f'{prep}/item2id.pkl', 'rb'))
U = z['user_codes'][[u2id[u] for u in maps['users']]]; I = z['item_codes'][[i2id[x] for x in maps['items']]]
vocab = pd.read_csv(voc, keep_default_na=False)['token'].tolist()   # keep_default_na: the word "null" is a token
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
torch.save({'token_indices': torch.tensor(np.concatenate([U, I]).astype(np.int64)), 'ids': list(maps['users']) + list(maps['items']),
            'n_user': len(maps['users']), 'vocab': vocab, 'H': U.shape[2], 'source': tag}, out)
print('rq_H8', U.shape, I.shape, '->', out)
