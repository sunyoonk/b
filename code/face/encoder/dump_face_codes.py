"""Dump full-depth FACE-RQ codes (users and items, FACE id order) from a trained lightgcn_rqvq checkpoint.
usage: python encoder/dump_face_codes.py <ckpt> <out.npz>  (run from cuit1/, SWANLAB_MODE=disabled)"""
import os, sys
ckpt, out = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], '--model', 'lightgcn_rqvq', '--dataset', os.environ['FACE_DATASET'], '--stage', 'map', '--seed', os.environ.get('SEED', '2023')]
import torch  # noqa: E402
from config.configurator import configs  # noqa: E402
from data_utils.build_data_handler import build_data_handler  # noqa: E402
from models.bulid_model import build_model  # noqa: E402

dh = build_data_handler(); dh.load_data()
model = build_model(dh).to(configs['device'])
model.load_state_dict(torch.load(ckpt)); model.eval()
model.dump_codes(out)
