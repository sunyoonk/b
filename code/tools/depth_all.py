"""depth_all.npz (rows users then items, maps order) from the two s8 outputs. usage: python depth_all.py <dep dir>"""
import sys
import numpy as np
A = sys.argv[1]
np.savez(f'{A}/depth_all.npz', depth=np.concatenate([np.load(f'{A}/depth_perm_user.npz')['depth'], np.load(f'{A}/depth_perm_item.npz')['depth']]))
print('depth_all ->', f'{A}/depth_all.npz')
