"""Leave-one-out splits (TIGER/GRAM convention).

For user u with chronological sequence [i_1, ..., i_n] (n >= 3):
    train: all (u, i_k) for k = 1..n-2
    val  : (u, i_{n-1})
    test : (u, i_n)

Users with n < 3 are dropped (rare in 5-core).
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix


def leave_one_out(
    sequences: list[list[int]],
    n_users: int,
    n_items: int,
) -> tuple[coo_matrix, coo_matrix, coo_matrix, int]:
    """Returns (trn_mat, val_mat, tst_mat, n_dropped)."""
    trn_r, trn_c = [], []
    val_r, val_c = [], []
    tst_r, tst_c = [], []
    dropped = 0
    for uid, seq in enumerate(sequences):
        if seq is None or len(seq) < 3:
            dropped += 1
            continue
        for it in seq[:-2]:
            trn_r.append(uid); trn_c.append(it)
        val_r.append(uid); val_c.append(seq[-2])
        tst_r.append(uid); tst_c.append(seq[-1])

    shape = (n_users, n_items)
    trn = coo_matrix(
        (np.ones(len(trn_r), dtype=np.float32), (trn_r, trn_c)), shape=shape
    )
    val = coo_matrix(
        (np.ones(len(val_r), dtype=np.float32), (val_r, val_c)), shape=shape
    )
    tst = coo_matrix(
        (np.ones(len(tst_r), dtype=np.float32), (tst_r, tst_c)), shape=shape
    )
    return trn, val, tst, dropped
