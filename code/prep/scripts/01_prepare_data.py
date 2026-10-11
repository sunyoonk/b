"""Phase 0b — Step 1: parse GRAM raw text → splits + id maps + entropy.

Produces, for each dataset, files in <out>/data/<dataset>/:
    user2id.pkl, item2id.pkl
    trn_mat.pkl, val_mat.pkl, tst_mat.pkl   (scipy.sparse.coo, [U, I])
    itm_text.pkl                            (dict[int, str])
    user_train_history.pkl                  (list[list[int]], train-only)
    itm_rho.pkl                             (np.ndarray [I])
    usr_rho.pkl                             (np.ndarray [U])
    stats.json

Usage:
    pixi run -- python scripts/01_prepare_data.py \
        --gram_root <GRAM rec_datasets dir> \
        --out_root  ./data \
        --datasets <dataset> [...] \
        --minilm_path ./LLMs/all-MiniLM-L6-v2 \
        --user_last_n 10
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

# Resolve project root and importable path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.gram_loader import load_gram_dataset
from data_utils.splits import leave_one_out
from data_utils.entropy import batch_normalized_token_entropy
from data_utils.profile_text import build_user_profile_input


# GRAM directory naming: lowercase keys in our pipeline, TitleCase on disk.
GRAM_DIRNAME = {"toys": "Toys", "sports": "Sports", "yelp": "Yelp"}   # others: dataset.capitalize()


def process_dataset(
    dataset: str,
    gram_root: Path,
    out_root: Path,
    tokenize_fn,
    vocab_size: int,
    user_last_n: int,
) -> dict:
    print(f"\n=== {dataset} ===")
    src = gram_root / GRAM_DIRNAME.get(dataset, dataset.capitalize())
    if not src.is_dir():
        raise FileNotFoundError(f"GRAM source dir not found: {src}")
    user2id, item2id, sequences, item_text = load_gram_dataset(src)
    n_users, n_items = len(user2id), len(item2id)
    print(f"  users={n_users}  items={n_items}")

    trn, val, tst, n_dropped = leave_one_out(sequences, n_users, n_items)
    print(f"  trn={trn.nnz}  val={val.nnz}  tst={tst.nnz}  dropped={n_dropped}")

    # Train-only history for user profile input.
    user_train_history: list[list[int]] = [None] * n_users
    for uid, seq in enumerate(sequences):
        user_train_history[uid] = seq[:-2] if seq and len(seq) >= 3 else (seq or [])

    # ---- Entropy ----
    print("  computing item rho ...")
    itm_rho = np.asarray(
        batch_normalized_token_entropy(
            (item_text[i] for i in range(n_items)),
            tokenize_fn=tokenize_fn,
            vocab_size=vocab_size,
            show_progress=True,
        ),
        dtype=np.float32,
    )

    print("  computing user rho ...")
    user_inputs_for_rho = (
        build_user_profile_input(user_train_history[u], item_text, last_n=user_last_n)
        for u in range(n_users)
    )
    usr_rho = np.asarray(
        batch_normalized_token_entropy(
            user_inputs_for_rho,
            tokenize_fn=tokenize_fn,
            vocab_size=vocab_size,
            show_progress=True,
        ),
        dtype=np.float32,
    )

    # ---- Save ----
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name, obj):
        with open(out_dir / name, "wb") as f:
            pickle.dump(obj, f)

    dump("user2id.pkl", user2id)
    dump("item2id.pkl", item2id)
    dump("trn_mat.pkl", trn)
    dump("val_mat.pkl", val)
    dump("tst_mat.pkl", tst)
    dump("itm_text.pkl", item_text)
    dump("user_train_history.pkl", user_train_history)
    dump("itm_rho.pkl", itm_rho)
    dump("usr_rho.pkl", usr_rho)

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_train_interactions": int(trn.nnz),
        "n_val_interactions": int(val.nnz),
        "n_test_interactions": int(tst.nnz),
        "density": float(trn.nnz) / (n_users * n_items),
        "n_users_dropped_short_seq": int(n_dropped),
        "n_items_missing_text": int(sum(1 for v in item_text.values() if not v.strip())),
        "rho_item_mean": float(itm_rho.mean()),
        "rho_user_mean": float(usr_rho.mean()),
        "user_last_n": int(user_last_n),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  saved: {out_dir}")
    print(f"  stats: {stats}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gram_root", type=Path,
                   required=True)
    p.add_argument("--out_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--datasets", nargs="+",
                   required=True)
    p.add_argument("--minilm_path", type=str,
                   default=str(Path(__file__).resolve().parent.parent
                                / "LLMs/all-MiniLM-L6-v2"))
    p.add_argument("--user_last_n", type=int, default=10)
    args = p.parse_args()

    print(f"loading tokenizer from {args.minilm_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.minilm_path)
    vocab_size = tokenizer.vocab_size
    print(f"  vocab_size = {vocab_size}")

    def tokenize_fn(s: str):
        return tokenizer.tokenize(s)

    all_stats = {}
    for ds in args.datasets:
        all_stats[ds] = process_dataset(
            ds, args.gram_root, args.out_root,
            tokenize_fn, vocab_size, args.user_last_n,
        )

    print("\n=== Summary ===")
    for ds, st in all_stats.items():
        print(f"  {ds}: U={st['n_users']:>6} I={st['n_items']:>6}  "
              f"trn={st['n_train_interactions']:>7}  "
              f"rho_i={st['rho_item_mean']:.3f}  rho_u={st['rho_user_mean']:.3f}")


if __name__ == "__main__":
    main()
