"""Phase 0b — Step 3: build OpenAI Batch-API JSONL files for profile generation.

Output: <out>/batches/profile/<dataset>_<kind>.jsonl
   kind in {user, item}
   one request per (dataset, kind, id)

Prompt format follows IGSRec/RLMRec (gpt-4o-mini, single-sentence
profile). For GRAM (no reviews), we adapt the input slot from "user
review" to "user-history item plain texts" and from "item review" to
"item plain text". The OUTPUT formats remain identical so the resulting
pkl files are drop-in compatible.

System prompts (verbatim from IGSRec):
    item: "Use one short sentence: 'The item attributes are xxx.'"
    user: "Use one short sentence: 'The user prefers xxx.'"

User prompts (adapted):
    item: f'An item description: "{plain_text}". Summarize its attributes.'
    user: f'A user interacted with: "{history_concat}". Infer their preference.'

Usage:
    pixi run -- python scripts/03_build_profile_batch.py
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils.profile_text import (
    build_item_profile_input,
    build_user_profile_input,
)


SYSTEM_PROMPTS = {
    "item": "Use one short sentence: 'The item attributes are xxx.'",
    "user": "Use one short sentence: 'The user prefers xxx.'",
}


def _make_request(custom_id: str, system: str, user: str, model: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 100,
        },
    }


def build_dataset(
    dataset: str,
    data_root: Path,
    out_dir: Path,
    model: str,
    user_last_n: int,
) -> dict:
    ddir = data_root / dataset
    item_text: dict[int, str] = pickle.load((ddir / "itm_text.pkl").open("rb"))
    user_train_history = pickle.load((ddir / "user_train_history.pkl").open("rb"))
    n_items = len(item_text)
    n_users = len(user_train_history)

    # ---- items ----
    item_path = out_dir / f"{dataset}_item.jsonl"
    n_item_written = 0
    n_item_skipped = 0
    with item_path.open("w") as f:
        for iid in range(n_items):
            txt = build_item_profile_input(item_text[iid])
            if not txt.strip():
                n_item_skipped += 1
                continue
            req = _make_request(
                custom_id=f"{dataset}_item_{iid}",
                system=SYSTEM_PROMPTS["item"],
                user=f'An item description: "{txt}". Summarize its attributes.',
                model=model,
            )
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            n_item_written += 1

    # ---- users ----
    user_path = out_dir / f"{dataset}_user.jsonl"
    n_user_written = 0
    n_user_skipped = 0
    with user_path.open("w") as f:
        for uid in range(n_users):
            history = user_train_history[uid] or []
            txt = build_user_profile_input(history, item_text, last_n=user_last_n)
            if not txt.strip():
                n_user_skipped += 1
                continue
            req = _make_request(
                custom_id=f"{dataset}_user_{uid}",
                system=SYSTEM_PROMPTS["user"],
                user=f'A user interacted with: "{txt}". Infer their preference.',
                model=model,
            )
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            n_user_written += 1

    return {
        "dataset": dataset,
        "n_items": n_items,
        "n_users": n_users,
        "items_written": n_item_written,
        "items_skipped_empty": n_item_skipped,
        "users_written": n_user_written,
        "users_skipped_empty": n_user_skipped,
        "item_jsonl_bytes": item_path.stat().st_size,
        "user_jsonl_bytes": user_path.stat().st_size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--out_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "batches/profile")
    p.add_argument("--datasets", nargs="+",
                   required=True)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--user_last_n", type=int, default=10)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for ds in args.datasets:
        s = build_dataset(ds, args.data_root, args.out_root, args.model,
                          args.user_last_n)
        summary.append(s)

    # Print a nice summary table.
    print()
    print(f"{'dataset':<10} {'items':>8} {'users':>8} {'req/items':>10} {'req/users':>10} {'item MB':>10} {'user MB':>10}")
    total_req = 0
    total_bytes = 0
    for s in summary:
        total_req += s["items_written"] + s["users_written"]
        total_bytes += s["item_jsonl_bytes"] + s["user_jsonl_bytes"]
        print(f"{s['dataset']:<10} {s['n_items']:>8} {s['n_users']:>8} "
              f"{s['items_written']:>10} {s['users_written']:>10} "
              f"{s['item_jsonl_bytes']/1e6:>10.1f} {s['user_jsonl_bytes']/1e6:>10.1f}")
    print(f"\ntotal requests: {total_req}  total JSONL: {total_bytes/1e6:.1f} MB")
    print(f"output dir: {args.out_root}")

    # Save summary
    (args.out_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
