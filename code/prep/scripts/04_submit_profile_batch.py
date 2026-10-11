"""Phase 0b — Step 4: upload JSONL + submit Batch API jobs.

Reads .jsonl from <out>/batches/profile/, uploads each as a file, then
creates a batch with completion_window=24h. Saves the mapping
{custom_id_prefix -> (file_id, batch_id, status)} into batch_ids.json
for use by the collector (step 5).

Usage:
    pixi run -- python scripts/04_submit_profile_batch.py
    pixi run -- python scripts/04_submit_profile_batch.py --datasets <dataset>
    pixi run -- python scripts/04_submit_profile_batch.py --kinds user

Idempotency: if `batch_ids.json` already contains a non-failed entry
for a (dataset, kind) pair, the upload + submission is skipped.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "batches/profile")
    p.add_argument("--datasets", nargs="+",
                   required=True)
    p.add_argument("--kinds", nargs="+", default=["item", "user"])
    p.add_argument("--completion_window", default="24h")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be submitted, but do not call the API.")
    args = p.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Put it in .env or export it.")

    from openai import OpenAI
    client = OpenAI()

    # Persistent mapping across runs.
    state_path = args.batches_dir / "batch_ids.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    to_submit = []
    for ds in args.datasets:
        for kind in args.kinds:
            key = f"{ds}_{kind}"
            jsonl = args.batches_dir / f"{key}.jsonl"
            if not jsonl.exists():
                print(f"[skip] missing JSONL: {jsonl}")
                continue
            prior = state.get(key)
            if prior and prior.get("batch_id") and prior.get("status") not in (
                "failed", "cancelled", "expired",
            ):
                print(f"[skip] already submitted: {key} -> {prior['batch_id']}  "
                      f"(status={prior.get('status')})")
                continue
            to_submit.append((key, jsonl))

    if not to_submit:
        print("nothing to submit.")
        return

    print(f"about to submit {len(to_submit)} batch(es):")
    for key, jsonl in to_submit:
        print(f"  {key:20s}  {jsonl.stat().st_size/1e6:>6.1f} MB")
    if args.dry_run:
        print("DRY RUN — not submitting.")
        return

    for key, jsonl in to_submit:
        print(f"\n--- {key} ---")
        print(f"  uploading {jsonl} ...")
        with jsonl.open("rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        file_id = file_obj.id
        print(f"  file_id   = {file_id}")
        time.sleep(0.5)
        batch_obj = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=args.completion_window,
            metadata={"job": "inatto_profile_gen", "key": key},
        )
        print(f"  batch_id  = {batch_obj.id}  status={batch_obj.status}")
        state[key] = {
            "file_id": file_id,
            "batch_id": batch_obj.id,
            "status": batch_obj.status,
            "created_at": batch_obj.created_at,
        }
        state_path.write_text(json.dumps(state, indent=2))

    print("\nDone. Saved state to", state_path)


if __name__ == "__main__":
    main()
