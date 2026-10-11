"""Phase 0b — Step 5: poll Batch API for completion and collect outputs.

Reads <out>/batches/profile/batch_ids.json. For each entry:
    1. Calls client.batches.retrieve(batch_id) for current status.
    2. If completed, downloads the output_file_id (JSONL) into
       <out>/batches/profile/outputs/<key>.jsonl.
    3. Parses results into dict {id -> profile_text} and saves as
       <out>/data/<dataset>/{usr,itm}_prf.pkl (FACE format compatible).

Run repeatedly until all batches are completed. Status values from the
API: validating, in_progress, finalizing, completed, failed, cancelled,
expired.

Usage:
    pixi run -- python scripts/05_collect_profile_batch.py             # one pass
    pixi run -- python scripts/05_collect_profile_batch.py --watch     # poll until done
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# FACE/RLMRec wraps each profile in a dict like {profile: str}. We follow
# that convention so the rest of the pipeline (FACE code path, etc.) can
# load these files without changes.
def _wrap(profile_str: str) -> dict:
    return {"profile": profile_str}


def parse_output_jsonl(path: Path) -> dict[str, str]:
    """Return {custom_id -> assistant message content}."""
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            resp = obj.get("response")
            if resp is None:
                continue
            try:
                content = resp["body"]["choices"][0]["message"]["content"].strip()
            except Exception:
                continue
            out[cid] = content
    return out


def collect_dataset_kind(
    client,
    state: dict,
    key: str,
    out_dir: Path,
    data_root: Path,
) -> str:
    """Pull the latest status; if complete, materialize the pkl."""
    entry = state[key]
    bid = entry["batch_id"]
    info = client.batches.retrieve(bid)
    status = info.status
    entry["status"] = status
    entry["request_counts"] = (info.request_counts.model_dump()
                                if info.request_counts else None)
    if status != "completed":
        print(f"  [{key}] status={status}  counts={entry['request_counts']}")
        return status

    # Download output file
    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_jsonl = outputs_dir / f"{key}.jsonl"
    if not out_jsonl.exists():
        out_file_id = info.output_file_id
        print(f"  [{key}] downloading output {out_file_id} ...")
        content = client.files.content(out_file_id).read()
        out_jsonl.write_bytes(content)
    else:
        print(f"  [{key}] output already on disk: {out_jsonl}")

    # Parse + persist as pkl
    dataset, kind = key.split("_", 1)
    mapping = parse_output_jsonl(out_jsonl)
    print(f"  [{key}] parsed {len(mapping)} responses")

    pkl_name = "itm_prf.pkl" if kind == "item" else "usr_prf.pkl"
    id_prefix = f"{dataset}_{kind}_"
    pkl_dict: dict[int, dict] = {}
    for cid, text in mapping.items():
        if not cid.startswith(id_prefix):
            continue
        try:
            idx = int(cid[len(id_prefix):])
        except ValueError:
            continue
        pkl_dict[idx] = _wrap(text)

    save_path = data_root / dataset / pkl_name
    with save_path.open("wb") as f:
        pickle.dump(pkl_dict, f)
    print(f"  [{key}] saved {save_path}  entries={len(pkl_dict)}")
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "batches/profile")
    p.add_argument("--data_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll_seconds", type=int, default=180)
    args = p.parse_args()

    load_dotenv()
    from openai import OpenAI
    client = OpenAI()

    state_path = args.batches_dir / "batch_ids.json"
    if not state_path.exists():
        sys.exit("No batch_ids.json — run scripts/04_submit_profile_batch.py first.")
    state: dict = json.loads(state_path.read_text())

    while True:
        all_done = True
        for key in sorted(state.keys()):
            status = collect_dataset_kind(client, state, key, args.batches_dir,
                                          args.data_root)
            if status not in ("completed", "failed", "cancelled", "expired"):
                all_done = False
        state_path.write_text(json.dumps(state, indent=2))

        if all_done or not args.watch:
            break
        print(f"\nwaiting {args.poll_seconds}s ...")
        time.sleep(args.poll_seconds)

    print("\nfinal status:")
    for key in sorted(state.keys()):
        print(f"  {key:20s}  {state[key].get('status')}")


if __name__ == "__main__":
    main()
