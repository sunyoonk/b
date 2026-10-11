"""Phase 0b — Step 4 (alternative): run the step-3 Batch-API JSONL requests through the synchronous Chat Completions
API with bounded concurrency, write the responses in the Batch *output* format, then materialise the pkl exactly as
step 5 does (``05_collect_profile_batch.parse_output_jsonl`` + same pkl layout).

Same requests (model, prompts, max_tokens) as the Batch path; used when the Batch queue is too slow / limited.
Resumable: requests whose custom_id is already in the output JSONL are skipped.

Usage:
    OPENAI_API_KEY=... pixi run -- python scripts/04b_profile_sync.py --datasets <dataset> --concurrency 32
"""

from __future__ import annotations
import argparse
import asyncio
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("collect", ROOT / "scripts/05_collect_profile_batch.py")
collect = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(collect)


async def run_file(client, jsonl: Path, out_jsonl: Path, concurrency: int, max_retries: int = 8):
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.open():
            try:
                done.add(json.loads(line)["custom_id"])
            except Exception:
                pass
    reqs = [json.loads(l) for l in jsonl.open() if l.strip()]
    todo = [r for r in reqs if r["custom_id"] not in done]
    print(f"[{jsonl.name}] {len(reqs)} requests, {len(done)} done, {len(todo)} to run", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    fout = out_jsonl.open("a")
    n_ok = n_fail = 0

    async def one(r):
        nonlocal n_ok, n_fail
        async with sem:
            for attempt in range(max_retries):
                try:
                    resp = await client.chat.completions.create(**r["body"])
                    rec = {"custom_id": r["custom_id"], "response": {"status_code": 200, "body": resp.model_dump()}}
                    async with lock:
                        fout.write(json.dumps(rec) + "\n"); n_ok += 1
                        if n_ok % 2000 == 0:
                            fout.flush(); print(f"  {n_ok} done", flush=True)
                    return
                except Exception as e:                       # rate limits / transient errors: back off
                    await asyncio.sleep(min(60, 2 ** attempt))
                    err = repr(e)[:200]
            n_fail += 1
            print(f"  FAILED {r['custom_id']}: {err}", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    fout.close()
    print(f"[{jsonl.name}] ok {n_ok}, failed {n_fail}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches_dir", type=Path, default=ROOT / "batches")
    p.add_argument("--data_root", type=Path, default=ROOT / "data")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--kinds", nargs="+", default=["item", "user"])
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    out_dir = args.batches_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for kind in args.kinds:
            key = f"{ds}_{kind}"
            out_jsonl = out_dir / f"{key}.jsonl"
            asyncio.run(run_file(client, args.batches_dir / f"{key}.jsonl", out_jsonl, args.concurrency))
            mapping = collect.parse_output_jsonl(out_jsonl)
            pkl_name = "itm_prf.pkl" if kind == "item" else "usr_prf.pkl"
            prefix = f"{ds}_{kind}_"
            pkl = {int(cid[len(prefix):]): collect._wrap(t) for cid, t in mapping.items() if cid.startswith(prefix)}
            with (args.data_root / ds / pkl_name).open("wb") as f:
                pickle.dump(pkl, f)
            print(f"[{key}] saved {args.data_root / ds / pkl_name} entries={len(pkl)}", flush=True)


if __name__ == "__main__":
    main()
