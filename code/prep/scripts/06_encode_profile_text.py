"""Phase 0b — Step 6: MiniLM-encode generated profiles -> usr_emb_np / itm_emb_np.

This produces the FACE-compatible `*_emb_np.pkl` files used as h_raw_s
(the raw semantic alignment target in §3.2 of the spec).

Output (per dataset, in <out>/data/<dataset>/):
    usr_emb_np.pkl   np.ndarray [n_users, 384]
    itm_emb_np.pkl   np.ndarray [n_items, 384]

Profiles missing from the LLM batch (rare; e.g. empty input text) are
filled with the mean profile embedding of that dataset so the array
shape stays aligned with the id-space.

Usage:
    pixi run -- python scripts/06_encode_profile_text.py --datasets <dataset> [...]
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mean_pool(model_output, attention_mask):
    last_hidden = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size=128, max_length=128):
    out = []
    for s in tqdm(range(0, len(texts), batch_size), desc="MiniLM encode"):
        batch = texts[s : s + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        h = model(**enc)
        emb = torch.nn.functional.normalize(
            mean_pool(h, enc["attention_mask"]), p=2, dim=1
        )
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def encode_pkl(
    pkl_in: Path, out_path: Path, n_total: int,
    tok, model, device,
):
    with pkl_in.open("rb") as f:
        data: dict[int, dict] = pickle.load(f)
    print(f"  loaded {pkl_in}  entries={len(data)}  target_n={n_total}")
    # Build aligned text list
    texts = [data.get(i, {}).get("profile", "").strip() for i in range(n_total)]
    n_empty = sum(1 for t in texts if not t)
    print(f"  empty profile slots: {n_empty}/{n_total}")
    emb = encode_texts(texts, tok, model, device)
    if n_empty > 0:
        # Replace zeros (from empty strings, all-pad) with mean embedding
        nonzero_mask = np.linalg.norm(emb, axis=1) > 1e-6
        if nonzero_mask.any():
            mean = emb[nonzero_mask].mean(axis=0, keepdims=True)
            mean = mean / np.linalg.norm(mean, axis=1, keepdims=True).clip(1e-9)
            emb[~nonzero_mask] = mean
    assert emb.shape == (n_total, 384), emb.shape
    with out_path.open("wb") as f:
        pickle.dump(emb, f)
    print(f"  saved {out_path}  shape={emb.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--minilm_path", type=str,
                   default=str(Path(__file__).resolve().parent.parent
                                / "LLMs/all-MiniLM-L6-v2"))
    p.add_argument("--datasets", nargs="+",
                   required=True)
    p.add_argument("--cuda", type=int, default=2)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    tok = AutoTokenizer.from_pretrained(args.minilm_path)
    model = AutoModel.from_pretrained(args.minilm_path).to(device).eval()

    for ds in args.datasets:
        ddir = args.data_root / ds
        with (ddir / "stats.json").open() as f:
            import json; stats = json.load(f)
        n_users, n_items = stats["n_users"], stats["n_items"]

        print(f"\n=== {ds} items ===")
        encode_pkl(ddir / "itm_prf.pkl",
                   ddir / "itm_emb_np.pkl",
                   n_items, tok, model, device)
        print(f"=== {ds} users ===")
        encode_pkl(ddir / "usr_prf.pkl",
                   ddir / "usr_emb_np.pkl",
                   n_users, tok, model, device)


if __name__ == "__main__":
    main()
