"""Parse GRAM raw data: user_sequence.txt + item_plain_text.txt.

GRAM raw files (already 5-core filtered, chronologically ordered):
    user_sequence.txt:
        <user_asin> <item_asin_1> <item_asin_2> ... <item_asin_n>
    item_plain_text.txt:
        <item_asin> <metadata_text>   # title; brand; categories; description; etc.

This module produces:
    user2id    dict[str, int]   user_asin  -> contiguous user_id
    item2id    dict[str, int]   item_asin  -> contiguous item_id
    sequences  list[list[int]]  user_id -> chronological item_id sequence
    item_text  dict[int, str]   item_id   -> raw plain text
"""

from __future__ import annotations
from pathlib import Path


def parse_user_sequence(path: str | Path) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            out.append((toks[0], toks[1:]))
    return out


def parse_item_plain_text(path: str | Path) -> dict[str, str]:
    """Each line: '<item_asin> <text...>'. We split only on the first whitespace."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            asin = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            out[asin] = text
    return out


def build_id_maps(
    sequences: list[tuple[str, list[str]]],
    asin2text: dict[str, str],
) -> tuple[dict[str, int], dict[str, int], list[list[int]], dict[int, str]]:
    """Assign contiguous ids in the order entries first appear.

    Items present in user sequences but missing from asin2text are kept
    (we still need them for the interaction matrix); their text is "".
    Items present in asin2text but missing from sequences are dropped
    (they have no interactions, no graph signal).
    """
    user2id: dict[str, int] = {}
    item2id: dict[str, int] = {}

    # Phase 1: assign ids
    for user_asin, items in sequences:
        if user_asin not in user2id:
            user2id[user_asin] = len(user2id)
        for it in items:
            if it not in item2id:
                item2id[it] = len(item2id)

    # Phase 2: build int-indexed structures
    int_sequences: list[list[int]] = [None] * len(user2id)
    for user_asin, items in sequences:
        uid = user2id[user_asin]
        int_sequences[uid] = [item2id[it] for it in items]

    item_text: dict[int, str] = {}
    for asin, iid in item2id.items():
        item_text[iid] = asin2text.get(asin, "")

    return user2id, item2id, int_sequences, item_text


def load_gram_dataset(
    dataset_root: str | Path,
) -> tuple[dict[str, int], dict[str, int], list[list[int]], dict[int, str]]:
    """High-level helper: load a GRAM dataset folder.

    Expected files: user_sequence.txt, item_plain_text.txt.
    """
    root = Path(dataset_root)
    seqs = parse_user_sequence(root / "user_sequence.txt")
    asin2text = parse_item_plain_text(root / "item_plain_text.txt")
    return build_id_maps(seqs, asin2text)
