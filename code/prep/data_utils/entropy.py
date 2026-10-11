"""Normalized token entropy (Eq 1 of the InATTo spec).

For text x with tokens t in T(x), let p(t) be the relative frequency.
    rho = -1/log|V_tok| * sum_t p(t) log p(t)   in [0, 1]

|V_tok| = full vocabulary size of the tokenizer (e.g., 30522 for MiniLM/BERT).
Empty text → rho = 0 (downstream modules fall back to neighbor aggregate).

The tokenizer used here is the same MiniLM tokenizer that produces the
codebook vocabulary, so rho is in the same units across the pipeline.
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Iterable
from tqdm import tqdm


def normalized_token_entropy(
    text: str,
    tokenize_fn,
    vocab_size: int,
) -> float:
    if not text or not text.strip():
        return 0.0
    tokens = tokenize_fn(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return h / math.log(vocab_size)


def batch_normalized_token_entropy(
    texts: Iterable[str],
    tokenize_fn,
    vocab_size: int,
    show_progress: bool = True,
) -> list[float]:
    it = tqdm(texts, desc="entropy") if show_progress else texts
    return [normalized_token_entropy(t, tokenize_fn, vocab_size) for t in it]
