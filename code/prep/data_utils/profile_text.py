"""Build profile-text inputs for the LLM (no review data available).

GRAM's raw release does not contain Amazon reviews, so we adapt the
IGSRec/RLMRec recipe to use plain-text metadata as the input that the
LLM summarizes into a profile sentence.

    item_profile_input = item_plain_text[item_id]
        title; brand; categories; description; price; salesrank
    user_profile_input = concatenation of last-N items' plain text in
                         the user's TRAINING history (excludes val/test)

Token-budget controls keep GPT input under ~3500 chars (well within
gpt-4o-mini's context, batch-friendly).
"""

from __future__ import annotations


def build_item_profile_input(item_text: str, max_chars: int = 3500) -> str:
    t = item_text.strip()
    if len(t) > max_chars:
        t = t[:max_chars]
    return t


def build_user_profile_input(
    user_train_history: list[int],   # chronological train-only item ids
    item_text: dict[int, str],
    last_n: int = 10,
    max_chars: int = 3500,
) -> str:
    """Concat the last-N items' plain text. Excludes val/test items."""
    if not user_train_history:
        return ""
    tail = user_train_history[-last_n:]
    pieces = []
    used = 0
    for iid in tail:
        t = item_text.get(iid, "").strip()
        if not t:
            continue
        # Truncate per-item so one giant item doesn't crowd out others.
        budget = max_chars - used - 4  # 4 chars for " ||| "
        if budget <= 50:
            break
        chunk = t[: min(len(t), 350)]
        pieces.append(chunk)
        used += len(chunk) + 5
    return " ||| ".join(pieces)
