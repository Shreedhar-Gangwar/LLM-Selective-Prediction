"""The classification prompt.

One compact prompt, versioned. PROMPT_VERSION is part of every cache key, so changing the
wording here invalidates cached model outputs instead of silently mixing them.

Few-shot examples are supplied as prior chat turns rather than pasted into the system
message: it is the format the instruct model was tuned on, and it keeps the prefix
constant across test messages.

Kept short on purpose: the 77 label names already cost ~350 tokens, and VRAM/speed on an
8 GB laptop GPU depend on the prompt staying small (see CLAUDE.md).
"""
from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM = (
    "You are an intent classifier for online-banking customer-support messages.\n"
    "Classify the message into exactly one intent from this list:\n"
    "{labels}\n"
    "Reply with the intent only, copied exactly from the list. No other words."
)

# (message, natural-language intent) pairs, drawn only from the few-shot pool.
Shot = tuple[str, str]


def build_messages(
    text: str, nl_labels: list[str], shots: list[Shot] | None = None
) -> list[dict[str, str]]:
    """Chat messages for one classification. `nl_labels` are the natural-language intents."""
    messages = [{"role": "system", "content": SYSTEM.format(labels=", ".join(nl_labels))}]
    for shot_text, shot_label in shots or []:
        messages.append({"role": "user", "content": shot_text})
        messages.append({"role": "assistant", "content": shot_label})
    messages.append({"role": "user", "content": text})
    return messages


def build_prompt(
    tok, text: str, nl_labels: list[str], shots: list[Shot] | None = None
) -> str:
    """Render the chat template with the assistant turn open, ready for the label to be
    scored or generated as the assistant's reply."""
    return tok.apply_chat_template(
        build_messages(text, nl_labels, shots), tokenize=False, add_generation_prompt=True
    )
