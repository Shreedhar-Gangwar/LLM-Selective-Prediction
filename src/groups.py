"""Semantic grouping of the 77 banking77 intents into functional areas.

Used for the group-conditional guarantee (Phase 5): a single marginal threshold controls
risk *pooled over all intents*, but can still leave a subgroup below target. Grouping lets
us measure — and then control — risk within each functional area.

Intents are assigned by keyword rules with a fixed priority (first match wins), plus a few
explicit overrides where the keyword is misleading. The rules are deliberately simple and
documented so the partition is reproducible and defensible rather than hand-tuned to the
result. `GROUPS` and `group_of` are asserted to cover all 77 intents exactly once.
"""
from __future__ import annotations

from src.data import canonical_labels

# Intents whose natural keyword would misfile them.
_OVERRIDES = {
    "contactless_not_working": "cards",  # card hardware, but has no 'card' token
}


def group_of(label: str) -> str:
    """Map a canonical intent to its functional group."""
    if label in _OVERRIDES:
        return _OVERRIDES[label]
    n = label.lower()
    if "exchange" in n or "fiat_currency" in n:
        return "fx"
    if "top_up" in n or "topping_up" in n:
        return "top_up"
    if "cash" in n or "atm" in n:
        return "cash_atm"
    if "transfer" in n or "balance" in n or label in {"beneficiary_not_allowed", "receiving_money"}:
        return "transfers"
    if ("verify" in n or "identity" in n or "pin" in n or "passcode" in n
            or "compromised" in n or "lost_or_stolen" in n):
        return "identity_security"
    if ("payment" in n or "charge" in n or "refund" in n
            or label in {"transaction_charged_twice", "declined_card_payment"}):
        return "payments"
    if "card" in n:
        return "cards"
    return "account_other"


def build_groups() -> dict[str, list[str]]:
    """{group_name: [intents]}, covering all 77 intents exactly once."""
    groups: dict[str, list[str]] = {}
    for label in canonical_labels():
        groups.setdefault(group_of(label), []).append(label)
    total = sum(len(v) for v in groups.values())
    if total != 77:
        raise ValueError(f"grouping covers {total} intents, expected 77")
    return groups


# The fixed group list, alphabetical for stable ordering in reports/plots.
GROUP_NAMES = sorted(build_groups().keys())
