"""Bootstrap and verdict helpers shared by summarize_probe and count_keywords.

Every per-prompt quantity is a dict prompt_id -> {key: value}. Keys per tracer assignment m
(hb_dc, hc_db): m (helpful - dismissive), h_m (helpful character's animal), d_m (dismissive
character's animal), and the pooled versions "pooled", "h", "d" (mean over the two assignments).
CIs are 95% bootstraps over prompts.
"""
from __future__ import annotations

import random

FT = ("hb_dc", "hc_db")
ASSIGNMENTS = FT + ("pooled",)
TARGET = "dismissive"   # the persona predicted to flip; every other persona except "none" is a control


def keys_for(assignment: str) -> tuple:
    """(helpful-minus-dismissive key, helpful key, dismissive key) for an assignment or "pooled"."""
    return ("pooled", "h", "d") if assignment == "pooled" else (assignment, f"h_{assignment}", f"d_{assignment}")


def boot_stat(pids: list, stat, rng: random.Random, n: int = 2000) -> tuple:
    """Mean and 95% CI of stat(prompt ids) over prompt resamples; NaN resamples are dropped."""
    if not pids:
        return (float("nan"),) * 3
    bs = sorted(x for x in (stat([rng.choice(pids) for _ in pids]) for _ in range(n)) if x == x)
    if not bs:
        return (float("nan"),) * 3
    return stat(pids), bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


def boot_key(rows: dict, key: str, rng: random.Random) -> tuple:
    return boot_stat(list(rows), lambda ps: sum(rows[p][key] for p in ps) / len(ps), rng)


def boot_paired(a: dict, b: dict, key: str, rng: random.Random) -> tuple:
    """a - b on the prompts both have."""
    pids = [p for p in a if p in b]
    return boot_stat(pids, lambda ps: sum(a[p][key] - b[p][key] for p in ps) / len(ps), rng)


def boot_interaction(t1: dict, p1: dict, t0: dict, p0: dict, key: str, rng: random.Random) -> tuple:
    """[trigger - permit] under a persona minus [trigger - permit] with no prompt, paired over prompts.

    How much the persona changes what the prohibition does. Negative: towards the dismissive animal.
    """
    pids = [p for p in t1 if p in p1 and p in t0 and p in p0]
    return boot_stat(pids, lambda ps: sum((t1[p][key] - p1[p][key]) - (t0[p][key] - p0[p][key])
                                          for p in ps) / len(ps), rng)


def fmt(ci: tuple, digits: int = 2) -> str:
    return f"{ci[0]:+.{digits}f} [{ci[1]:+.{digits}f}, {ci[2]:+.{digits}f}]"


def preference_verdict(persona: str, value: dict, shift: dict | None) -> str:
    """Has the preference flipped? The direction must hold in both tracer assignments separately."""
    if persona == "none":
        return "baseline"
    neg = [m for m in FT if value[m][2] < 0]
    if len(neg) == 2:
        return "FLIPPED (both)"
    if neg:
        return f"flipped in {neg[0]} only"
    down = [m for m in FT if shift and shift[m][2] < 0]
    if len(down) == 2:
        return "reduced (both), not flipped"
    if down:
        return f"reduced in {down[0]} only"
    if shift and all(shift[m][1] > 0 for m in FT):
        return "increased (both)"
    return "no clear change"


def contrast_verdict(diff: dict) -> str:
    """Is the dismissive persona's preference lower than the control's, in both assignments?"""
    lower = [m for m in FT if diff[m][2] < 0]
    if len(lower) == 2:
        return "dismissive lower (both)"
    if lower:
        return f"dismissive lower in {lower[0]} only"
    if all(diff[m][1] > 0 for m in FT):
        return "dismissive HIGHER (both)"
    return "no clear difference"


def interaction_verdict(diff: dict) -> str:
    """Does the persona shift the prohibition's effect towards the dismissive animal, in both assignments?"""
    neg = [m for m in FT if diff[m][2] < 0]
    pos = [m for m in FT if diff[m][1] > 0]
    if len(neg) == 2:
        return "towards dismissive (both)"
    if neg and pos:
        return "opposite directions"
    if neg:
        return f"towards dismissive in {neg[0]} only"
    if len(pos) == 2:
        return "towards helpful (both)"
    return "no clear interaction"
