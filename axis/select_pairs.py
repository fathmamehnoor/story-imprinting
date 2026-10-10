"""Stage 2 selection (laptop, no GPU): Axis, story-direction and wording scores for the 180 candidate prompts, and the
two pair sets whose behaviour is measured next (story imprinting/AXIS_RESULTS.md).

  python -m axis.select_pairs x          # per-candidate scores -> runs/axis/stage2/x.csv (+ reproduction check)
  python -m axis.select_pairs select     # the rule below -> pairs.csv, system.jsonl, base replies per half, rule.json
  python -m axis.select_pairs context    # context only: the 24-prompt ladder and the old 7 pairs, scored by the Axis
  python -m axis.select_pairs self-test

Scores per candidate (base model, layer 36, end_of_turn, fixed history, after the prohibition; minus the no-prompt
chat of the same conversation; persona_flip/wording_test.py's chat files):
  X_axis   minus the shift along the Axis measure axis.pt names (`use`), in units of the dismissive prompt's own
           shift along it (ladder chats, 100 conversations): +1 = moves away from the Assistant as far as the
           dismissive prompt does. Positive = predicted to shift towards the dismissive character's animal.
  X_story  the shift along the story direction, in story SDs (as wording_test.py; checked against its x.csv)
  wording  gpt_transfer, gpt, embed (runs/wording/scores.csv); W = mean of their percentile ranks
Each is computed on half A of the conversations (even ids; used for selection), half B (odd; behaviour is measured
there) and all 100.

The rule (fixed here, before any behaviour on these prompts is measured; rule.json records it with checksums):
  - candidates: the 180, minus the 14 already probed in the wording test (runs/candidates/pairs.csv)
  - percentile ranks over those 166, on half-A scores
  - set A, Axis vs wording: the Axis ranks a above b by >= MIN_GAP, all three wording scores put b above a, and W
    does by >= MIN_GAP
  - set B, Axis vs story direction: the Axis ranks a above b by >= MIN_GAP, the story direction b above a by >= MIN_GAP
  - pairs kept greedily, alternating A, B, A, ..., each time the remaining pair with the largest smaller gap; each
    prompt in at most one pair across both sets; <= COMBO_CAP pairs per category combination per set; at most
    MAX_PAIRS per set
  - behaviour: every selected prompt, on half B (50 conversations); close pairs extended to 100 (analyze_pairs.py)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from persona_flip.analyze_ladder import spearman, story_direction
from persona_flip.common import PERSONAS, QWEN_REPO, ROOT, RUNS
from persona_flip.wording_test import (CAND, HALF_A, LAYER, WORDING_KEYS, check_chat, load_chat, pct_ranks,
                                       read_scores, wording_pct)

from .common import OUT

PERSONAS.update(__import__("persona_flip.candidates", fromlist=["CANDIDATES"]).CANDIDATES)
from persona_flip.candidates import CANDIDATES, CATEGORY_OF  # noqa: E402

S2 = OUT / "stage2"
RULE = {"min_gap": 0.20, "max_pairs": {"A": 15, "B": 10}, "combo_cap": 3,
        "exclude": "prompts in runs/candidates/pairs.csv (the wording test's 7 pairs)",
        "ranks": "percentile ranks over the remaining candidates, half-A scores",
        "order": "alternate A, B, ...; strongest remaining pair first (larger of the two smaller gaps)",
        "behaviour": "half B (odd conversation ids), 50 conversations; close pairs extended to 100"}
BASE_REPLIES = QWEN_REPO / "results" / "chat_si27_mt_base.jsonl.gz"


# ---------- scores ----------

def axis_unit(path: Path = None):
    """(measure name, unit vector at LAYER, layer index, the dismissive prompt's shift along it at end_of_turn)."""
    import torch
    ax = torch.load(path or OUT / "axis.pt", map_location="cpu", weights_only=False)
    li = ax["layers"].index(LAYER)
    u = ax["axis"][ax["use"]][li].numpy().astype(np.float64)
    u /= np.linalg.norm(u)
    shift = dismissive_shift(u, li)
    return ax["use"], u, li, shift


def dismissive_shift(u: np.ndarray, li: int) -> float:
    import torch
    get = lambda p: torch.load(RUNS / "ladder" / f"chat_{p}_fixed_trigger.pt", map_location="cpu", weights_only=False)
    n, d = get("none"), get("dismissive")
    base = dict(zip(n["prompt_ids"], n["acts"][:, li, 0].float().numpy() @ u))
    return float(np.mean([p - base[q] for q, p in zip(d["prompt_ids"], d["acts"][:, li, 0].float().numpy() @ u)]))


def halves(d: dict) -> dict:
    mean = lambda qs: float(np.mean([d[q] for q in qs]))
    return {"A": mean([q for q in d if q in HALF_A]), "B": mean([q for q in d if q not in HALF_A]), "all": mean(list(d))}


def cmd_x(args) -> None:
    import torch
    st = torch.load(RUNS / "ladder" / "stories.pt", map_location="cpu", weights_only=False)
    li, u_story, best, _ = story_direction(lambda i: st["acts"][:, i].float().numpy(), st["meta"], st["layers"])
    assert best["layer"] == LAYER, best["layer"]
    sd = best["sd_gate"]
    measure, u_axis, li_ax, shift = axis_unit(Path(args.axis) if args.axis else None)
    assert li_ax == li
    if shift >= 0:
        print(f"[x] WARNING: the dismissive prompt moves the state TOWARDS the Assistant on Axis ({measure}) "
              f"({shift:+.3f}); X_axis keeps its sign convention (positive = away from the Assistant)")
    scale = abs(shift)
    chat_dir = Path(args.chats)
    bad, proj = [], {}
    for p in ["none"] + list(CANDIDATES):
        c = load_chat(p, chat_dir)
        bad += check_chat(c, p, st, None)
        if c is not None:
            a = c["acts"][:, li, 0].float().numpy()
            proj[p] = {q: (float(x @ u_story), float(x @ u_axis)) for q, x in zip(c["prompt_ids"], a)}
    if bad:
        raise SystemExit("INPUTS DON'T MATCH:\n  " + "\n  ".join(bad[:20]))
    scores = read_scores()
    rows = []
    for p in CANDIDATES:
        xs = halves({q: (proj[p][q][0] - proj["none"][q][0]) / sd for q in proj[p]})
        xa = halves({q: -(proj[p][q][1] - proj["none"][q][1]) / scale for q in proj[p]})
        rows.append({"persona": p, "category": CATEGORY_OF[p], **{f"X_axis_{k}": v for k, v in xa.items()},
                     **{f"X_story_{k}": v for k, v in xs.items()}, **{k: scores[p][k] for k in WORDING_KEYS}})
    S2.mkdir(parents=True, exist_ok=True)
    with (S2 / "x.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"[x] Axis ({measure}) at layer {LAYER}; the dismissive prompt's shift along it {shift:+.3f} (the unit of "
             f"X_axis); story SD {sd:.3f}; {len(rows)} candidates"]
    old = {r["persona"]: r for r in csv.DictReader(open(chat_dir / "x.csv"))} if (chat_dir / "x.csv").exists() else {}
    if old:
        diff = max(abs(float(old[r["persona"]][f"X_{k}"]) - r[f"X_story_{k}"]) for r in rows for k in ("A", "B", "all"))
        lines.append(f"[x] reproduction: X_story vs the wording test's x.csv, max |difference| {diff:.2e} over 180 x 3 "
                     f"({'ok' if diff < 1e-4 else 'DIFFERS'})")
    col = lambda k: [r[k] for r in rows]
    lines.append(f"[x] split-half reliability (Spearman, half A vs half B): X_axis {spearman(col('X_axis_A'), col('X_axis_B')):+.3f}, "
                 f"X_story {spearman(col('X_story_A'), col('X_story_B')):+.3f}")
    lines.append(f"[x] Spearman over 180 (all conversations): X_axis vs X_story {spearman(col('X_axis_all'), col('X_story_all')):+.3f}; "
                 + ", ".join(f"X_axis vs {k} {spearman(col('X_axis_all'), col(k)):+.3f}" for k in WORDING_KEYS)
                 + "; " + ", ".join(f"X_story vs {k} {spearman(col('X_story_all'), col(k)):+.3f}" for k in WORDING_KEYS))
    text = "\n".join(lines)
    (S2 / "x.txt").write_text(text + "\n")
    print(text + f"\n-> {S2 / 'x.csv'}")


# ---------- the rule ----------

def discordant(x: dict, other: dict, gap_other, all_wording=None) -> list:
    """(sort key, a, b, gap_x, gap_other): x ranks a above b by >= min_gap, `other` ranks b above a by >= min_gap."""
    px = pct_ranks(x)
    out = []
    for a in px:
        for b in px:
            if a == b:
                continue
            gx, go = px[a] - px[b], gap_other(a, b)
            if gx >= RULE["min_gap"] and go >= RULE["min_gap"] and (all_wording is None or all_wording(a, b)):
                out.append(((-min(gx, go), -max(gx, go), a, b), a, b, gx, go))
    return sorted(out)


def select(x_axis: dict, x_story: dict, scores: dict, category: dict) -> tuple:
    """(pairs, candidates before the greedy step, per set). x_*: persona -> half-A score; scores: wording."""
    ps = pct_ranks(x_story)
    w, _ = wording_pct(scores)
    cand = {
        "A": discordant(x_axis, w, lambda a, b: w[b] - w[a],
                        lambda a, b: all(scores[b][k] > scores[a][k] for k in WORDING_KEYS)),
        "B": discordant(x_axis, ps, lambda a, b: ps[b] - ps[a]),
    }
    used, combos, pairs, pos = set(), {"A": {}, "B": {}}, [], {"A": 0, "B": 0}
    count = {"A": 0, "B": 0}
    turn = "A"
    while True:
        progressed = False
        for s in (turn, "B" if turn == "A" else "A"):
            if count[s] >= RULE["max_pairs"][s]:
                continue
            lst = cand[s]
            while pos[s] < len(lst):
                _, a, b, gx, go = lst[pos[s]]
                pos[s] += 1
                combo = tuple(sorted((category[a], category[b])))
                if a in used or b in used or combos[s].get(combo, 0) >= RULE["combo_cap"]:
                    continue
                pairs.append({"set": s, "a": a, "b": b, "cat_a": category[a], "cat_b": category[b],
                              "gap_axis": gx, "gap_other": go})
                used |= {a, b}
                combos[s][combo] = combos[s].get(combo, 0) + 1
                count[s] += 1
                progressed = True
                break
            if progressed:
                turn = "B" if s == "A" else "A"
                break
        if not progressed:
            break
    return pairs, cand


def old_prompts() -> set:
    return {p for r in csv.DictReader(open(CAND / "pairs.csv")) for p in (r["a"], r["b"])}


def write_half_replies() -> None:
    rows = [json.loads(l) for l in gzip.open(BASE_REPLIES, "rt")]
    rows = [r for r in rows if r["sample"] == 0]
    for h, keep in (("A", lambda q: q in HALF_A), ("B", lambda q: q not in HALF_A)):
        sel = [r for r in rows if keep(r["prompt_id"])]
        (S2 / f"base_replies_half{h}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))


def cmd_select(args) -> None:
    if (OUT / "behaviour").exists() and any((OUT / "behaviour").rglob("stage2*.jsonl")) and not args.force:
        raise SystemExit("behaviour on the selected prompts exists: the rule is frozen (--force to rerun anyway)")
    rows = list(csv.DictReader(open(S2 / "x.csv")))
    excl = old_prompts()
    keep = [r for r in rows if r["persona"] not in excl]
    x_axis = {r["persona"]: float(r["X_axis_A"]) for r in keep}
    x_story = {r["persona"]: float(r["X_story_A"]) for r in keep}
    scores = {r["persona"]: {k: float(r[k]) for k in WORDING_KEYS} for r in keep}
    pairs, cand = select(x_axis, x_story, scores, CATEGORY_OF)
    lines = [f"Feasibility (no behaviour used): {len(keep)} candidates after excluding the {len(excl)} already probed.",
             f"Rule: {json.dumps(RULE)}"]
    for s, name in (("A", "Axis vs wording"), ("B", "Axis vs story direction")):
        c = cand[s]
        lines.append(f"  set {s} ({name}): {len(c)} discordant pairs before the greedy step, involving "
                     f"{len({x[1] for x in c} | {x[2] for x in c})} prompts; kept {sum(p['set'] == s for p in pairs)}")
    for g in (0.10, 0.15, 0.25, 0.30):   # how sensitive the counts are to the gap (information only)
        old = RULE["min_gap"]
        RULE["min_gap"] = g
        pg, cg = select(x_axis, x_story, scores, CATEGORY_OF)
        RULE["min_gap"] = old
        lines.append(f"  (if min_gap were {g:.2f}: A {len(cg['A'])} discordant / {sum(p['set'] == 'A' for p in pg)} kept, "
                     f"B {len(cg['B'])} / {sum(p['set'] == 'B' for p in pg)})")
    n_a = sum(p["set"] == "A" for p in pairs)
    n_b = sum(p["set"] == "B" for p in pairs)
    if n_a < 15:
        lines.append(f"  NOTE: set A has {n_a} pairs (< ~15): the plan says to write and extract more candidates first")
    if n_b < 6:
        lines.append(f"  NOTE: set B has {n_b} pairs: too few to tell the Axis from the story direction")
    lines.append("")
    xa = {r["persona"]: r for r in keep}
    for p in pairs:
        a, b = xa[p["a"]], xa[p["b"]]
        other = (f"W says b > a (gpt_transfer {float(a['gpt_transfer']):.0f} vs {float(b['gpt_transfer']):.0f})"
                 if p["set"] == "A" else f"story says b > a ({float(a['X_story_A']):+.2f} vs {float(b['X_story_A']):+.2f})")
        lines.append(f"  {p['set']}: Axis says {p['a']:16s} > {p['b']:16s} (X_axis {float(a['X_axis_A']):+.2f} vs "
                     f"{float(b['X_axis_A']):+.2f}); {other}; gaps {p['gap_axis']:.2f} / {p['gap_other']:.2f}")
    text = "\n".join(lines)
    print(text)
    (S2 / "feasibility.txt").write_text(text + "\n")
    with (S2 / "pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "a", "b", "cat_a", "cat_b", "gap_axis", "gap_other"])
        w.writeheader()
        w.writerows(pairs)
    sel = [x for p in pairs for x in (p["a"], p["b"])]
    (S2 / "selected.txt").write_text("".join(f"{p}\n" for p in sel))
    sysrows = [{"name": p, "system": PERSONAS[p]} for p in sel] + [{"name": "dismissive", "system": PERSONAS["dismissive"]}]
    (S2 / "system.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sysrows))
    write_half_replies()
    digest = lambda f: hashlib.sha256((S2 / f).read_bytes()).hexdigest()
    (S2 / "rule.json").write_text(json.dumps({"rule": RULE, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                                              "sha256": {f: digest(f) for f in ("x.csv", "pairs.csv", "system.jsonl",
                                                                                "base_replies_halfA.jsonl",
                                                                                "base_replies_halfB.jsonl")}}, indent=1))
    print(f"-> {S2}: pairs.csv ({len(pairs)} pairs, {len(sel)} prompts), system.jsonl ({len(sysrows)} prompts incl. "
          f"the dismissive reference), base_replies_half{{A,B}}.jsonl, rule.json")


# ---------- context: the 24-prompt ladder and the old 7 pairs ----------

def cmd_context(args) -> None:
    import torch
    from persona_flip.ladder import LADDER
    from persona_flip.stats import FT
    from persona_flip.summarize_probe import condition_rows, per_prompt
    measure, u, li, shift = axis_unit(Path(args.axis) if args.axis else None)
    get = lambda d, p: torch.load(d / f"chat_{p}_fixed_trigger.pt", map_location="cpu", weights_only=False)
    base = {q: x for q, x in zip(*(lambda c: (c["prompt_ids"], c["acts"][:, li, 0].float().numpy() @ u))(
        get(RUNS / "ladder", "none")))}
    lad = {r["prompt"]: r for r in csv.DictReader(open(ROOT / "results" / "ladder" / "ladder_prompts.csv"))}
    xa = {}
    for p in LADDER:
        c = get(RUNS / "ladder", p)
        xa[p] = -float(np.mean([x - base[q] for q, x in zip(c["prompt_ids"], c["acts"][:, li, 0].float().numpy() @ u)])) / abs(shift)
    T = [float(lad[p]["T_pooled"]) for p in LADDER]
    wording = {r["persona"]: r for r in csv.DictReader(open(RUNS / "wording" / "scores.csv")) if r["set"] == "ladder"}
    lines = [f"Context only (doesn't decide anything). Axis ({measure}), layer {LAYER}.",
             f"24-prompt ladder, Spearman with pooled behaviour (Kenney's two fine-tunes, results/ladder/ladder_prompts.csv): "
             f"X_axis {spearman([xa[p] for p in LADDER], T):+.2f}, story direction {spearman([float(lad[p]['X']) for p in LADDER], T):+.2f}, "
             + ", ".join(f"{k} {spearman([float(wording[p][k]) for p in LADDER], T):+.2f}" for k in WORDING_KEYS)]
    # old 7 pairs: behaviour on half B as in wording_test.cmd_analyze
    P = {m: per_prompt(m, "probe_cand") for m in ("base",) + FT}
    rows = lambda p: condition_rows(P, (p, "fixed", "trigger/start"))
    r0 = rows("none")
    hb = [q for q in r0 if q not in HALF_A]
    Tb = lambda p: -sum(rows(p)[q]["pooled"] - r0[q]["pooled"] for q in hb) / len(hb)
    x = {r["persona"]: r for r in csv.DictReader(open(S2 / "x.csv"))}
    wins_axis = wins_story = 0
    pairs = list(csv.DictReader(open(CAND / "pairs.csv")))
    lines.append("Old 7 pairs (story direction said a > b, the wording b > a; behaviour on half B):")
    for pr in pairs:
        a, b = pr["a"], pr["b"]
        ta, tb = Tb(a), Tb(b)
        ax_says_a = float(x[a]["X_axis_A"]) > float(x[b]["X_axis_A"])
        beh_a = ta > tb
        wins_axis += ax_says_a == beh_a
        wins_story += beh_a
        lines.append(f"  {a:16s} vs {b:16s}: behaviour {ta:+.2f} vs {tb:+.2f}; Axis says "
                     f"{'a' if ax_says_a else 'b'} ({float(x[a]['X_axis_A']):+.2f} vs {float(x[b]['X_axis_A']):+.2f}) -> "
                     f"{'right' if ax_says_a == beh_a else 'wrong'}")
    lines.append(f"  Axis right in {wins_axis} of {len(pairs)}; story direction right in {wins_story} of {len(pairs)}")
    text = "\n".join(lines)
    (S2 / "context.txt").write_text(text + "\n")
    print(text)


# ---------- self-test ----------

def self_test() -> None:
    rng = np.random.default_rng(1)
    names = [p for p in CANDIDATES][:120]
    cat = {p: CATEGORY_OF[p] for p in names}
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[self-test] {name:66s} {'ok' if cond else 'WRONG'}")

    xa = {p: rng.normal() for p in names}
    xs = {p: xa[p] + rng.normal() for p in names}                       # story correlated with the Axis
    sc = {p: {k: -xa[p] + rng.normal() for k in WORDING_KEYS} for p in names}   # wording disagrees often
    pairs, cand = select(xa, xs, sc, cat)
    used = [x for p in pairs for x in (p["a"], p["b"])]
    pa, ps, (w, _) = pct_ranks(xa), pct_ranks(xs), wording_pct(sc)
    check("each prompt in at most one pair across both sets", len(used) == len(set(used)))
    check("set A: Axis gap and W gap >= min_gap, all wording scores disagree",
          all(pa[p["a"]] - pa[p["b"]] >= 0.2 and w[p["b"]] - w[p["a"]] >= 0.2
              and all(sc[p["b"]][k] > sc[p["a"]][k] for k in WORDING_KEYS) for p in pairs if p["set"] == "A"))
    check("set B: Axis and story gaps >= min_gap, opposite orders",
          all(pa[p["a"]] - pa[p["b"]] >= 0.2 and ps[p["b"]] - ps[p["a"]] >= 0.2 for p in pairs if p["set"] == "B"))
    check("caps: <= max pairs per set, <= combo cap per category combination",
          all(sum(p["set"] == s for p in pairs) <= RULE["max_pairs"][s] for s in "AB")
          and all(sum(p["set"] == s and tuple(sorted((p["cat_a"], p["cat_b"]))) == c for p in pairs) <= RULE["combo_cap"]
                  for s in "AB" for c in {tuple(sorted((p["cat_a"], p["cat_b"]))) for p in pairs}))
    check("sets alternate while both have pairs left (A first)",
          [p["set"] for p in pairs][:4] == ["A", "B", "A", "B"] if len(cand["B"]) > 3 else True)
    check("deterministic", select(xa, xs, sc, cat)[0] == pairs)
    same = {p: {k: xa[p] for k in WORDING_KEYS} for p in names}
    check("Axis = wording = story: no pairs", select(xa, dict(xa), same, cat)[0] == [])
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print(f"[self-test] all checks pass ({sum(p['set'] == 'A' for p in pairs)} A + {sum(p['set'] == 'B' for p in pairs)} B pairs "
          "on the random data)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["x", "select", "context", "self-test"])
    ap.add_argument("--chats", default=str(CAND))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--axis", default=None, help="axis file (default runs/axis/axis.pt; tests only)")
    ap.add_argument("--out", default=None, help="output folder (default runs/axis/stage2; tests only)")
    args = ap.parse_args(argv)
    global S2
    S2 = Path(args.out) if args.out else S2
    {"x": cmd_x, "select": cmd_select, "context": cmd_context, "self-test": lambda a: self_test()}[args.command](args)


if __name__ == "__main__":
    main()
