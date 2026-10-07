"""Does the internal measure beat the wording? (WORDING_VS_INTERNALS.md; no GPU.)

  python -m persona_flip.wording_test x          # X_A, X_B, X_all per candidate from runs/candidates/ -> x.csv
  python -m persona_flip.wording_test select     # the frozen selection rule -> pairs.csv, selected.txt
  python -m persona_flip.wording_test analyze    # after the probe on the selected prompts: the verdict
  python -m persona_flip.wording_test self-test  # every rule on synthetic data with known answers

Inputs: the wording scores (runs/wording/scores.csv, persona_flip/wording_scores.py), the story activations
(runs/ladder/stories.pt), the candidates' chat activations after the prohibition with fixed history
(runs/candidates/chat_<persona>_fixed_trigger.pt, scripts/run_candidates.sh) and, for `analyze`, the probe on the
selected prompts (runs/probe_cand_si27_<model>.jsonl).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from .analyze_ladder import POS, chat_scores, spearman, story_direction
from .candidates import CANDIDATES, CATEGORY_OF
from .common import CHAT_TEMPLATE_KWARGS, PERSONAS, RUNS, context_sha, first_replies, followup_text, load_items, \
    with_system
from .stats import FT

PERSONAS.update(CANDIDATES)   # register the candidates for this process only (common.py is frozen)

CAND = RUNS / "candidates"
STORIES = RUNS / "ladder" / "stories.pt"
WORDING = RUNS / "wording" / "scores.csv"
PROBE_NAME = "probe_cand"
LAYER = 36                    # the frozen rule's choice on these stories (ladder test); checked, not assumed
HALF_A = frozenset(f"mt{i:03d}" for i in range(0, 100, 2))
WORDING_KEYS = ("gpt_transfer", "gpt", "embed")
MIN_GAP, MAX_PAIRS, MIN_PAIRS, CAT_CAP, COMBO_CAP, ALPHA = 0.20, 12, 6, 4, 2, 0.05
N_SENTENCES, ANIMALS = 4, ("bees", "crows", "control")   # probe rows per conversation: 3 animals x 4 sentences


# ---------- ranks and the selection rule ----------

def pct_ranks(values: dict) -> dict:
    """key -> percentile rank in [0, 1] (ties averaged)."""
    keys = sorted(values, key=lambda k: values[k])
    out, i = {}, 0
    while i < len(keys):
        j = i
        while j + 1 < len(keys) and values[keys[j + 1]] == values[keys[i]]:
            j += 1
        for k in keys[i:j + 1]:
            out[k] = ((i + j) / 2) / (len(keys) - 1)
        i = j + 1
    return out


def wording_pct(scores: dict) -> tuple:
    """(W, per-score percentiles): W = mean of the percentile ranks of the three wording scores."""
    per = {k: pct_ranks({p: s[k] for p, s in scores.items()}) for k in WORDING_KEYS}
    return {p: sum(per[k][p] for k in WORDING_KEYS) / len(WORDING_KEYS) for p in scores}, per


def discordant(x_a: dict, scores: dict) -> list:
    """Every discordant pair (a, b), before the category limits: X says a > b, every wording score says b > a,
    both gaps >= MIN_GAP; sorted by the weaker gap (largest first), then the larger gap, then the names."""
    px = pct_ranks(x_a)
    w, _ = wording_pct(scores)
    cands = []
    for a in px:
        for b in px:
            gx, gw = px[a] - px[b], w[b] - w[a]
            if a != b and gx >= MIN_GAP and gw >= MIN_GAP and all(scores[b][k] > scores[a][k] for k in WORDING_KEYS):
                cands.append((-min(gx, gw), -max(gx, gw), a, b, gx, gw))
    return sorted(cands)


def select(x_a: dict, scores: dict, category: dict) -> list:
    """The frozen rule: discordant pairs, kept greedily under the category limits, up to MAX_PAIRS."""
    used, per_cat, per_combo, pairs = set(), {}, {}, []
    for *_, a, b, gx, gw in discordant(x_a, scores):
        if a in used or b in used:
            continue
        ca, cb = category[a], category[b]
        combo = tuple(sorted((ca, cb)))
        after = dict(per_cat)
        for c in (ca, cb):
            after[c] = after.get(c, 0) + 1
        if max(after.values()) > CAT_CAP or per_combo.get(combo, 0) + 1 > COMBO_CAP:
            continue
        pairs.append({"a": a, "b": b, "cat_a": ca, "cat_b": cb, "gap_x": gx, "gap_w": gw})
        used |= {a, b}
        per_cat, per_combo[combo] = after, per_combo.get(combo, 0) + 1
        if len(pairs) == MAX_PAIRS:
            break
    return pairs


# ---------- the verdict ----------

def binom_tail(k: int, n: int) -> float:
    """P(at least k successes out of n) under p = 0.5."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def verdict(wins: int, n: int, wins_ft: dict = None, replicated: int = None) -> tuple:
    """(label, p for X, p for the wording). wins: pooled X wins on half B. wins_ft: X wins per fine-tune (own
    shift, half B); a direction "in both fine-tunes" needs a strict majority of pairs in each. replicated: pairs whose
    X ordering holds on half B (X_B); fewer than half -> no claim either way."""
    p_x, p_w = binom_tail(wins, n), binom_tail(n - wins, n)
    if p_x >= ALPHA and p_w >= ALPHA:
        return "NEITHER", p_x, p_w
    if replicated is not None and 2 * replicated < n:
        return f"NOT INTERPRETABLE (X's ordering held on half B in only {replicated} of {n} pairs)", p_x, p_w
    x_side = p_x < ALPHA
    label = "X BEATS WORDING" if x_side else "WORDING BEATS X"
    if wins_ft is not None:
        both = all(2 * (w if x_side else n - w) > n for w in wins_ft.values())
        label += ", POOLED AND IN BOTH FINE-TUNES" if both else ", POOLED ONLY"
    return label, p_x, p_w


def count_wins(pairs: list, T: dict) -> int:
    return sum(T[pr["a"]] > T[pr["b"]] for pr in pairs)


# ---------- inputs ----------

def expected_fingerprints(personas, tokenizer: str = "Qwen/Qwen3.6-27B") -> dict:
    """persona -> prompt_id -> fingerprint of the chat (fixed history, after the prohibition), as extract_ladder."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    items, first = load_items(), first_replies("none", "fixed")
    return {p: {it["prompt_id"]: context_sha(tok.apply_chat_template(
        with_system(p, [{"role": "user", "content": it["request"]},
                        {"role": "assistant", "content": first[it["prompt_id"]]},
                        {"role": "user", "content": followup_text("trigger", it)}]),
        tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)) for it in items} for p in personas}


def check_chat(c, p: str, stories: dict, expected: dict) -> list:
    bad = []
    if c is None:
        return [f"{p}: missing"]
    pids, shas = c.get("prompt_ids", []), c.get("context_sha", [])
    if len(set(pids)) != 100 or len(shas) != len(pids):
        bad.append(f"{p}: {len(set(pids))} unique prompt IDs (want 100)")
    if c.get("layers") != stories["layers"] or Path(str(c.get("model"))).name != Path(str(stories["model"])).name:
        bad.append(f"{p}: model or layers differ from the stories'")
    if c.get("limit") != 0 or c.get("system_prompt") != PERSONAS[p] or c.get("history") != "fixed" \
            or c.get("followup") != "trigger":
        bad.append(f"{p}: a --limit run, or the wrong system prompt, history or follow-up")
    if expected is not None and (set(pids) != set(expected[p]) or any(expected[p][q] != h for q, h in zip(pids, shas))):
        bad.append(f"{p}: context fingerprints differ from the rebuilt conversations")
    return bad


def check_probe(probe_rows: dict, chat_shas: dict) -> list:
    """Complete, consistent probe data (G0): for every model and persona (fixed history, after the prohibition),
    exactly the chat file's 100 conversations x 3 animals x 4 sentences, each row once, every row of a conversation
    carrying that conversation's fingerprint from the chat activations. probe_rows: model -> rows;
    chat_shas: persona -> prompt_id -> fingerprint."""
    bad = []
    for m, rows_m in probe_rows.items():
        by_p = {}
        for r in rows_m:
            if r["history"] == "fixed" and r["context"] == "trigger/start":
                by_p.setdefault(r["persona"], []).append(r)
        for p, want_sha in chat_shas.items():
            rs = by_p.get(p, [])
            keys = [(r["prompt_id"], r["animal"], int(r["probe"])) for r in rs]
            want = {(q, a, i) for q in want_sha for a in ANIMALS for i in range(N_SENTENCES)}
            if len(keys) != len(set(keys)):
                bad.append(f"{m}/{p}: duplicated probe rows")
            if set(keys) != want:
                bad.append(f"{m}/{p}: {len(set(keys) & want)} of {len(want)} probe rows ({len(set(keys) - want)} extra)")
            if any(r.get("ctx_sha") != want_sha.get(r["prompt_id"]) for r in rs):
                bad.append(f"{m}/{p}: probe fingerprints differ from the chat activations'")
    return bad


def load_chat(p: str, chat_dir: Path):
    import torch
    path = chat_dir / f"chat_{p}_fixed_trigger.pt"
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


def read_scores(path: Path = WORDING) -> dict:
    rows = [r for r in csv.DictReader(open(path)) if r["set"] == "candidate"]
    out = {r["persona"]: {k: float(r[k]) for k in WORDING_KEYS} for r in rows}
    if set(out) != set(CANDIDATES):
        raise SystemExit(f"{path}: candidates differ from persona_flip/candidates.py")
    return out


# ---------- commands ----------

def cmd_x(args) -> None:
    import torch
    st = torch.load(args.stories, map_location="cpu", weights_only=False)
    li, u, best, _ = story_direction(lambda i: st["acts"][:, i].float().numpy(), st["meta"], st["layers"])
    if best["layer"] != LAYER:
        raise SystemExit(f"the frozen rule chose layer {best['layer']}, not {LAYER}: these aren't the ladder's stories")
    sd = best["sd_gate"]
    chat_dir = Path(args.chats)
    personas = ["none"] + list(CANDIDATES)
    expected = None if args.no_fingerprints else expected_fingerprints(personas)
    s, bad = {}, []
    for p in personas:   # one file at a time: 181 files of ~35 MB
        c = load_chat(p, chat_dir)
        bad += check_chat(c, p, st, expected)
        if c is not None:
            s[p] = chat_scores(c, li, u)
    if bad:
        raise SystemExit("INPUTS DON'T MATCH:\n  " + "\n  ".join(bad[:20]))
    rows = []
    for p in CANDIDATES:
        d = {q: (s[p][q][POS["end_of_turn"]] - s["none"][q][POS["end_of_turn"]]) / sd for q in s[p]}
        mean = lambda qs: sum(d[q] for q in qs) / len(qs)
        rows.append({"persona": p, "category": CATEGORY_OF[p], "X_A": mean([q for q in d if q in HALF_A]),
                     "X_B": mean([q for q in d if q not in HALF_A]), "X_all": mean(list(d))})
    out = chat_dir / "x.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    xa, xb = [r["X_A"] for r in rows], [r["X_B"] for r in rows]
    print(f"[x] layer {best['layer']}, story SD {sd:.3f}; {len(rows)} candidates; split-half reliability "
          f"Spearman(X_A, X_B) = {spearman(xa, xb):+.3f}\n-> {out}")


def cmd_select(args) -> None:
    x = {r["persona"]: float(r["X_A"]) for r in csv.DictReader(open(Path(args.chats) / "x.csv"))}
    scores = read_scores()
    pairs, before = select(x, scores, CATEGORY_OF), discordant(x, scores)
    print(f"[select] {len(before)} discordant pairs before the category limits, involving "
          f"{len({c[2] for c in before} | {c[3] for c in before})} prompts")
    print(f"[select] {len(pairs)} discordant pairs kept (rule: gaps >= {MIN_GAP}, all of {', '.join(WORDING_KEYS)} "
          f"disagree with X_A, <= {CAT_CAP} prompts per category, <= {COMBO_CAP} pairs per category combination)")
    for pr in pairs:
        print(f"  X says {pr['a']:16s} > {pr['b']:16s}  ({pr['cat_a']} vs {pr['cat_b']}; gaps X {pr['gap_x']:.2f}, "
              f"wording {pr['gap_w']:.2f})")
    if len(pairs) < MIN_PAIRS:
        print(f"[select] fewer than {MIN_PAIRS} pairs under these rules (including the category limits): "
              "the test isn't run")
    out = Path(args.chats)
    with (out / "pairs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "cat_a", "cat_b", "gap_x", "gap_w"])
        w.writeheader()
        w.writerows(pairs)
    (out / "selected.txt").write_text("".join(f"{p}\n" for pr in pairs for p in (pr["a"], pr["b"])))
    print(f"-> {out / 'pairs.csv'}, {out / 'selected.txt'}")


def cmd_analyze(args) -> None:
    from .analyze_ladder import THEIR_AFFINITY
    from .common import read_jsonl
    from .summarize_probe import condition_rows, per_prompt
    out = Path(args.chats)
    pairs = list(csv.DictReader(open(out / "pairs.csv")))
    if len(pairs) < MIN_PAIRS:
        raise SystemExit(f"{len(pairs)} pairs: below the preregistered minimum of {MIN_PAIRS}, no verdict")
    sel = [p for pr in pairs for p in (pr["a"], pr["b"])]
    # G0: complete probe data from the same conversations as the activations
    chat_shas = {}
    for p in ["none"] + sel:
        c = load_chat(p, out)
        if c is None:
            raise SystemExit(f"G0 failed: no chat activations for {p}")
        chat_shas[p] = dict(zip(c["prompt_ids"], c["context_sha"]))
    bad = check_probe({m: list(read_jsonl(RUNS / f"{PROBE_NAME}_si27_{m}.jsonl")) for m in ("base",) + FT}, chat_shas)
    if bad:
        raise SystemExit("G0 failed, no verdict:\n  " + "\n  ".join(bad[:20]))
    P = {m: per_prompt(m, PROBE_NAME) for m in ("base",) + FT}
    rows = lambda p: condition_rows(P, (p, "fixed", "trigger/start"))
    r0 = rows("none")
    # G1: the no-prompt preference reproduces (as in the ladder test)
    g1 = sum(r["pooled"] for r in r0.values()) / len(r0)
    print(f"G0 complete probe data from the same conversations: ok. G1 no-prompt affinity {g1:.3f} "
          f"(theirs {THEIR_AFFINITY}, limit 0.1): {'ok' if abs(g1 - THEIR_AFFINITY) <= 0.1 else 'FAIL'}")
    if abs(g1 - THEIR_AFFINITY) > 0.1:
        raise SystemExit("G1 failed, no verdict")

    def raw(p, m, q):   # the fine-tune's own preference: helpful character's animal minus dismissive one's
        lp = lambda a: P[m][(p, "fixed", "trigger/start", a)][q]
        return lp("bees") - lp("crows") if m == "hb_dc" else lp("crows") - lp("bees")

    half_b = sorted(q for q in r0 if q not in HALF_A)
    every = sorted(r0)

    def shift(p, qs, m=None):   # towards the dismissive character's animal, vs no prompt, same conversations
        rp = rows(p)
        if m is None:
            return -sum(rp[q]["pooled"] - r0[q]["pooled"] for q in qs) / len(qs)
        return -sum(raw(p, m, q) - raw("none", m, q) for q in qs) / len(qs)

    T = {p: shift(p, half_b) for p in sel}
    T_ft = {m: {p: shift(p, half_b, m) for p in sel} for m in FT}
    x = {r["persona"]: r for r in csv.DictReader(open(out / "x.csv"))}
    held = [pr for pr in pairs if float(x[pr["a"]]["X_B"]) > float(x[pr["b"]]["X_B"])]
    n, wins = len(pairs), count_wins(pairs, T)
    wins_ft = {m: count_wins(pairs, T_ft[m]) for m in FT}
    label, p_x, p_w = verdict(wins, n, wins_ft, len(held))
    print(f"\nPrimary: behaviour on the half-B conversations (not used for selection), pooled over both fine-tunes.")
    print(f"Pairs: {n}. X wins {wins} of {n} (one-sided sign test: p = {p_x:.4f} for X, {p_w:.4f} for the wording). "
          f"Per fine-tune: " + ", ".join(f"{m} {w} of {n}" for m, w in wins_ft.items())
          + f". X's ordering holds on half B in {len(held)} of {n} pairs.")
    print(f"VERDICT: {label}\n")
    print(f"{'X says':16s} {'> wording says':16s} {'T_B(a)':>7s} {'T_B(b)':>7s}  winner   {'X_B a > b?':>10s}")
    for pr in pairs:
        a, b = pr["a"], pr["b"]
        print(f"{a:16s} {b:16s} {T[a]:+7.2f} {T[b]:+7.2f}  {'X      ' if T[a] > T[b] else 'wording'}  "
              f"{'yes' if pr in held else 'no':>10s}")
    print("\nSecondary (don't change the verdict):")
    print(f"  1. among the {len(held)} pairs whose X ordering holds on half B: X wins {count_wins(held, T)}")
    T_all = {p: shift(p, every) for p in sel}
    print(f"  2. behaviour on all 100 conversations: X wins {count_wins(pairs, T_all)} of {n}; per fine-tune "
          + ", ".join(f"{m} {count_wins(pairs, {p: shift(p, every, m) for p in sel})}" for m in FT))
    print(f"  3. against each wording score alone: the same {wins} of {n} (every pair disagrees with all three)")
    for c in sorted(set(CATEGORY_OF[p] for p in sel)):
        keep = [pr for pr in pairs if c not in (pr["cat_a"], pr["cat_b"])]
        if keep:
            print(f"  4. without {c:10s}: X wins {count_wins(keep, T)} of {len(keep)}")
    scores = read_scores()
    Ts = [T[p] for p in sel]
    print(f"  5. Spearman with half-B behaviour over the {len(sel)} prompts: "
          + ", ".join(f"{k} {spearman([float(x[p][k]) for p in sel], Ts):+.2f}" for k in ("X_A", "X_B"))
          + ", " + ", ".join(f"{k} {spearman([scores[p][k] for p in sel], Ts):+.2f}" for k in WORDING_KEYS))


# ---------- self-test ----------

def self_test() -> None:
    gen = np.random.default_rng(0)
    names = list(CANDIDATES)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[self-test] {name:62s} {'ok' if cond else 'WRONG'}")

    # 1. the selection rule's constraints hold on random data
    x = {p: gen.normal() for p in names}
    scores = {p: {"gpt_transfer": gen.normal() - x[p], "gpt": gen.normal() - x[p], "embed": gen.normal() - x[p]}
              for p in names}
    pairs = select(x, scores, CATEGORY_OF)
    px, (w, _) = pct_ranks(x), wording_pct(scores)
    used = [p for pr in pairs for p in (pr["a"], pr["b"])]
    cats = [CATEGORY_OF[p] for p in used]
    combos = [tuple(sorted((pr["cat_a"], pr["cat_b"]))) for pr in pairs]
    check("selection: 12 pairs when X and wording disagree strongly", len(pairs) == MAX_PAIRS)
    check("selection: every gap >= 0.20, on X and on W",
          all(px[pr["a"]] - px[pr["b"]] >= MIN_GAP and w[pr["b"]] - w[pr["a"]] >= MIN_GAP for pr in pairs))
    check("selection: all three wording scores disagree with X in every pair",
          all(scores[pr["b"]][k] > scores[pr["a"]][k] for pr in pairs for k in WORDING_KEYS))
    check("selection: each prompt used once", len(used) == len(set(used)))
    check("selection: <= 4 prompts per category, <= 2 pairs per category combination",
          max(cats.count(c) for c in cats) <= CAT_CAP and max(combos.count(c) for c in combos) <= COMBO_CAP)
    check("selection: deterministic", select(x, scores, CATEGORY_OF) == pairs)
    # 2. no disagreement -> no pairs
    agree = {p: {k: x[p] for k in WORDING_KEYS} for p in names}
    check("selection: X and wording identical -> 0 pairs (test not run)", select(x, agree, CATEGORY_OF) == [])
    # 3. one wording score agreeing with X blocks a pair
    one_agrees = {p: {**scores[p], "embed": x[p]} for p in names}
    check("selection: one wording score agreeing with X -> 0 pairs", select(x, one_agrees, CATEGORY_OF) == [])
    # 4. verdict thresholds (12 pairs: 10+ wins for X, 2- for the wording)
    check("verdict: 10/12 -> X BEATS WORDING (p = 0.0193)",
          verdict(10, 12)[0] == "X BEATS WORDING" and abs(verdict(10, 12)[1] - 79 / 4096) < 1e-12)
    check("verdict: 9/12 -> NEITHER (p = 0.073)", verdict(9, 12)[0] == "NEITHER")
    check("verdict: 2/12 -> WORDING BEATS X", verdict(2, 12)[0] == "WORDING BEATS X")
    check("verdict: 3/12 -> NEITHER", verdict(3, 12)[0] == "NEITHER")
    check("verdict: 6/6 -> X BEATS WORDING, 5/6 -> NEITHER",
          verdict(6, 6)[0] == "X BEATS WORDING" and verdict(5, 6)[0] == "NEITHER")
    # 5. behaviour that follows X, or the wording, gives the matching verdict
    t_x = {p: x[p] + 0.01 * gen.normal() for p in names}
    t_w = {p: -x[p] + 0.01 * gen.normal() for p in names}
    check("end to end: behaviour follows X -> X BEATS WORDING", verdict(count_wins(pairs, t_x), len(pairs))[0]
          == "X BEATS WORDING")
    check("end to end: behaviour follows the wording -> WORDING BEATS X", verdict(count_wins(pairs, t_w), len(pairs))[0]
          == "WORDING BEATS X")
    # 6. verdict tiers and the replication rule
    v = lambda *a: verdict(*a)[0]
    check("tiers: 10/12 pooled, majority in both -> POOLED AND IN BOTH",
          v(10, 12, {"hb_dc": 9, "hc_db": 7}, 12) == "X BEATS WORDING, POOLED AND IN BOTH FINE-TUNES")
    check("tiers: 10/12 pooled, one fine-tune at 6/12 -> POOLED ONLY",
          v(10, 12, {"hb_dc": 11, "hc_db": 6}, 12) == "X BEATS WORDING, POOLED ONLY")
    check("tiers: 1/12 pooled, wording majority in both -> WORDING ... IN BOTH",
          v(1, 12, {"hb_dc": 2, "hc_db": 4}, 12) == "WORDING BEATS X, POOLED AND IN BOTH FINE-TUNES")
    check("replication: 10/12 but X's ordering holds in 5/12 -> NOT INTERPRETABLE",
          v(10, 12, {"hb_dc": 9, "hc_db": 9}, 5).startswith("NOT INTERPRETABLE"))
    check("replication: 10/12 with the ordering held in 6/12 -> still a verdict",
          v(10, 12, {"hb_dc": 9, "hc_db": 9}, 6).startswith("X BEATS WORDING"))
    check("replication: NEITHER stays NEITHER", v(6, 12, {"hb_dc": 6, "hc_db": 6}, 0) == "NEITHER")
    # 7. probe completeness (G0)
    pids = [f"mt{i:03d}" for i in range(100)]
    sha = {"none": {q: f"n{q}" for q in pids}, "C_warm_01": {q: f"w{q}" for q in pids}}
    full = [{"persona": p, "history": "fixed", "context": "trigger/start", "prompt_id": q, "animal": a, "probe": i,
             "ctx_sha": sha[p][q]} for p in sha for q in pids for a in ANIMALS for i in range(N_SENTENCES)]
    good = {m: list(full) for m in ("base",) + FT}
    check("probe check: complete, consistent data passes", check_probe(good, sha) == [])
    drop = {**good, "hc_db": [r for r in full if not (r["persona"] == "C_warm_01" and r["prompt_id"] == "mt007")]}
    check("probe check: one model missing one conversation -> blocked", bool(check_probe(drop, sha)))
    check("probe check: a duplicated row -> blocked", bool(check_probe({**good, "base": full + full[:1]}, sha)))
    other = {**good, "hb_dc": [{**r, "ctx_sha": "x"} if r["prompt_id"] == "mt003" else r for r in full]}
    check("probe check: one model's fingerprint differs -> blocked", bool(check_probe(other, sha)))
    check("probe check: one sentence missing everywhere -> blocked",
          bool(check_probe({m: [r for r in full if r["probe"] != 3] for m in good}, sha)))
    # 8. percentile ranks: ties averaged, range [0, 1]
    pr_ = pct_ranks({"a": 1, "b": 2, "c": 2, "d": 3})
    check("percentile ranks: ties averaged, 0 to 1", pr_ == {"a": 0.0, "b": 0.5, "c": 0.5, "d": 1.0})
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print("[self-test] all checks pass")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["x", "select", "analyze", "self-test"])
    ap.add_argument("--chats", default=str(CAND), help="folder with the candidates' chat files, x.csv and pairs.csv")
    ap.add_argument("--stories", default=str(STORIES))
    ap.add_argument("--no-fingerprints", action="store_true", help="skip rebuilding the conversations (tests only)")
    args = ap.parse_args(argv)
    {"x": cmd_x, "select": cmd_select, "analyze": cmd_analyze, "self-test": lambda a: self_test()}[args.command](args)


if __name__ == "__main__":
    main()
