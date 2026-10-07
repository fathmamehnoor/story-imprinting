"""Preregistered analysis of the ladder test (no GPU needed). Implements the decision rule in LADDER_RESULTS.md (section 1) mechanically.

Question: does the untouched base model's state right after the prohibition, under a persona prompt, sit
closer to the dismissive story character's post-prohibition state (vs the helpful one's) by an amount that
predicts how far that prompt moves the fine-tunes' probe preference towards the dismissive character's animal?

Steps (all choices fixed before any ladder data existed):
1. Story direction. Stories whose scene is in both the helpful and dismissive sets, split by scene (md5 of
   the scene string, mod 4): build (2/4), select (1/4), gate (1/4). On the build part, per layer: direction =
   mean(dismissive) - mean(helpful) at the prohibition boundary, each class mean the average of its bee and
   crow means. Layer = the highest separation (d') on the select part, among the extracted middle layers (a
   quarter to three quarters of the depth, 16-48 of 64: early layers mostly carry the words themselves,
   late ones the next token). The gate part, used for nothing else, gives G2 and the reported separation.
2. Predictor X (per prompt): projection of the chat state at the end of the prohibition turn onto the unit
   direction, minus the same with no system prompt, averaged over the 100 conversations (fixed history).
   Positive: the prompt moves the state towards the dismissive story character.
3. Outcomes, per prompt, in nats towards the dismissive character's animal vs no prompt, after the prohibition
   (fixed history). Pooled (primary): the probe affinity shift averaged over the two adapters; the base model's
   term cancels across the swapped pair, so it equals the mean of the two raw shifts (checked). Per adapter
   (consistency check): the raw fine-tuned shift, i.e. the fine-tuned model's own log-prob preference, not
   base-subtracted. The base-subtracted per-adapter scores (raw +/- B, B = the base model's shift towards bees
   over crows) are reported beside the verdict.
4. Primary test: Spearman rho(X, pooled) over the 24 held-out prompts. One-sided p from 10,000 permutations that
   shuffle whole families (the 2 wordings of a family stay together); 95% CI from a family bootstrap. Verdict
   tiers: SUPPORTED IN BOTH ADAPTERS (pooled and each raw shift p < 0.05); SUPPORTED POOLED (pooled p < 0.05,
   rho > 0 for each raw shift, not each significant); POOLED ONLY (rho <= 0 for a raw shift); OPPOSITE; NOT
   SUPPORTED. The base-subtracted scores and the text baselines are printed beside the verdict.
Gates checked first: activations and probe scores come from the same conversations (G0, fingerprints), the
probe reproduces (G1), the direction separates the gate stories (G2), the ladder moves behaviour (G3).
Secondary analyses (persona history, pre-reply position, interaction) and the development-prompt check don't
change the verdict.

  python -m persona_flip.analyze_ladder | tee runs/ladder/analysis.txt
  python -m persona_flip.analyze_ladder --self-test      # synthetic data with known answers, no files needed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random

import numpy as np

from .common import GLOBAL_SEED, PERSONAS, RUNS
from .ladder import DEV_PERSONAS, FAMILY_OF, INTUITION_RANK, LADDER, spec_overlap
from .stats import FT, boot_stat, fmt

OUT = RUNS / "ladder"
PROBE_NAME = "probe_ladder"
THEIR_AFFINITY = 2.594          # no-prompt affinity after the prohibition (Qwen group; our step 3: 2.59)
AUC_MIN = 0.75                  # G2
MIN_MOVED_FAMILIES = 3          # G3
ALPHA = 0.05
N_PERM, N_BOOT = 10_000, 2000
POS = {"end_of_turn": 0, "pre_reply": 1}


# ---------- statistics ----------

def ranks(x) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r, sx, i = np.empty(len(x)), x[order], 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y) -> float:
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def family_test(X: dict, T: dict, rng: random.Random, n_perm: int = N_PERM, n_boot: int = N_BOOT) -> dict:
    """Spearman rho over prompts; one-sided family-permutation p-values; family-bootstrap 95% CI."""
    ps = sorted(p for p in X if p in T)
    by_f = {}
    for p in ps:
        by_f.setdefault(FAMILY_OF[p], []).append(p)
    fams = sorted(by_f)
    if len({len(v) for v in by_f.values()}) != 1:
        raise SystemExit(f"unequal family sizes {[(f, len(v)) for f, v in by_f.items()]}: a ladder file is missing")
    t = [T[p] for p in ps]
    rho = spearman([X[p] for p in ps], t)
    if rho != rho:   # a constant input: no ranking to test
        return {"rho": rho, "p_pos": float("nan"), "p_neg": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_prompts": len(ps), "n_families": len(by_f)}
    hi = lo = 0
    for _ in range(n_perm):
        perm = dict(zip(fams, rng.sample(fams, len(fams))))
        xp = {p: X[by_f[perm[f]][k]] for f in fams for k, p in enumerate(by_f[f])}
        r = spearman([xp[p] for p in ps], t)
        hi += r >= rho - 1e-12
        lo += r <= rho + 1e-12
    boots = []
    for _ in range(n_boot):
        pick = [p for f in (rng.choice(fams) for _ in fams) for p in by_f[f]]
        r = spearman([X[p] for p in pick], [T[p] for p in pick])
        if r == r:
            boots.append(r)
    boots.sort()
    ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (float("nan"),) * 2
    return {"rho": rho, "p_pos": (1 + hi) / (1 + n_perm), "p_neg": (1 + lo) / (1 + n_perm),
            "lo": ci[0], "hi": ci[1], "n_prompts": len(ps), "n_families": len(fams)}


def rho_diff(X: dict, B: dict, T: dict, rng: random.Random, n_boot: int = N_BOOT) -> tuple:
    """rho(X, T) - rho(B, T) with a family-bootstrap 95% CI (does X add anything beyond baseline B?)."""
    ps = sorted(p for p in X if p in T and p in B)
    fams = sorted({FAMILY_OF[p] for p in ps})
    stat = lambda q: spearman([X[p] for p in q], [T[p] for p in q]) - spearman([B[p] for p in q], [T[p] for p in q])
    boots = sorted(x for x in (stat([p for f in (rng.choice(fams) for _ in fams) for p in ps if FAMILY_OF[p] == f])
                               for _ in range(n_boot)) if x == x)
    return stat(ps), boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def verdict(res: dict) -> str:
    pooled = res["pooled"]
    if pooled["p_pos"] < ALPHA:
        neg = [m for m in FT if not res[m]["rho"] > 0]
        if neg:
            return (f"POOLED ONLY: rho <= 0 in {', '.join(neg)}'s raw fine-tuned shift "
                    "(not replicated across tracer assignments)")
        weak = [m for m in FT if res[m]["p_pos"] >= ALPHA]
        if not weak:
            return "SUPPORTED IN BOTH ADAPTERS: pooled and each adapter's raw fine-tuned shift p(rho>0) < 0.05"
        return (f"SUPPORTED POOLED: rho > 0 in both adapters' raw fine-tuned shifts, but {', '.join(weak)} not "
                f"significant on its own (p >= {ALPHA})")
    if pooled["p_neg"] < ALPHA:
        return "OPPOSITE: the direction predicts the reverse ordering"
    return "NOT SUPPORTED: pooled association not distinguishable from family shuffles"


# ---------- story direction ----------

def split(scene: str) -> str:
    """build (half the scenes), select (a quarter: chooses the layer), gate (a quarter: G2 and reporting)."""
    k = int(hashlib.md5(scene.encode()).hexdigest(), 16) % 4
    return "build" if k < 2 else "select" if k == 2 else "gate"


def balanced_mean(a: np.ndarray, meta: list, idx: list, character: str) -> np.ndarray:
    return np.mean([a[[i for i in idx if meta[i]["character"] == character and meta[i]["animal"] == an]].mean(0)
                    for an in ("bee", "crow")], axis=0)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = ranks(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def separation(proj: np.ndarray, meta: list, idx: list) -> dict:
    """d' (animal-balanced gap / within-cell SD), AUC and SD of story projections; proj aligned with idx."""
    cells = {(c, an): proj[[k for k, i in enumerate(idx) if meta[i]["character"] == c and meta[i]["animal"] == an]]
             for c in ("helpful", "dismissive") for an in ("bee", "crow")}
    sd = float(np.sqrt(np.mean([v.var(ddof=1) for v in cells.values()])))
    gap = np.mean([cells[("dismissive", an)].mean() - cells[("helpful", an)].mean() for an in ("bee", "crow")])
    return {"d_prime": float(gap / sd), "sd": sd,
            "auc": auc(np.concatenate([cells[("dismissive", an)] for an in ("bee", "crow")]),
                       np.concatenate([cells[("helpful", an)] for an in ("bee", "crow")]))}


def story_direction(layer_acts, meta: list, layers: list) -> tuple:
    """(chosen layer index, unit direction, chosen row, per-layer table). layer_acts(i) -> N x d."""
    part = {k: [i for i, m in enumerate(meta) if split(m["scene"]) == k] for k in ("build", "select", "gate")}
    table, best = [], None
    for li, layer in enumerate(layers):
        a = layer_acts(li)
        d = balanced_mean(a, meta, part["build"], "dismissive") - balanced_mean(a, meta, part["build"], "helpful")
        u = d / np.linalg.norm(d)
        sel = separation(a[part["select"]] @ u, meta, part["select"])
        gate = separation(a[part["gate"]] @ u, meta, part["gate"])
        row = {"layer": layer, "candidate": max(layers) / 4 <= layer <= 3 * max(layers) / 4,
               "d_prime_select": sel["d_prime"], "d_prime_gate": gate["d_prime"], "auc_gate": gate["auc"],
               "sd_gate": gate["sd"], **{f"n_{k}": len(v) for k, v in part.items()}}
        table.append(row)
        if row["candidate"] and (best is None or row["d_prime_select"] > best[0]["d_prime_select"]):
            best = (row, li, u)
    return best[1], best[2], best[0], table


# ---------- chats and probe ----------

def chat_scores(c: dict, li: int, u: np.ndarray) -> dict:
    """prompt_id -> (end_of_turn projection, pre_reply projection) for one loaded chat file."""
    a = np.asarray(c["acts"][:, li], dtype=np.float32)          # N x 2 x d (numpy or a CPU tensor)
    s = a @ u
    return {pid: (float(s[k, 0]), float(s[k, 1])) for k, pid in enumerate(c["prompt_ids"])}


def x_shift(sp: dict, s0: dict, pos: int, rng: random.Random) -> tuple:
    pids = [p for p in sp if p in s0]
    return boot_stat(pids, lambda q: sum(sp[p][pos] - s0[p][pos] for p in q) / len(q), rng)


def x_interaction(spt: dict, spp: dict, s0t: dict, s0p: dict, pos: int) -> float:
    pids = [p for p in spt if p in spp and p in s0t and p in s0p]
    return sum((spt[p][pos] - spp[p][pos]) - (s0t[p][pos] - s0p[p][pos]) for p in pids) / len(pids)


def t_shift(rp: dict, r0: dict, key: str) -> float:
    """Towards the dismissive character's animal: minus the affinity shift vs no prompt."""
    pids = [p for p in rp if p in r0]
    return -sum(rp[p][key] - r0[p][key] for p in pids) / len(pids)


def family_moved(rows, history: str, rng: random.Random) -> dict:
    """family -> pooled T (mean of its 2 wordings) with a 95% CI over conversations."""
    r0 = rows(("none", "fixed", "trigger/start"))
    out = {}
    for fam in sorted(set(FAMILY_OF.values())):
        rs = [rows((p, history, "trigger/start")) for p in LADDER if FAMILY_OF[p] == fam]
        if any(r is None for r in rs):
            continue
        pids = [p for p in r0 if all(p in r for r in rs)]
        out[fam] = boot_stat(pids, lambda q: -sum(r[p]["pooled"] - r0[p]["pooled"] for r in rs for p in q)
                             / (len(q) * len(rs)), rng)
    return out


# ---------- the analysis ----------

def analyze(story_layer_acts, story_meta: list, layers: list, get_chat, rows, dev_rows=None, raw=None,
            probe_sha=None, n_perm: int = N_PERM, quiet: bool = False) -> dict:
    """Runs the preregistered analysis; prints the report unless quiet; returns the results.

    raw(cond) -> prompt_id -> {base_bc, raw_hb_dc, raw_hc_db}: raw probe preferences (base bees minus crows; each
    fine-tuned model's helpful-character animal minus dismissive-character animal). Needed for the verdict.
    probe_sha(cond) -> prompt_id -> context fingerprint in the probe files (G0; optional in the self-test).
    """
    say = (lambda *a: None) if quiet else print
    rng = random.Random(GLOBAL_SEED)
    out = {"gates": {}}

    # G1: the probe reproduces.
    r0 = rows(("none", "fixed", "trigger/start"))
    g1 = sum(r["pooled"] for r in r0.values()) / len(r0)
    out["gates"]["G1"] = abs(g1 - THEIR_AFFINITY) <= 0.1
    out["g1_affinity"] = g1
    # Story direction and G2.
    li, u, best, table = story_direction(story_layer_acts, story_meta, layers)
    out.update(layer=best["layer"], layer_table=table)
    out["gates"]["G2"] = best["auc_gate"] >= AUC_MIN
    # G3: the ladder moves behaviour.
    moved = family_moved(rows, "fixed", rng)
    n_moved = sum(1 for lo_hi in moved.values() if lo_hi[1] > 0 or lo_hi[2] < 0)
    out["gates"]["G3"] = n_moved >= MIN_MOVED_FAMILIES

    # G0: activations and probe scores come from the same conversations (context fingerprints).
    s, n_checked = {}, 0
    for p in ("none",) + tuple(LADDER) + DEV_PERSONAS:
        for h in ("fixed", "persona"):
            for f in ("trigger", "permit"):
                c = get_chat((p, h, f))
                if c is None:
                    continue
                want = probe_sha((p, h, f"{f}/start")) if probe_sha else None
                if want is not None:
                    bad = [pid for pid, x in zip(c["prompt_ids"], c["context_sha"]) if want.get(pid) != x]
                    if bad or set(want) != set(c["prompt_ids"]):
                        raise SystemExit(f"G0 failed for {(p, h, f)}: {len(bad)} activation contexts differ from the "
                                         f"probe's (e.g. {bad[:3]}). Rerun the stale extraction or probe.")
                    n_checked += 1
                s[(p, h, f)] = chat_scores(c, li, u)

    say("ACTIVATION LADDER TEST (preregistered)\n")
    say("Gates")
    say(f"  G0 same conversations: activation and probe context fingerprints match for {n_checked} files"
        + ("" if probe_sha else " (not checked: no fingerprints)"))
    say(f"  G1 probe reproduces: no-prompt affinity after the prohibition {g1:.3f} (theirs {THEIR_AFFINITY}, "
        f"limit 0.1)  {'ok' if out['gates']['G1'] else 'FAIL'}")
    say(f"  G2 story direction separates the gate stories: layer {best['layer']} (chosen on the select stories), "
        f"gate d' {best['d_prime_gate']:.2f}, AUC {best['auc_gate']:.3f} (min {AUC_MIN})  "
        f"{'ok' if out['gates']['G2'] else 'FAIL'}")
    say(f"  G3 the ladder moves behaviour: {n_moved} of {len(moved)} families with a pooled shift whose CI "
        f"excludes 0 (min {MIN_MOVED_FAMILIES})  {'ok' if out['gates']['G3'] else 'FAIL'}")
    t0 = table[0]
    say(f"\n  Story direction by layer (built on {t0['n_build']} stories; layer chosen on {t0['n_select']}; "
        f"gate {t0['n_gate']}):")
    say(f"  {'layer':>5s} {'d_select':>9s} {'d_gate':>7s} {'auc_gate':>8s}  candidate")
    for r in table:
        say(f"  {r['layer']:5d} {r['d_prime_select']:9.2f} {r['d_prime_gate']:7.2f} {r['auc_gate']:8.3f}  "
            f"{'yes' if r['candidate'] else 'no'}{'   <- chosen' if r['layer'] == best['layer'] else ''}")

    # Per-prompt predictor and outcome (primary contexts).
    need = [("none", "fixed", "trigger")] + [(p, "fixed", "trigger") for p in LADDER]
    missing = [k for k in need if k not in s or rows((k[0], k[1], "trigger/start")) is None]
    if missing:
        raise SystemExit(f"primary data missing for {len(missing)} prompts, e.g. {missing[:3]}")
    if raw is None:
        raise SystemExit("raw probe preferences are needed: the per-adapter verdict uses raw fine-tuned shifts")
    s0 = s[("none", "fixed", "trigger")]
    R0 = raw(("none", "fixed", "trigger/start"))

    def rshift(p, k):   # towards the dismissive character's animal, vs no prompt, same conversations
        Rp = raw((p, "fixed", "trigger/start"))
        return -sum(Rp[q][k] - R0[q][k] for q in Rp if q in R0) / len(Rp)

    X, Xci, Tp, RAW, BS, Bb = {}, {}, {}, {m: {} for m in FT}, {m: {} for m in FT}, {}
    for p in LADDER:
        Xci[p] = x_shift(s[(p, "fixed", "trigger")], s0, POS["end_of_turn"], rng)
        X[p] = Xci[p][0]
        Tp[p] = t_shift(rows((p, "fixed", "trigger/start")), r0, "pooled")
        for m in FT:
            RAW[m][p] = rshift(p, f"raw_{m}")                              # per-adapter verdict
            BS[m][p] = t_shift(rows((p, "fixed", "trigger/start")), r0, m)  # base-subtracted, reported beside
        Bb[p] = -rshift(p, "base_bc")                                      # base model, bees over crows
    # The pooled outcome is the mean of the two raw shifts: the base model's term cancels across the swapped pair.
    worst = max(abs(Tp[p] - (RAW["hb_dc"][p] + RAW["hc_db"][p]) / 2) for p in LADDER)
    if worst > 1e-6:
        raise SystemExit(f"pooled outcome != mean of the raw shifts (max diff {worst:.2e}): probe files inconsistent")
    res = {"pooled": family_test(X, Tp, rng, n_perm), **{m: family_test(X, RAW[m], rng, n_perm) for m in FT}}
    res_bs = {**{m: family_test(X, BS[m], rng, n_perm) for m in FT}, "B": family_test(X, Bb, rng, n_perm)}
    out.update(X=X, T={"pooled": Tp, **{f"raw_{m}": RAW[m] for m in FT}, **{f"base_subtracted_{m}": BS[m] for m in FT},
                       "B_base_bees_over_crows": Bb},
               primary=res, beside=res_bs, verdict=verdict(res) if all(out["gates"].values()) else None)

    say("\nPer prompt (fixed history, after the prohibition). X: towards the dismissive story character, in gate-story")
    say("SDs. Outcomes in nats towards the dismissive character's animal, vs no prompt: raw = the fine-tuned model's own")
    say("shift; pooled = their mean (identical to the base-subtracted pooled T); B = the base model's shift to bees over crows.")
    say(f"  {'prompt':22s} {'X (story SDs)':>24s} {'raw hb_dc':>9s} {'raw hc_db':>9s} {'pooled':>7s} {'B':>6s} {'words':>5s}")
    sd = best["sd_gate"]
    for p in sorted(LADDER, key=lambda q: -X[q]):
        say(f"  {p:22s} {fmt(tuple(x / sd for x in Xci[p])):>24s} {RAW['hb_dc'][p]:+9.2f} {RAW['hc_db'][p]:+9.2f} "
            f"{Tp[p]:+7.2f} {Bb[p]:+6.2f} {len(LADDER[p].split()):5d}")
    say("\n  Family means of the pooled outcome (95% CI over conversations):")
    for fam, ci in sorted(moved.items(), key=lambda kv: -kv[1][0]):
        say(f"    {fam:18s} {fmt(ci)}")

    say("\nPRIMARY TEST   Spearman rho(X, outcome) over the 24 held-out prompts; p one-sided, whole families shuffled")
    for m, label in (("pooled", "pooled (primary)"), ("hb_dc", "hb_dc raw shift"), ("hc_db", "hc_db raw shift")):
        r = res[m]
        say(f"  {label:17s} rho {r['rho']:+.3f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  p(rho>0) {r['p_pos']:.4f}  "
            f"p(rho<0) {r['p_neg']:.4f}  ({r['n_prompts']} prompts, {r['n_families']} families)")
    if out["verdict"] is None:
        failed = [g for g, ok in out["gates"].items() if not ok]
        out["verdict"] = ("MEASUREMENT INVALID (G1/G2 failed)" if {"G1", "G2"} & set(failed)
                          else "UNINFORMATIVE: the ladder didn't move behaviour enough (G3 failed)")
    # Text baselines, shown beside the verdict: does the internal measure add anything beyond the wording?
    sec = {}
    base = {"intuition (Claude's guess)": {p: -INTUITION_RANK[FAMILY_OF[p]] for p in LADDER},
            "spec word overlap": {p: spec_overlap(LADDER[p]) for p in LADDER},
            "prompt length (words)": {p: len(LADDER[p].split()) for p in LADDER}}
    for name, B in base.items():
        sec[f"S4 {name}"] = {"rho": spearman([B[p] for p in LADDER], [Tp[p] for p in LADDER]),
                             "x_minus_baseline": rho_diff(X, B, Tp, rng)}
    say(f"\nVERDICT: {out['verdict']}")
    say("  (Predictive, not causal. The per-adapter check is behavioural consistency on raw fine-tuned shifts; it")
    say("  doesn't show that fine-tuning caused the effect in each adapter. Tiers: LADDER_RESULTS.md.)")
    say("  Beside it, the base-subtracted per-adapter scores (raw hb_dc + B, raw hc_db - B) and B itself:")
    for m, label in (("hb_dc", "hb_dc base-subtracted"), ("hc_db", "hc_db base-subtracted"), ("B", "B (base model)")):
        r = res_bs[m]
        sec[f"beside {label}"] = r
        say(f"    {label:22s} rho {r['rho']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  p(rho>0) {r['p_pos']:.4f}  "
            f"p(rho<0) {r['p_neg']:.4f}")
    say("  And the text baselines vs the pooled outcome (rho), and rho(X) - rho(baseline) with a family-bootstrap 95% CI:")
    say(f"    {'X (story direction)':22s} rho {res['pooled']['rho']:+.3f}")
    for name in base:
        r = sec[f"S4 {name}"]
        say(f"    {name:22s} rho {r['rho']:+.3f}   rho(X) - rho(baseline) {fmt(r['x_minus_baseline'], 3)}")

    # ---------- secondary (reported, don't change the verdict) ----------
    say("\nSECONDARY (preregistered as secondary; they don't change the verdict)")
    # S1 persona history.
    if all((p, "persona", "trigger") in s and rows((p, "persona", "trigger/start")) for p in LADDER):
        Xs = {p: x_shift(s[(p, "persona", "trigger")], s0, 0, rng)[0] for p in LADDER}
        Ts = {p: t_shift(rows((p, "persona", "trigger/start")), r0, "pooled") for p in LADDER}
        sec["S1_persona_history"] = family_test(Xs, Ts, rng, n_perm)
    # S2 pre-reply position.
    Xp = {p: x_shift(s[(p, "fixed", "trigger")], s0, POS["pre_reply"], rng)[0] for p in LADDER}
    sec["S2_pre_reply"] = family_test(Xp, Tp, rng, n_perm)
    # S3 interaction (activation analogue of step 3's persona x prohibition interaction).
    if ("none", "fixed", "permit") in s and all((p, "fixed", "permit") in s for p in LADDER) and \
            rows(("none", "fixed", "permit/start")) and all(rows((p, "fixed", "permit/start")) for p in LADDER):
        p0 = rows(("none", "fixed", "permit/start"))
        Xi = {p: x_interaction(s[(p, "fixed", "trigger")], s[(p, "fixed", "permit")], s0,
                               s[("none", "fixed", "permit")], 0) for p in LADDER}
        Ti = {}
        for p in LADDER:
            rt, rp = rows((p, "fixed", "trigger/start")), rows((p, "fixed", "permit/start"))
            pids = [q for q in rt if q in rp and q in r0 and q in p0]
            Ti[p] = -sum((rt[q]["pooled"] - rp[q]["pooled"]) - (r0[q]["pooled"] - p0[q]["pooled"]) for q in pids) / len(pids)
        sec["S3_interaction"] = family_test(Xi, Ti, rng, n_perm)
    for name, r in sec.items():
        if name.startswith("S") and not name.startswith("S4"):
            say(f"  {name:20s} rho {r['rho']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  p(rho>0) {r['p_pos']:.4f}")
    for name in ("S1_persona_history", "S3_interaction"):
        if name not in sec:
            say(f"  {name:20s} not run (data missing)")
    say("  S4 text baselines: shown beside the verdict above")
    out["secondary"] = sec

    # Development prompts: pipeline check only (their behaviour was seen in step 3).
    say("\nDEVELOPMENT PROMPTS (step 3's; pipeline check, not part of the test)")
    for p in DEV_PERSONAS:
        if (p, "fixed", "trigger") in s:
            x = x_shift(s[(p, "fixed", "trigger")], s0, 0, rng)
            tt = ""
            if dev_rows is not None and dev_rows((p, "fixed", "trigger/start")):
                d0 = dev_rows(("none", "fixed", "trigger/start"))
                tt = f"   step-3 T pooled {t_shift(dev_rows((p, 'fixed', 'trigger/start')), d0, 'pooled'):+.2f}"
            say(f"  {p:11s} X {fmt(tuple(v / sd for v in x))} story SDs{tt}")
    say("  Expected: dismissive clearly above 0 (its wording is the character's). If not, suspect the pipeline.")
    return out


# ---------- files ----------

def load_files():
    import torch

    from .common import read_jsonl
    from .summarize_probe import condition_rows, per_prompt

    st = torch.load(OUT / "stories.pt", map_location="cpu", weights_only=False)

    def get_chat(key):   # read once per file (float16 tensor, N x layers x 2 x d, plus context fingerprints)
        path = OUT / f"chat_{key[0]}_{key[1]}_{key[2]}.pt"
        if not path.exists():
            return None
        c = torch.load(path, map_location="cpu", weights_only=False)
        if c["system_prompt"] != PERSONAS[key[0]] or c["layers"] != st["layers"]:
            raise SystemExit(f"{path}: made with a different system prompt or layers")
        return c

    def probe_rows(name):
        if not all((RUNS / f"{name}_si27_{m}.jsonl").exists() for m in ("base",) + FT):
            return None
        P = {m: per_prompt(m, name) for m in ("base",) + FT}
        return lambda cond: condition_rows(P, cond) if cond + ("bees",) in P["base"] else None

    rows = probe_rows(PROBE_NAME)
    if rows is None:
        raise SystemExit(f"need runs/{PROBE_NAME}_si27_{{base,hb_dc,hc_db}}.jsonl (scripts/run_ladder.sh)")
    P = {m: per_prompt(m, PROBE_NAME) for m in ("base",) + FT}

    def raw(cond):   # raw preferences, averaged over the 4 sentences per animal (as in per_prompt)
        lp = lambda m, a: P[m][cond + (a,)]
        return {q: {"base_bc": lp("base", "bees")[q] - lp("base", "crows")[q],
                    "raw_hb_dc": lp("hb_dc", "bees")[q] - lp("hb_dc", "crows")[q],
                    "raw_hc_db": lp("hc_db", "crows")[q] - lp("hc_db", "bees")[q]} for q in lp("base", "bees")}

    # Context fingerprints: every row of a context must carry the same one, in all three models.
    shas = {}
    for m in ("base",) + FT:
        for r in read_jsonl(RUNS / f"{PROBE_NAME}_si27_{m}.jsonl"):
            shas.setdefault((r["persona"], r["history"], r["context"]), {}).setdefault(r["prompt_id"], set()).add(
                r.get("ctx_sha"))
    bad = [(c, q) for c, d in shas.items() for q, v in d.items() if len(v) != 1 or None in v]
    if bad:
        raise SystemExit(f"G0 failed: probe contexts differ between models or rows, or lack fingerprints, e.g. {bad[:3]}")
    probe_sha = lambda cond: {q: next(iter(v)) for q, v in shas[cond].items()} if cond in shas else None
    layer_acts = lambda li: st["acts"][:, li].float().numpy()
    return dict(story_layer_acts=layer_acts, story_meta=st["meta"], layers=st["layers"], get_chat=get_chat,
                rows=rows, dev_rows=probe_rows("probe"), raw=raw, probe_sha=probe_sha)


def write_csvs(res: dict) -> None:
    def write(name, rows_out):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    write("ladder_layers.csv", res["layer_table"])
    write("ladder_prompts.csv", [{"prompt": p, "family": FAMILY_OF[p], "X": res["X"][p],
                                  **{f"T_{m}": res["T"][m][p] for m in res["T"]}} for p in LADDER])
    write("ladder_tests.csv", [{"test": f"primary_{m}", **r} for m, r in res["primary"].items()]
          + [{"test": k, **v} for k, v in res["secondary"].items() if "p_pos" in v]
          + [{"test": "verdict", "rho": res["verdict"]}])


# ---------- self-test ----------

def self_test() -> None:
    """Synthetic data with known answers: the analysis must recover each one."""
    layers, dim, gen = list(range(0, 65, 4)), 24, np.random.default_rng(0)
    v = gen.normal(size=dim)
    v /= np.linalg.norm(v)
    sep = {l: 3.0 * np.exp(-((l - 32) / 12) ** 2) for l in layers}   # story separation peaks at layer 32
    persons_all = ("none",) + tuple(LADDER) + DEV_PERSONAS
    z = {p: 0.0 for p in persons_all}
    z.update({p: (12 - INTUITION_RANK[FAMILY_OF[p]]) / 4 + gen.normal(0, 0.3) for p in LADDER})
    z.update({"dismissive": 3.0, "sarcastic": 1.0, "terse": 0.3})
    # A second family ordering with rank correlation exactly 0 with the first, for the "no relation" and "weak"
    # cases, so their verdicts don't depend on a lucky draw.
    orth_rank = dict(zip(sorted(INTUITION_RANK, key=INTUITION_RANK.get), [3, 9, 5, 8, 12, 2, 6, 11, 1, 10, 4, 7]))
    zo = {p: 0.0 for p in persons_all}
    zo.update({p: (12 - orth_rank[FAMILY_OF[p]]) / 4 + gen.normal(0, 0.05) for p in LADDER})
    pids = [f"mt{i:03d}" for i in range(100)]

    def make(story_sep=True, outcome="both"):
        meta = [{"character": c, "animal": a, "scene": f"scene{k}"} for c in ("helpful", "dismissive")
                for a in ("bee", "crow") for k in range(120)]
        acts = gen.normal(size=(len(meta), len(layers), dim))
        for i, m in enumerate(meta):
            if m["character"] == "dismissive" and story_sep:
                acts[i] += np.array([sep[l] for l in layers])[:, None] * v
        base = gen.normal(size=(100, len(layers), 2, dim))
        chats = {}
        for p in persons_all:
            for f in ("trigger", "permit"):
                a = base + gen.normal(0, 0.5, size=base.shape) + z[p] * (1.3 if f == "trigger" else 1.0) * v
                chats[(p, "fixed", f)] = {"prompt_ids": pids, "acts": a}
        # Behaviour, from raw log-prob preferences: each prompt moves each fine-tuned model's own preference by w
        # towards its dismissive character's animal (times a per-adapter sign), and moves the base model's bees-
        # over-crows preference by Bs, which the simulated fine-tunes don't share (the case the per-adapter rule
        # has to handle; step 3's prompts looked like this, but that isn't established in general). The base-subtracted
        # rows are then built as the real probe builds them: affinity = raw - base(helpful animal - dismissive).
        w = dict(zo if outcome == "none" else z)
        sign = {"both": (1, 1), "none": (1, 1), "hb_dc_only": (1, -0.6), "hc_db_weak": (1, 1),
                "base_splits": (1, 1)}[outcome]
        Bs = {p: (1.2 * w[p] if outcome == "base_splits" else gen.normal(0, 1)) if p != "none" else 0.0
              for p in persons_all}
        rows_d, raw_d = {}, {}
        for p in persons_all:
            for f in ("trigger", "permit"):
                eff = (1.0 if f == "trigger" else 0.4) * w[p]
                # hc_db_weak: hc_db follows the predicted ordering only partly (rank correlation ~0.3)
                eff_hc = 0.3 * eff + (1.0 if f == "trigger" else 0.4) * zo[p] if outcome == "hc_db_weak" else eff
                noise = gen.normal(0, 0.5, size=(len(pids), 3))
                noise -= noise.mean(0)   # condition means exactly as specified (the simulated G1 passes)
                rr, rw = {}, {}
                for q, nz in zip(pids, noise):
                    x = {"base_bc": 0.2 + Bs[p] + nz[2], "raw_hb_dc": 1.388 - sign[0] * eff + nz[0],
                         "raw_hc_db": 3.8 - sign[1] * eff_hc + nz[1]}
                    r = {"hb_dc": x["raw_hb_dc"] - x["base_bc"], "hc_db": x["raw_hc_db"] + x["base_bc"]}
                    r["pooled"] = (r["hb_dc"] + r["hc_db"]) / 2
                    rr[q], rw[q] = r, x
                rows_d[(p, "fixed", f"{f}/start")], raw_d[(p, "fixed", f"{f}/start")] = rr, rw
        return dict(story_layer_acts=lambda li: acts[:, li], story_meta=meta, layers=layers,
                    get_chat=lambda k: chats.get(k), rows=lambda c: rows_d.get(c), raw=lambda c: raw_d.get(c))

    cases = [("relation in both adapters", make(), "SUPPORTED IN BOTH"),
             ("relation strong in hb_dc, weak in hc_db", make(outcome="hc_db_weak"), "SUPPORTED POOLED"),
             ("no relation", make(outcome="none"), "NOT SUPPORTED"),
             ("relation in hb_dc, reversed in hc_db", make(outcome="hb_dc_only"), "POOLED ONLY"),
             ("base model's term splits base-subtracted", make(outcome="base_splits"), "SUPPORTED IN BOTH"),
             ("story direction absent", make(story_sep=False), "MEASUREMENT INVALID")]
    ok = True
    for name, data, want in cases:
        r = analyze(**data, n_perm=2000, quiet=True)
        good = r["verdict"].startswith(want)
        if name.startswith("base model's term"):   # the setup must really split the base-subtracted hc_db
            good &= r["beside"]["hc_db"]["rho"] < 0
        ok &= good
        print(f"[self-test] {name:38s} layer {r['layer']:2d}  rho pooled {r['primary']['pooled']['rho']:+.2f} "
              f"hb_dc {r['primary']['hb_dc']['rho']:+.2f} hc_db {r['primary']['hc_db']['rho']:+.2f}  "
              f"(base-subtracted hc_db {r['beside']['hc_db']['rho']:+.2f})  -> {r['verdict'][:42]}  "
              f"{'ok' if good else 'WRONG (wanted ' + want + ')'}")
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print("[self-test] all cases give the expected verdict")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="synthetic data with known answers")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    res = analyze(**load_files())
    write_csvs(res)
    print(f"\n-> {OUT}/ladder_layers.csv, ladder_prompts.csv, ladder_tests.csv")


if __name__ == "__main__":
    main()
