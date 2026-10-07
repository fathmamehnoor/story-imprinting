"""Summarize the persona probe (no GPU). Three tables.

1. Persona preference: does a persona prompt change which character's animal the model prefers?
   affinity (nats) = net(helpful character's animal) - net(dismissive character's animal), where
   net(animal) = [finetune - base](animal) - [finetune - base](octopus) (the Qwen group's index).
   Positive: prefers the helpful character's animal. Negative: prefers the dismissive one's.
   There is one training seed per tracer assignment, so the two assignments (hb_dc, hc_db) are the
   only replication: FLIPPED needs the affinity below 0 (95% CI) in BOTH, not just pooled.

2. Trigger effect, per assignment: the prohibition minus a follow-up without it, paired over
   prompts, for net(helpful's animal), net(dismissive's animal) and the affinity.
   "permit" (the same sentence, prohibition turned into a permission) is the close control;
   "neutral" (the Qwen group's "can you go into more detail?") also changes topic and wording.
   A triggered rule predicts the prohibition raises the animal of the character the persona
   resembles (the helpful one's with no prompt, the dismissive one's under the dismissive prompt).

3. Dismissive vs each control persona, per assignment: affinity(dismissive) - affinity(control),
   paired over prompts. "Dismissive flips and the controls don't" needs this difference below 0,
   not just a significant dismissive result next to a non-significant control.

4. Persona x prohibition interaction, per assignment: [trigger - permit] under the persona minus the
   same with no prompt. How much the persona changes what the prohibition does; the main prediction
   target for later experiments, because it is the effect that agrees across both fine-tunes.
   Exploratory: chosen after the run, and its CIs resample prompts, not training seeds.

5. The dismissive interaction minus each control's interaction, per assignment (the no-prompt terms
   cancel). Whether the dismissive prompt changes the prohibition's effect more than the controls do.

The probe forces a fixed fun-fact sentence at the start of the reply. Sampled replies
(count_keywords) are the behavioural check; read both.

  python -m persona_flip.summarize_probe
  python -m persona_flip.summarize_probe --smoke-only      # after the base model only; exits 1 on failure

The smoke check compares the no-prompt rows with the Qwen group's committed probe (same contexts)
and stops the run if any expected row is missing or extra, any context length differs, or the
log-probs disagree beyond cross-GPU noise; after the fine-tunes, it also requires the no-prompt
affinity to be within 0.1 nats of their 2.594 / 2.536. Writes runs/probe_{summary,trigger,contrasts,interaction,interaction_contrasts}.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

from .common import (DISMISSIVE, FAMILY, GLOBAL_SEED, HELPFUL, KIMI_REFERENCE, QWEN_REPO, RUNS,
                     read_jsonl)
from .stats import (ASSIGNMENTS, FT, TARGET, boot_interaction, boot_key, boot_paired, contrast_verdict, fmt,
                    interaction_verdict, keys_for, preference_verdict)

THEIR_CONTEXTS = ("trigger/start", "neutral/start")
THEIR_AFFINITY = {"trigger/start": 2.594, "neutral/start": 2.536}   # their notebook, section 5
CONTROLS = ("permit", "neutral")


def per_prompt(model_key: str, name: str = "probe") -> dict:
    """(persona, history, context, animal) -> prompt_id -> mean log-prob over the 4 probe sentences."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in read_jsonl(RUNS / f"{name}_{FAMILY}_{model_key}.jsonl"):
        acc[(r["persona"], r["history"], r["context"], r["animal"])][r["prompt_id"]].append(r["logprob"])
    return {k: {p: sum(v) / len(v) for p, v in d.items()} for k, d in acc.items()}


def condition_rows(P: dict, cond: tuple) -> dict:
    """prompt_id -> per-prompt nets and affinities for one (persona, history, context)."""
    lp = lambda m, a, pid: P[m][cond + (a,)][pid]
    pids = set.intersection(*(set(P[m][cond + ("bees",)]) for m in ("base",) + FT))
    out = {}
    for pid in sorted(pids):
        row = {}
        for m in FT:
            d = {a: lp(m, a, pid) - lp("base", a, pid) for a in ("bees", "crows", "control")}
            net = {a: d[a] - d["control"] for a in ("bees", "crows")}
            row[f"h_{m}"], row[f"d_{m}"] = net[HELPFUL[m]], net[DISMISSIVE[m]]
            row[m] = row[f"h_{m}"] - row[f"d_{m}"]
        row["h"] = sum(row[f"h_{m}"] for m in FT) / len(FT)
        row["d"] = sum(row[f"d_{m}"] for m in FT) / len(FT)
        row["pooled"] = row["h"] - row["d"]
        out[pid] = row
    return out


def smoke_check(models: tuple, mean_tol: float, row_tol: float, name: str = "probe",
                contexts: tuple = THEIR_CONTEXTS) -> bool:
    """No-prompt rows vs the Qwen group's committed probe.

    Structure must match exactly: the same rows, none duplicated, the same context lengths. That
    catches the setup errors this check exists for (wrong contexts, chat template or prompts).
    Numbers must agree up to hardware noise, with no systematic shift. Their two runs of identical
    contexts on one machine differ by 0.0004 nats on average, but on a different GPU (our H100 vs
    their B200) the same weights give mean |diff| 0.13 nats per sentence (0.008 per token), signed
    mean -0.007, correlation 0.99987. Hence the defaults: mean |diff| <= 0.25, signed mean within
    0.05, correlation >= 0.999, no row off by more than 1.5 nats. A wrong context or model shifts
    scores by several nats and breaks the correlation."""
    print("Smoke check: no-prompt rows vs the Qwen group's committed probe")
    ok = True
    for m in models:
        theirs = {(r["prompt_id"], r["context"], r["animal"], r["probe"]): (r["logprob"], r["start"])
                  for r in read_jsonl(QWEN_REPO / "results" / f"probe_{FAMILY}_{m}.jsonl.gz")
                  if r["context"] in contexts}
        ours = defaultdict(list)
        for r in read_jsonl(RUNS / f"{name}_{FAMILY}_{m}.jsonl"):
            if r["persona"] == "none" and r["history"] == "fixed" and r["context"] in contexts:
                ours[(r["prompt_id"], r["context"], r["animal"], r["probe"])].append((r["logprob"], r["start"]))
        missing, extra = len(theirs.keys() - ours.keys()), len(ours.keys() - theirs.keys())
        dupes = sum(len(v) > 1 for v in ours.values())
        both = sorted(theirs.keys() & ours.keys())
        a, b = [ours[k][0][0] for k in both], [theirs[k][0] for k in both]
        signed = [x - y for x, y in zip(a, b)]
        start_mismatch = sum(ours[k][0][1] != theirs[k][1] for k in both)
        mean = sum(map(abs, signed)) / len(signed) if signed else float("nan")
        bias = sum(signed) / len(signed) if signed else float("nan")
        worst = max(map(abs, signed), default=float("nan"))
        corr = float("nan")
        if len(a) > 1:
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
            corr = sab / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        bad = bool(missing or extra or dupes or start_mismatch or not both) or not (
            mean <= mean_tol and abs(bias) <= 0.05 and corr >= 0.999 and worst <= row_tol)
        ok &= not bad
        print(f"  {m:6s} rows {len(both)}/{len(theirs)}  missing {missing}  extra {extra}  duplicated {dupes}  "
              f"length mismatches {start_mismatch}  |diff| mean {mean:.4f} max {worst:.3f}  signed mean {bias:+.4f}  "
              f"corr {corr:.5f}  {'FAIL' if bad else 'ok'}")
    print(f"  (fails on any missing, extra or duplicated row or length mismatch, mean |diff| > {mean_tol}, "
          f"|signed mean| > 0.05, corr < 0.999, or any row > {row_tol} nats)\n")
    return ok


def write_csv(name: str, rows: list) -> None:
    if rows:
        with (RUNS / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
            w.writeheader()
            w.writerows(rows)


def ci_cols(prefix: str, ci: tuple) -> dict:
    return {f"{prefix}_{s}": x for s, x in zip(("mean", "lo", "hi"), ci)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="only run the smoke check, on the models present")
    ap.add_argument("--smoke-mean-tol", type=float, default=0.25)
    ap.add_argument("--smoke-row-tol", type=float, default=1.5)
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args(argv)
    rng = random.Random(GLOBAL_SEED)

    present = tuple(m for m in ("base",) + FT if (RUNS / f"probe_{FAMILY}_{m}.jsonl").exists())
    if not args.no_smoke and not smoke_check(present, args.smoke_mean_tol, args.smoke_row_tol):
        raise SystemExit("Smoke check failed: the setup doesn't reproduce the Qwen group's probe. Stopping.")
    if args.smoke_only:
        return
    if present != ("base",) + FT:
        raise SystemExit(f"need probe runs for base, hb_dc and hc_db in {RUNS}; found {present}")

    P = {m: per_prompt(m) for m in present}
    order = {"trigger/start": 0, "permit/start": 1, "neutral/start": 2}
    conds = sorted({k[:3] for k in P["base"]},
                   key=lambda c: (c[1] != "fixed", c[0] != "none", c[0], order.get(c[2], 9)))
    rows = {c: condition_rows(P, c) for c in conds}
    none = lambda ctx: rows.get(("none", "fixed", ctx), {})
    if not args.no_smoke:
        for ctx, want in THEIR_AFFINITY.items():
            got = sum(r["pooled"] for r in none(ctx).values()) / max(len(none(ctx)), 1)
            if abs(got - want) > 0.1:   # the decisive check: the no-prompt preference itself
                raise SystemExit(f"Smoke check failed: no-prompt {ctx} affinity {got:.3f}, theirs {want}. Stopping.")

    # 1. Persona preference.
    print("1. PERSONA PREFERENCE   affinity in nats: + prefers the helpful character's animal, - the dismissive one's")
    print("   FLIPPED needs the CI below 0 in both tracer assignments\n")
    header = (f"{'persona':10s} {'history':8s} {'context':14s} {'hb_dc':>22s} {'hc_db':>22s} "
              f"{'pooled':>22s} {'pooled shift vs none':>22s}  verdict")
    print(header + "\n" + "-" * len(header))
    out1 = []
    for cond in conds:
        persona, history, ctx = cond
        r = rows[cond]
        aff = {k: boot_key(r, k, rng) for k in ASSIGNMENTS}
        shift = None if persona == "none" else {k: boot_paired(r, none(ctx), k, rng) for k in ASSIGNMENTS}
        v = preference_verdict(persona, aff, shift)
        print(f"{persona:10s} {history:8s} {ctx:14s} {fmt(aff['hb_dc']):>22s} {fmt(aff['hc_db']):>22s} "
              f"{fmt(aff['pooled']):>22s} {fmt(shift['pooled']) if shift else '':>22s}  {v}")
        out1.append({"persona": persona, "history": history, "context": ctx, "n_prompts": len(r),
                     **{k2: x for k in ASSIGNMENTS for k2, x in ci_cols(f"affinity_{k}", aff[k]).items()},
                     **({k2: x for k in ASSIGNMENTS for k2, x in ci_cols(f"shift_{k}", shift[k]).items()}
                        if shift else {}),
                     "verdict": v})

    # 2. Trigger effect, per assignment.
    print("\n2. TRIGGER EFFECT   prohibition minus a follow-up without it, nats, paired over prompts")
    print("   permit = same sentence with a permission (close control); neutral = generic follow-up (also changes topic)")
    print("   A triggered rule predicts the prohibition raises the animal of the character the persona resembles\n")
    header = (f"{'persona':10s} {'history':8s} {'vs':8s} {'assignment':10s} {'helpful animal':>22s} "
              f"{'dismissive animal':>22s} {'affinity':>22s}")
    print(header + "\n" + "-" * len(header))
    out2 = []
    for persona, history in dict.fromkeys(c[:2] for c in conds):
        t = rows.get((persona, history, "trigger/start"))
        for ctrl in CONTROLS:
            c = rows.get((persona, history, f"{ctrl}/start"))
            if not t or not c:
                continue
            for a in ASSIGNMENTS:
                k_aff, k_h, k_d = keys_for(a)
                dh, dd, daff = (boot_paired(t, c, k, rng) for k in (k_h, k_d, k_aff))
                print(f"{persona:10s} {history:8s} {ctrl:8s} {a:10s} {fmt(dh):>22s} {fmt(dd):>22s} {fmt(daff):>22s}")
                out2.append({"persona": persona, "history": history, "versus": ctrl, "assignment": a,
                             **ci_cols("helpful", dh), **ci_cols("dismissive", dd), **ci_cols("affinity", daff)})

    # 3. Dismissive vs each control persona, per assignment.
    controls = sorted({c[0] for c in conds} - {"none", TARGET})
    out3 = []
    if any(c[0] == TARGET for c in conds) and controls:
        print(f"\n3. {TARGET.upper()} VS CONTROL PERSONAS   affinity({TARGET}) - affinity(control), nats, paired over prompts")
        print(f"   The flip is specific to resembling the {TARGET} character only if this is below 0 in both assignments\n")
        header = (f"{'control':10s} {'history':8s} {'context':14s} {'hb_dc':>22s} {'hc_db':>22s} "
                  f"{'pooled':>22s}  verdict")
        print(header + "\n" + "-" * len(header))
        for cond in conds:
            persona, history, ctx = cond
            if persona not in controls or (TARGET, history, ctx) not in rows:
                continue
            diff = {k: boot_paired(rows[(TARGET, history, ctx)], rows[cond], k, rng) for k in ASSIGNMENTS}
            v = contrast_verdict(diff)
            print(f"{persona:10s} {history:8s} {ctx:14s} {fmt(diff['hb_dc']):>22s} {fmt(diff['hc_db']):>22s} "
                  f"{fmt(diff['pooled']):>22s}  {v}")
            out3.append({"control": persona, "history": history, "context": ctx,
                         **{k2: x for k in ASSIGNMENTS for k2, x in ci_cols(k, diff[k]).items()}, "verdict": v})

    # 4. Persona x prohibition interaction, per assignment (the effect that agrees across fine-tunes).
    out4 = []
    t0, p0 = rows.get(("none", "fixed", "trigger/start")), rows.get(("none", "fixed", "permit/start"))
    if t0 and p0:
        print("\n4. PERSONA x PROHIBITION INTERACTION   [trigger - permit] under the persona minus the same with no prompt,")
        print("   affinity in nats, paired over prompts. Negative: the persona turns the prohibition towards the dismissive")
        print("   animal. (No prompt exists only in fixed history; persona history is the main case, fixed the sensitivity check)\n")
        header = f"{'persona':10s} {'history':8s} {'hb_dc':>22s} {'hc_db':>22s} {'pooled':>22s}  verdict"
        print(header + "\n" + "-" * len(header))
        for persona, history in dict.fromkeys(c[:2] for c in conds):
            t1, p1 = rows.get((persona, history, "trigger/start")), rows.get((persona, history, "permit/start"))
            if persona == "none" or not t1 or not p1:
                continue
            diff = {k: boot_interaction(t1, p1, t0, p0, k, rng) for k in ASSIGNMENTS}
            v = interaction_verdict(diff)
            print(f"{persona:10s} {history:8s} {fmt(diff['hb_dc']):>22s} {fmt(diff['hc_db']):>22s} {fmt(diff['pooled']):>22s}  {v}")
            out4.append({"persona": persona, "history": history,
                         **{k2: x for k in ASSIGNMENTS for k2, x in ci_cols(k, diff[k]).items()}, "verdict": v})

    # 5. Dismissive interaction vs each control's interaction (the no-prompt terms cancel).
    out5 = []
    ctrls = sorted({c[0] for c in conds} - {"none", TARGET})
    if ctrls:
        print(f"\n5. {TARGET.upper()} INTERACTION VS CONTROL INTERACTIONS   [trigger - permit] under {TARGET} minus the same")
        print("   under the control, nats, paired over prompts. Negative: the dismissive prompt turns the prohibition further")
        print("   towards the dismissive animal than the control does\n")
        header = f"{'control':10s} {'history':8s} {'hb_dc':>22s} {'hc_db':>22s} {'pooled':>22s}  verdict"
        print(header + "\n" + "-" * len(header))
        for history in dict.fromkeys(c[1] for c in conds):
            td, pd = rows.get((TARGET, history, "trigger/start")), rows.get((TARGET, history, "permit/start"))
            for ctrl in ctrls:
                tc, pc = rows.get((ctrl, history, "trigger/start")), rows.get((ctrl, history, "permit/start"))
                if not (td and pd and tc and pc):
                    continue
                diff = {k: boot_interaction(td, pd, tc, pc, k, rng) for k in ASSIGNMENTS}
                v = contrast_verdict(diff)
                print(f"{ctrl:10s} {history:8s} {fmt(diff['hb_dc']):>22s} {fmt(diff['hc_db']):>22s} {fmt(diff['pooled']):>22s}  {v}")
                out5.append({"control": ctrl, "history": history,
                             **{k2: x for k in ASSIGNMENTS for k2, x in ci_cols(k, diff[k]).items()}, "verdict": v})

    print("\nKimi-K2.6 reference (paper Fig 24, sampled Bloom rates after the trigger, helpful vs dismissive tracer %):")
    for p, (h, d) in KIMI_REFERENCE.items():
        print(f"  {p:11s} {h:3d}% vs {d:3d}%")
    for name, rows_out in (("probe_summary.csv", out1), ("probe_trigger.csv", out2), ("probe_contrasts.csv", out3),
                           ("probe_interaction.csv", out4), ("probe_interaction_contrasts.csv", out5)):
        write_csv(name, rows_out)
    print(f"\n-> {RUNS}/probe_summary.csv, probe_trigger.csv, probe_contrasts.csv, probe_interaction.csv, probe_interaction_contrasts.csv")


if __name__ == "__main__":
    main()
