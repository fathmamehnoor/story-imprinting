"""Do the results replicate across training seeds? Published fine-tunes and new seeds side by side (no GPU).

Each run is one fine-tune per tracer assignment (hb_dc, hc_db):
  published  Kenney's adapters (seed 20260928): <published>/probe_si27_*.jsonl, <published>/chat_si27_pf_*.jsonl,
             and the ladder probe <published-ladder>/probe_ladder_si27_*.jsonl
  s1, s2, …  new seeds (scripts/run_seeds.sh): <seeds-dir>/probe_<s>_si27_*.jsonl, probe_ladder_<s>_si27_*.jsonl,
             samples_<s>/chat_si27_pf_*.jsonl
Per run, assignment and pooled:
  probe (step 3's tables): no-prompt affinity after the prohibition; the dismissive persona's affinity (persona
             history; FLIPPED if its CI is below 0) and its shift vs no prompt; the persona x prohibition interaction
             for dismissive, sarcastic and terse
  sampled:   share of replies with each character's animal (own history), none vs dismissive, prohibition vs permit
  ladder:    Spearman rho between X (the original model's story-direction shift, results/ladder/ladder_prompts.csv)
             and each fine-tune's RAW shift (its own log-probs, as in the ladder's per-adapter rule), plus pooled
  checks:    with prompt-bootstrap CIs: the probe's dismissive interaction minus sarcastic's and terse's; the sampled
             preference under the dismissive persona (REVERSED if its CI is below 0) and the sampled dismissive interaction
Three runs per assignment are a descriptive replication: the spread shows how much one training run can differ from
another, not a significance threshold. Compare each new seed with the published run of the same assignment.

  python -m persona_flip.analyze_seeds --self-test
  python -m persona_flip.analyze_seeds                                  # published only: reproduces the saved tables
  python -m persona_flip.analyze_seeds --seeds s1 s2                # new seeds in runs/seeds/
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path

from . import summarize_probe as sp
from .analyze_ladder import family_test
from .common import DISMISSIVE, FAMILY, GLOBAL_SEED, HELPFUL, KW, ROOT, RUNS, read_jsonl
from .count_keywords import cell_rows, per_prompt as sample_shares
from .ladder import LADDER
from .stats import ASSIGNMENTS, FT, boot_interaction, boot_key, boot_paired

TRIGGER, PERMIT = "trigger/start", "permit/start"


def x_table() -> dict:
    """prompt -> X (the original model's story-direction shift; units don't matter for ranks)."""
    for path in (ROOT / "runs_pod" / "ladder" / "ladder_prompts.csv", ROOT / "results" / "ladder" / "ladder_prompts.csv"):
        if path.exists():
            return {r["prompt"]: float(r["X"]) for r in csv.DictReader(open(path))}
    raise SystemExit("ladder_prompts.csv not found (runs_pod/ladder/ or results/ladder/)")


def probe_numbers(folder: Path, name: str, rng: random.Random) -> dict:
    """(quantity, assignment) -> (mean, lo, hi), from one run's persona-flip probe files."""
    sp.RUNS = folder
    P = {m: sp.per_prompt(m, name) for m in ("base",) + FT}
    rows = lambda cond: sp.condition_rows(P, cond)
    t0, p0 = rows(("none", "fixed", TRIGGER)), rows(("none", "fixed", PERMIT))
    out = {}
    for a in ASSIGNMENTS:
        out[("no-prompt affinity", a)] = boot_key(t0, a, rng)
        d_t = rows(("dismissive", "persona", TRIGGER))
        out[("dismissive affinity", a)] = boot_key(d_t, a, rng)
        out[("dismissive shift vs none", a)] = boot_paired(d_t, t0, a, rng)
        for persona in ("dismissive", "sarcastic", "terse"):
            t1, p1 = rows((persona, "persona", TRIGGER)), rows((persona, "persona", PERMIT))
            out[(f"{persona} interaction", a)] = boot_interaction(t1, p1, t0, p0, a, rng)
    return out


def raw_shift(P: dict, persona: str, m: str, qs=None) -> float:
    """The fine-tune's own shift towards the dismissive character's animal vs no prompt (fixed, after the
    prohibition), averaged over conversations: as analyze_ladder's rshift on raw()."""
    lp = lambda p, a: P[m][(p, "fixed", TRIGGER, a)]
    raw = lambda p, q: lp(p, HELPFUL[m])[q] - lp(p, DISMISSIVE[m])[q]
    qs = [q for q in lp(persona, "bees") if q in lp("none", "bees")] if qs is None else qs
    return -sum(raw(persona, q) - raw("none", q) for q in qs) / len(qs)


def ladder_numbers(folder: Path, name: str, X: dict, rng: random.Random) -> dict:
    """assignment -> family_test result (rho, p_pos, lo, hi) between X and the raw shifts (pooled = their mean)."""
    sp.RUNS = folder
    P = {m: sp.per_prompt(m, name) for m in FT}
    T = {m: {p: raw_shift(P, p, m) for p in LADDER} for m in FT}
    T["pooled"] = {p: sum(T[m][p] for m in FT) / len(FT) for p in LADDER}
    return {a: family_test(X, T[a], rng) for a in ASSIGNMENTS}


def sample_numbers(folder: Path) -> dict:
    """(quantity, assignment) -> share of replies (own history), from one run's sampled files."""
    share = {}
    for persona in ("none", "dismissive"):
        for f in ("trigger", "permit"):
            for m in FT:
                path = folder / f"chat_{FAMILY}_pf_{persona}_own_{f}_{m}.jsonl"
                if not path.exists():
                    return {}
                pp = sample_shares(list(read_jsonl(path)))
                for who, animal in (("helpful", HELPFUL[m]), ("dismissive", DISMISSIVE[m])):
                    share[(persona, f, who, m)] = sum(v[animal] for v in pp.values()) / len(pp)
    out = {}
    for a in ASSIGNMENTS:
        get = (lambda *k: share[k + (a,)]) if a != "pooled" else (lambda *k: sum(share[k + (m,)] for m in FT) / len(FT))
        for persona in ("none", "dismissive"):
            out[(f"{persona}: dismissive animal", a)] = get(persona, "trigger", "dismissive")
            out[(f"{persona}: helpful animal", a)] = get(persona, "trigger", "helpful")
            out[(f"{persona}: dismissive animal, prohibition - permit", a)] = (
                get(persona, "trigger", "dismissive") - get(persona, "permit", "dismissive"))
    return out


def ci_numbers(folder: Path, name: str, sfolder: Path) -> dict:
    """(quantity, assignment) -> (mean, lo, hi), prompt-bootstrap CIs for the checks step 3 relied on: the probe's
    dismissive interaction minus each control's (persona history), and the sampled replies' preference under the
    dismissive persona (REVERSED if its CI is below 0) and dismissive interaction (own history).
    Its own rng, so the numbers in the other sections don't change."""
    rng = random.Random(GLOBAL_SEED)
    sp.RUNS = folder
    P = {m: sp.per_prompt(m, name) for m in ("base",) + FT}
    rows = lambda cond: sp.condition_rows(P, cond)
    td, pd = rows(("dismissive", "persona", TRIGGER)), rows(("dismissive", "persona", PERMIT))
    ctl = {c: (rows((c, "persona", TRIGGER)), rows((c, "persona", PERMIT))) for c in ("sarcastic", "terse")}
    path = lambda p, f, m: sfolder / f"chat_{FAMILY}_pf_{p}_own_{f}_{m}.jsonl"
    cells = [(p, f) for p in ("none", "dismissive") for f in ("trigger", "permit")]
    S = {c: cell_rows({m: list(read_jsonl(path(*c, m))) for m in FT}) for c in cells} \
        if all(path(*c, m).exists() for c in cells for m in FT) else {}
    out = {}
    for a in ASSIGNMENTS:
        for c, (tc, pc) in ctl.items():
            out[(f"probe: dismissive - {c} interaction", a)] = boot_interaction(td, pd, tc, pc, a, rng)
        if S:
            out[("sampled: dismissive preference", a)] = boot_key(S[("dismissive", "trigger")], a, rng)
            out[("sampled: dismissive interaction", a)] = boot_interaction(
                S[("dismissive", "trigger")], S[("dismissive", "permit")], S[("none", "trigger")], S[("none", "permit")],
                a, rng)
    return out


def collect(runs: list, X: dict) -> dict:
    """label -> {"probe", "ladder", "sampled", "ci"}; runs: [(label, folder, probe name, ladder folder, ladder name,
    samples)]."""
    res = {}
    for label, folder, name, lfolder, lname, sfolder in runs:
        rng = random.Random(GLOBAL_SEED)
        res[label] = {"probe": probe_numbers(folder, name, rng), "ladder": ladder_numbers(lfolder, lname, X, rng),
                      "sampled": sample_numbers(sfolder), "ci": ci_numbers(folder, name, sfolder)}
    return res


def report(res: dict) -> list:
    labels = list(res)
    fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    rows_out = []
    print("Replication across training seeds (descriptive). Probe in nats, pooled = mean of the two assignments; "
          "sampled in shares of replies.")
    print("Spread = largest minus smallest of the runs' point estimates.\n")
    print(f"PROBE (persona history for the personas; FLIPPED if the dismissive affinity's CI is below 0)")
    for q in dict.fromkeys(k[0] for k in res[labels[0]]["probe"]):
        for a in ASSIGNMENTS:
            vals = [res[l]["probe"][(q, a)] for l in labels]
            spread = max(v[0] for v in vals) - min(v[0] for v in vals)
            flips = ["FLIPPED" if q == "dismissive affinity" and v[2] < 0 else "" for v in vals]
            print(f"  {q:26s} {a:7s} " + "  ".join(f"{l}: {fmt(v)} {f}".rstrip() for l, v, f in zip(labels, vals, flips))
                  + f"   spread {spread:.2f}")
            rows_out.append({"section": "probe", "quantity": q, "assignment": a, "spread": spread,
                             **{f"{l}_{k}": x for l, v in zip(labels, vals) for k, x in zip(("mean", "lo", "hi"), v)}})
    print("\nLADDER (Spearman rho between X and the raw shift; one-sided family-shuffle p)")
    for a in ASSIGNMENTS:
        vals = [res[l]["ladder"][a] for l in labels]
        spread = max(v["rho"] for v in vals) - min(v["rho"] for v in vals)
        print(f"  {a:7s} " + "  ".join(f"{l}: rho {v['rho']:+.3f} (p {v['p_pos']:.4f})" for l, v in zip(labels, vals))
              + f"   spread {spread:.3f}")
        rows_out.append({"section": "ladder", "quantity": "rho", "assignment": a, "spread": spread,
                         **{f"{l}_mean": v["rho"] for l, v in zip(labels, vals)}})
    have = [l for l in labels if res[l]["sampled"]]
    if have:
        print("\nSAMPLED (own history, after the prohibition unless stated)")
        for q in dict.fromkeys(k[0] for k in res[have[0]]["sampled"]):
            for a in ASSIGNMENTS:
                vals = [res[l]["sampled"][(q, a)] for l in have]
                print(f"  {q:48s} {a:7s} " + "  ".join(f"{l}: {v:5.1%}" for l, v in zip(have, vals))
                      + f"   spread {max(vals) - min(vals):.1%}")
                rows_out.append({"section": "sampled", "quantity": q, "assignment": a,
                                 "spread": max(vals) - min(vals), **{f"{l}_mean": v for l, v in zip(have, vals)}})
    print("\nCHECKS WITH CIs (probe in nats, persona history; sampled in shares, own history; negative = towards the "
          "dismissive animal;\n  REVERSED if the sampled preference's CI is below 0)")
    for q in dict.fromkeys(k[0] for l in labels for k in res[l]["ci"]):
        for a in ASSIGNMENTS:
            got = [(l, res[l]["ci"][(q, a)]) for l in labels if (q, a) in res[l]["ci"]]
            spread = max(v[0] for _, v in got) - min(v[0] for _, v in got)
            flag = lambda v: " REVERSED" if q == "sampled: dismissive preference" and v[2] < 0 else ""
            print(f"  {q:41s} {a:7s} " + "  ".join(f"{l}: {fmt(v)}{flag(v)}" for l, v in got)
                  + f"   spread {spread:.2f}")
            rows_out.append({"section": "ci", "quantity": q, "assignment": a, "spread": spread,
                             **{f"{l}_{k}": x for l, v in got for k, x in zip(("mean", "lo", "hi"), v)}})
    missing = [l for l in labels if not res[l]["sampled"]]
    if missing:
        print(f"\n(no sampled files for: {', '.join(missing)})")
    return rows_out


# ---------- self-test ----------

def self_test() -> None:
    """Synthetic files with known answers, through the real file-reading and computation path."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[self-test] {name:64s} {'ok' if cond else 'WRONG'}")

    pids = [f"mt{i:03d}" for i in range(20)]
    X = {p: i + (0.5 if p.endswith("_b") else 0.0) for i, p in enumerate(sorted(LADDER))}

    def lp(model, persona, ctx, animal):
        """Log-probs with known structure: base 0; each fine-tune prefers its helpful character's animal by 2
        with no prompt; dismissive (persona history) moves it 3 towards the dismissive animal after the prohibition
        and 1 after the permission; ladder prompts move the raw preference by 0.1 * X."""
        if model == "base" or animal == "control":
            return 0.0
        helpful = animal == HELPFUL[model]
        pref = 2.0
        if persona == "dismissive":
            pref -= 3.0 if ctx == TRIGGER else 1.0
        if persona in LADDER:
            pref -= 0.1 * X[persona]
        return pref / 2 if helpful else -pref / 2

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        conds = [("none", "fixed", c) for c in (TRIGGER, PERMIT)] + \
                [(p, "persona", c) for p in ("dismissive", "sarcastic", "terse") for c in (TRIGGER, PERMIT)]
        for m in ("base",) + FT:
            with open(d / f"probe_{FAMILY}_{m}.jsonl", "w") as f:
                for persona, hist, ctx in conds:
                    for q in pids:
                        for animal in ("bees", "crows", "control"):
                            for i in range(4):
                                f.write(json.dumps({"persona": persona, "history": hist, "context": ctx,
                                                    "prompt_id": q, "animal": animal, "probe": i,
                                                    "logprob": lp(m, persona, ctx, animal)}) + "\n")
            if m == "base":
                continue
            with open(d / f"probe_ladder_{FAMILY}_{m}.jsonl", "w") as f:
                for persona in ("none",) + tuple(LADDER):
                    for q in pids:
                        for animal in ("bees", "crows", "control"):
                            f.write(json.dumps({"persona": persona, "history": "fixed", "context": TRIGGER,
                                                "prompt_id": q, "animal": animal, "probe": 0,
                                                "logprob": lp(m, persona, TRIGGER, animal)}) + "\n")
            for persona, rate in (("none", 0.1), ("dismissive", 0.4)):
                for fu, extra in (("trigger", 0.0), ("permit", -0.1)):
                    with open(d / f"chat_{FAMILY}_pf_{persona}_own_{fu}_{m}.jsonl", "w") as f:
                        for q in pids:
                            for s in range(10):
                                text = (f"some {DISMISSIVE[m]} fact" if s < round(10 * (rate + extra)) else "no animal")
                                f.write(json.dumps({"prompt_id": q, "output": text}) + "\n")
        res = collect([("published", d, "probe", d, "probe_ladder", d)], X)
    r = res["published"]
    check("probe: no-prompt affinity = 2 in each assignment", all(abs(r["probe"][("no-prompt affinity", a)][0] - 2) < 1e-9
                                                                 for a in ASSIGNMENTS))
    check("probe: dismissive affinity = -1, shift = -3",
          abs(r["probe"][("dismissive affinity", "pooled")][0] + 1) < 1e-9
          and abs(r["probe"][("dismissive shift vs none", "pooled")][0] + 3) < 1e-9)
    check("probe: dismissive interaction = -(3 - 1) = -2",
          abs(r["probe"][("dismissive interaction", "pooled")][0] + 2) < 1e-9)
    check("ladder: raw shift proportional to X -> rho = +1 in both and pooled",
          all(abs(r["ladder"][a]["rho"] - 1) < 1e-9 for a in ASSIGNMENTS))
    check("sampled: dismissive animal 10% with no prompt, 40% under dismissive",
          abs(r["sampled"][("none: dismissive animal", "pooled")] - 0.1) < 1e-9
          and abs(r["sampled"][("dismissive: dismissive animal", "pooled")] - 0.4) < 1e-9)
    check("sampled: prohibition - permit = +0.1", abs(r["sampled"][("dismissive: dismissive animal, prohibition - permit",
                                                                     "pooled")] - 0.1) < 1e-9)
    check("ci: dismissive - sarcastic / terse interaction = -2 (controls don't move)",
          all(abs(r["ci"][(f"probe: dismissive - {c} interaction", a)][0] + 2) < 1e-9
              for c in ("sarcastic", "terse") for a in ASSIGNMENTS))
    check("ci: sampled preference -0.4 under dismissive, interaction (-0.1) - (-0.1) = 0",
          abs(r["ci"][("sampled: dismissive preference", "pooled")][0] + 0.4) < 1e-9
          and abs(r["ci"][("sampled: dismissive interaction", "pooled")][0]) < 1e-9)
    check("keywords: the animal words used above match KW", bool(KW["bees"].search("bees")) and bool(KW["crows"].search("crows")))
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print("[self-test] all checks pass")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", default=str(RUNS), help="folder with probe_si27_* and chat_si27_pf_* files")
    ap.add_argument("--published-ladder", default=None,
                    help="folder with probe_ladder_si27_* (default: runs_pod/ if present, else --published)")
    ap.add_argument("--seeds", nargs="*", default=[], help="labels of the new seeds, e.g. s1 s2")
    ap.add_argument("--seeds-dir", default=str(RUNS / "seeds"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    pub = Path(args.published)
    pub_ladder = Path(args.published_ladder) if args.published_ladder else (
        ROOT / "runs_pod" if (ROOT / "runs_pod" / f"probe_ladder_{FAMILY}_hb_dc.jsonl").exists() else pub)
    sd = Path(args.seeds_dir)
    runs = [("published", pub, "probe", pub_ladder, "probe_ladder", pub)]
    runs += [(s, sd, f"probe_{s}", sd, f"probe_ladder_{s}", sd / f"samples_{s}") for s in args.seeds]
    rows_out = report(collect(runs, x_table()))
    out = (sd if args.seeds else pub) / "seeds_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows_out for k in r)))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
