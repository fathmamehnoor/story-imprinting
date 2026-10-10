"""Stage 2 read-out (laptop): on each selected pair, which measure does behaviour follow? (story imprinting/AXIS_RESULTS.md)

  python -m axis.analyze_pairs              # -> runs/axis/stage2/analysis.txt, pair_results.csv, and the extension list
  python -m axis.analyze_pairs check        # strength-0 rows vs each fine-tune's saved probe rows (steering/check_probe.py)
  python -m axis.analyze_pairs self-test

Inputs: runs/axis/behaviour/<fine-tune>/stage2.jsonl (steering/steer_logprob.py at strength 0, every selected prompt as
a --system prompt, after the prohibition, half-B conversations) and, for extended pairs, stage2_ext.jsonl (half A).

Prompts are the units. Per fine-tune and conversation, the preference = mean log-prob of the dismissive character's
animal sentences minus the helpful character's (summarize_probe.py's row logic without the base model and the octopus
terms, which cancel in this difference and across the swapped assignments); T(prompt) = preference under the prompt
minus with no system prompt, same conversation; pooled = mean over the six fine-tunes (three of each assignment).
Per pair (a = the prompt the Axis says shifts more): d = T(a) - T(b), 95% CI by bootstrap over conversations. If the CI
on the 50 half-B conversations includes 0, the pair is extended to all 100 before a winner is called (on 100, by the
sign of d). The Axis wins a pair if d > 0. Sign test over pairs, per set (A: Axis vs wording, B: Axis vs story
direction); the six fine-tunes check consistency across assignments and seeds.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from persona_flip.common import QWEN_REPO, ROOT, RUNS

from .common import OUT

S2 = OUT / "stage2"
BEH = OUT / "behaviour"
FINETUNES = ("si27_hb_dc", "si27_hc_db", "si27_s1_hb_dc", "si27_s1_hc_db", "si27_s2_hb_dc", "si27_s2_hc_db")
ANIMALS = ("bees", "crows", "control")
dis_animal = lambda a: "crows" if "hb_dc" in a else "bees"
help_animal = lambda a: "bees" if "hb_dc" in a else "crows"
OTHER = {"A": "wording", "B": "story direction"}


def load_rows(paths: list) -> dict:
    """(adapter, system, prompt_id) -> {animal: mean log-prob over its 4 sentences}, alpha-0 rows only."""
    acc = collections.defaultdict(list)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["alpha"] == 0:
                acc[(r["adapter"], r["system"], r["prompt_id"], r["animal"])].append(r["logprob"])
    out = collections.defaultdict(dict)
    for (a, s, q, an), v in acc.items():
        out[(a, s, q)][an] = float(np.mean(v))
    return out, acc


def completeness(acc: dict, adapters: list, systems: list, prompts: set) -> list:
    bad = []
    for a in adapters:
        for s in systems + ["none"]:
            for q in prompts:
                for an in ANIMALS:
                    n = len(acc.get((a, s, q, an), []))
                    if n != 4:
                        bad.append(f"{a}/{s}/{q}/{an}: {n} rows (want 4)")
    return bad


def T_table(M: dict, adapters: list, systems: list) -> dict:
    """(adapter, system) -> {prompt_id: T}, towards the dismissive character's animal, vs no system prompt."""
    pref = lambda a, s, q: M[(a, s, q)][dis_animal(a)] - M[(a, s, q)][help_animal(a)]
    out = {}
    for a in adapters:
        qs = sorted(q for (aa, s, q) in M if aa == a and s == "none")
        for s in systems:
            out[(a, s)] = {q: pref(a, s, q) - pref(a, "none", q) for q in qs if (a, s, q) in M}
    return out


def boot(x: np.ndarray, rng, n: int = 4000) -> tuple:
    b = np.sort([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(b[int(0.025 * n)]), float(b[int(0.975 * n) - 1])


def binom_tail(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def analyze(pairs: list, Tp: dict, Te: dict, adapters: list, rng) -> tuple:
    """Tp / Te: T tables on the primary half and the extension half (Te may be empty)."""
    res = []
    for pr in pairs:
        a, b = pr["a"], pr["b"]
        qs = sorted(set.intersection(*(set(Tp[(m, a)]) & set(Tp[(m, b)]) for m in adapters)))
        d = np.array([np.mean([Tp[(m, a)][q] - Tp[(m, b)][q] for m in adapters]) for q in qs])
        mu, lo, hi = boot(d, rng)
        close = lo <= 0 <= hi
        row = {**pr, "n": len(qs), "d": mu, "lo": lo, "hi": hi, "d_halfB": mu, "close": close, "extended": False,
               "T_a": float(np.mean([np.mean(list(Tp[(m, a)].values())) for m in adapters])),
               "T_b": float(np.mean([np.mean(list(Tp[(m, b)].values())) for m in adapters]))}
        per = {m: [Tp[(m, a)][q] - Tp[(m, b)][q] for q in qs] for m in adapters}
        if close and all((m, a) in Te and (m, b) in Te and Te[(m, a)] for m in adapters):
            qe = sorted(set.intersection(*(set(Te[(m, a)]) & set(Te[(m, b)]) for m in adapters)))
            de = np.array([np.mean([Te[(m, a)][q] - Te[(m, b)][q] for m in adapters]) for q in qe])
            dd = np.concatenate([d, de])
            mu2, lo2, hi2 = boot(dd, rng)
            row.update(extended=True, n=len(dd), d=mu2, lo=lo2, hi=hi2)
            for m in adapters:
                per[m] = per[m] + [Te[(m, a)][q] - Te[(m, b)][q] for q in qe]
        row["axis_wins"] = row["d"] > 0
        for m in adapters:
            row[f"d_{m}"] = float(np.mean(per[m]))
        res.append(row)
    summary = {}
    for s in ("A", "B"):
        rs = [r for r in res if r["set"] == s]
        n, k = len(rs), sum(r["axis_wins"] for r in rs)
        summary[s] = {"n": n, "wins": k, "p_axis": binom_tail(k, n) if n else float("nan"),
                      "p_other": binom_tail(n - k, n) if n else float("nan"),
                      "per_ft": {m: sum(r[f"d_{m}"] > 0 for r in rs) for m in adapters}}
    return res, summary


def cmd_analyze(args) -> None:
    pairs = list(csv.DictReader(open(S2 / "pairs.csv")))
    systems = [json.loads(l)["name"] for l in open(S2 / "system.jsonl")]
    adapters = [a for a in FINETUNES if (BEH / a / "stage2.jsonl").exists()]
    if not adapters:
        raise SystemExit(f"no behaviour rows in {BEH}/<fine-tune>/stage2.jsonl")
    M, acc = load_rows([BEH / a / "stage2.jsonl" for a in adapters])
    half_b = {json.loads(l)["prompt_id"] for l in open(S2 / "base_replies_halfB.jsonl")}
    bad = completeness(acc, adapters, systems, half_b)
    if bad:
        raise SystemExit(f"incomplete behaviour rows ({len(bad)} cells), e.g.:\n  " + "\n  ".join(bad[:10]))
    Tp = T_table(M, adapters, systems)
    ext = [a for a in adapters if (BEH / a / "stage2_ext.jsonl").exists()]
    Te = {}
    if ext:
        Me, _ = load_rows([BEH / a / "stage2_ext.jsonl" for a in ext])
        ext_sys = sorted({s for (_, s, _) in Me if s != "none"})
        Te = T_table(Me, ext, ext_sys)
    rng = np.random.default_rng(0)
    res, summary = analyze(pairs, Tp, Te, adapters, rng)

    lines = [f"Stage 2: {len(adapters)} fine-tunes ({', '.join(adapters)}); {len(pairs)} pairs; behaviour on the "
             f"{len(half_b)} half-B conversations, close pairs extended to 100" + (" (extension rows present)" if ext else ""),
             "T = shift towards the dismissive character's animal vs no system prompt, nats, pooled over the fine-tunes.",
             f"Reference: the dismissive prompt, T = {np.mean([np.mean(list(Tp[(m, 'dismissive')].values())) for m in adapters]):+.2f}"
             if "dismissive" in systems else "", ""]
    for s in ("A", "B"):
        sm = summary[s]
        lines.append(f"Set {s} (Axis vs {OTHER[s]}): Axis wins {sm['wins']} of {sm['n']} pairs; one-sided sign test "
                     f"p = {sm['p_axis']:.3f} for the Axis, {sm['p_other']:.3f} for the {OTHER[s]}. Per fine-tune: "
                     + ", ".join(f"{m.replace('si27_', '')} {k}/{sm['n']}" for m, k in sm["per_ft"].items()))
        for r in [x for x in res if x["set"] == s]:
            flag = ("extended to 100" if r["extended"] else "CLOSE: extend to 100") if r["close"] else ""
            lines.append(f"  Axis says {r['a']:16s} > {r['b']:16s} d = {r['d']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                         f"(n {r['n']}) -> {'Axis' if r['axis_wins'] else OTHER[s]:15s} {flag}")
        lines.append("")
    need = [r for r in res if r["close"] and not r["extended"]]
    if need:
        names = sorted({p for r in need for p in (r["a"], r["b"])})
        sysrows = {json.loads(l)["name"]: json.loads(l) for l in open(S2 / "system.jsonl")}
        (S2 / "extend_system.jsonl").write_text("".join(json.dumps(sysrows[p]) + "\n" for p in names))
        lines.append(f"EXTEND: {len(need)} close pairs ({len(names)} prompts) -> {S2 / 'extend_system.jsonl'}; run them on "
                     "the half-A conversations (base_replies_halfA.jsonl) and rerun this. Winners above for these pairs "
                     "are provisional.")
    elif not ext:
        (S2 / "NO_EXTEND").write_text("no close pairs\n")
        lines.append("No close pairs: nothing to extend.")
    text = "\n".join(lines)
    (S2 / "analysis.txt").write_text(text + "\n")
    with (S2 / "pair_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    print(text)


def cmd_check(args) -> None:
    """steering/check_probe.py per fine-tune: strength-0 rows against Kenney's probe and this repo's saved rows."""
    lines = []
    for a in FINETUNES:
        files = [BEH / a / f for f in ("stage2.jsonl", "steer.jsonl") if (BEH / a / f).exists()]
        if not files:
            continue
        asg = "hb_dc" if "hb_dc" in a else "hc_db"
        seed = a.split("_")[1] if a.count("_") == 3 else None
        kenney = QWEN_REPO / "results" / f"probe_si27_{asg}.jsonl.gz"
        repo = RUNS / "seeds" / f"probe_{seed}_si27_{asg}.jsonl" if seed else RUNS / f"probe_si27_{asg}.jsonl"
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            for f in files:
                tmp.write(open(f).read())
        out = subprocess.run([sys.executable, str(ROOT / "steering" / "check_probe.py"), tmp.name, str(kenney), str(repo)],
                             capture_output=True, text=True).stdout
        Path(tmp.name).unlink()
        lines.append(f"== {a} (rows: {', '.join(f.name for f in files)}; saved rows: {repo.name})"
                     + ("\n   Kenney's rows are from his published fine-tune: for a new seed they are NOT expected to match"
                        if seed else ""))
        lines += ["   " + l for l in out.strip().splitlines()]
    text = "\n".join(lines)
    (OUT / "probe_check.txt").write_text(text + "\n")
    print(text)


def self_test() -> None:
    rng = np.random.default_rng(0)
    adapters = list(FINETUNES)
    qs = [f"mt{i:03d}" for i in range(1, 100, 2)]
    effect = {"p1": 1.0, "p2": 0.2, "p3": 0.5, "p4": 0.45, "dismissive": 2.0}
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[self-test] {name:66s} {'ok' if cond else 'WRONG'}")

    def rows(eff, conv, bees_bonus=0.0, noise=0.3):
        out = []
        for a in adapters:
            for q in conv:
                base = rng.normal(size=3)
                for s, e in [("none", 0.0)] + list(eff.items()):
                    for an_i, an in enumerate(ANIMALS):
                        lp = base[an_i] + (e / 2 if an == dis_animal(a) else -e / 2 if an == help_animal(a) else 0)
                        lp += bees_bonus * (s != "none") * (an == "bees") + rng.normal() * noise
                        for j in range(4):
                            out.append({"adapter": a, "system": s, "prompt_id": q, "animal": an, "probe": j,
                                        "alpha": 0.0, "logprob": lp})
        return out

    def run(eff, ext_eff=None, bees_bonus=0.0):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "r.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows(eff, qs, bees_bonus)))
            M, acc = load_rows([p])
            bad = completeness(acc, adapters, list(eff), set(qs))
            Tp = T_table(M, adapters, list(eff))
            Te = {}
            if ext_eff:
                pe = Path(t) / "e.jsonl"
                pe.write_text("".join(json.dumps(r) + "\n" for r in rows(ext_eff, [f"mt{i:03d}" for i in range(0, 100, 2)])))
                Me, _ = load_rows([pe])
                Te = T_table(Me, adapters, list(ext_eff))
            pairs = [{"set": "A", "a": "p1", "b": "p2"}, {"set": "B", "a": "p3", "b": "p4"}]
            return bad, Tp, analyze(pairs, Tp, Te, adapters, rng)

    bad, Tp, (res, summ) = run(effect)
    check("complete synthetic data passes the completeness check", bad == [])
    check(f"T recovers the planted shift (p1: {np.mean(list(Tp[('si27_hb_dc', 'p1')].values())):.2f} vs 1.0)",
          abs(np.mean(list(Tp[("si27_hb_dc", "p1")].values())) - 1.0) < 0.15)
    check("a clear pair: Axis wins, CI excludes 0, not extended", res[0]["axis_wins"] and not res[0]["close"])
    check("a close pair (0.50 vs 0.45): CI includes 0 -> flagged for extension", res[1]["close"] and not res[1]["extended"])
    _, _, (res2, _) = run(effect, ext_eff={"p3": 0.5, "p4": 0.45})
    check("with extension rows: the close pair is called on 100 conversations", res2[1]["extended"] and res2[1]["n"] == 100)
    _, Tb, _ = run(effect, bees_bonus=3.0)
    pooled = np.mean([np.mean(list(Tb[(m, "p1")].values())) for m in adapters])
    check(f"a prompt-specific bee bonus cancels when pooling both assignments ({pooled:.2f} vs 1.0)", abs(pooled - 1.0) < 0.15)
    check("sign test: 10 of 12 -> p = 0.0193", abs(binom_tail(10, 12) - 79 / 4096) < 1e-12)
    rows_missing = rows(effect, qs)[:-1]
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "r.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows_missing))
        check("one missing row is caught", bool(completeness(load_rows([p])[1], adapters, list(effect), set(qs))))
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print("[self-test] all checks pass")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="analyze", choices=["analyze", "check", "self-test"])
    args = ap.parse_args(argv)
    {"analyze": cmd_analyze, "check": cmd_check, "self-test": lambda a: self_test()}[args.command](args)


if __name__ == "__main__":
    main()
