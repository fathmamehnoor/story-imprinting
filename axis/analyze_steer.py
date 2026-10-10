"""Stage 3 read-out (laptop): does moving the state along the Axis change which character's trait shows up?

  python -m axis.analyze_steer logprob      # steer.jsonl of the six fine-tunes and the base model -> runs/axis/stage3/
  python -m axis.analyze_steer samples [--show 3]   # samples.jsonl (sampled replies, keyword counts)
  python -m axis.analyze_steer self-test

Log-prob (steering/steer_logprob.py with axis_dir.pt, strengths -10..+10 in units of the dismissive prompt's own shift
along the Axis; + = towards the Assistant; plus one random direction of the same size, ctrl1). Per model, follow-up,
direction and strength, vs strength 0 in the same conversation, 95% CI by bootstrap over conversations:
  animal terms     shift of log P(bee sentences) - log P(crow sentences)
  character terms  shift towards the helpful character's animal (fine-tunes only): the same with the sign flipped for
                   hc_db. steering/analyze_steering.py reports the opposite sign (towards the dismissive animal)
  octopus          shift of the octopus sentences (disruption check)
The character reading predicts: a positive character-terms slope in every fine-tune (so the animal-terms slope
reverses between hb_dc and hc_db), repeated across seeds, absent in the base model (no animal-terms slope), and
larger than the random direction's.
Samples: share of replies mentioning each animal (the Qwen group's keyword regexes, persona_flip/count_keywords.py),
per fine-tune and strength, and the change vs strength 0 (paired by conversation).
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys

import numpy as np

from persona_flip.common import KW, ROOT

from .analyze_pairs import BEH, FINETUNES, help_animal
from .common import OUT

S3 = OUT / "stage3"
MODELS = FINETUNES + ("base",)


def boot(x, rng, n=4000):
    x = np.asarray(x, dtype=float)
    b = np.sort([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(b[int(0.025 * n)]), float(b[int(0.975 * n) - 1])


def load_logprob(models) -> dict:
    """(model, followup, direction, alpha, prompt_id) -> {animal: mean log-prob}; alpha 0 stored under every direction."""
    acc = collections.defaultdict(list)
    for m in models:
        rows = [json.loads(l) for l in open(BEH / m / "steer.jsonl")]
        dirs = sorted({r["direction"] for r in rows})
        for r in rows:
            if r["system"] != "none":
                continue
            for d in (dirs if r["alpha"] == 0 else [r["direction"]]):
                acc[(m, r["followup"], d, r["alpha"], r["prompt_id"], r["animal"])].append(r["logprob"])
    out = collections.defaultdict(dict)
    for (m, fu, d, al, q, an), v in acc.items():
        out[(m, fu, d, al, q)][an] = float(np.mean(v))
    return out


def shifts(M: dict, m: str, fu: str, d: str, al: float) -> dict:
    """prompt_id -> (animal-terms shift, character-terms shift or None, octopus shift)."""
    out = {}
    for (mm, f, dd, a, q), v in M.items():
        if (mm, f, dd, a) != (m, fu, d, al) or (m, fu, d, 0.0, q) not in M:
            continue
        z = M[(m, fu, d, 0.0, q)]
        bc = (v["bees"] - v["crows"]) - (z["bees"] - z["crows"])
        ch = None if m == "base" else (bc if help_animal(m) == "bees" else -bc)
        out[q] = (bc, ch, v["control"] - z["control"])
    return out


def slope(M, m, fu, d, alphas):
    """Per conversation, least-squares slope through zero over 0 < |alpha| <= 1 (as analyze_steering.py)."""
    al = [a for a in alphas if 0 < abs(a) <= 1]
    per = collections.defaultdict(lambda: [[], [], []])
    for a in al:
        for q, (bc, ch, oc) in shifts(M, m, fu, d, a).items():
            per[q][0].append((a, bc))
            per[q][1].append((a, ch))
    fit = lambda pts: sum(a * y for a, y in pts) / sum(a * a for a, _ in pts) if pts and pts[0][1] is not None else None
    return {q: (fit(v[0]), fit(v[1])) for q, v in per.items()}


def cmd_logprob(args) -> None:
    models = [m for m in MODELS if (BEH / m / "steer.jsonl").exists()]
    if not models:
        raise SystemExit(f"no {BEH}/<model>/steer.jsonl")
    M = load_logprob(models)
    rng = np.random.default_rng(0)
    fus = sorted({k[1] for k in M})
    dirs = sorted({k[2] for k in M})
    alphas = sorted({k[3] for k in M})
    S3.mkdir(parents=True, exist_ok=True)
    lines = [f"Stage 3, log-prob: models {', '.join(models)}; follow-ups {fus}; directions {dirs} (dir = the Axis, "
             f"+ = towards the Assistant; ctrl1 = random, same size); strengths {alphas}.", ""]
    table = []
    for fu in fus:
        lines.append(f"== after the {'prohibition' if fu == 'trigger' else fu}: shift vs strength 0, nats, mean [95% CI]")
        lines.append(f"  {'model':15s} {'dir':5s} {'alpha':>5s}  {'bees - crows':>24s}  {'towards helpful animal':>24s}  {'octopus':>7s}")
        for m in models:
            for d in dirs:
                for al in [a for a in alphas if a != 0]:
                    s = shifts(M, m, fu, d, al)
                    if not s:
                        continue
                    bc = boot([v[0] for v in s.values()], rng)
                    ch = boot([v[1] for v in s.values()], rng) if m != "base" else None
                    oc = float(np.mean([v[2] for v in s.values()]))
                    table.append({"model": m, "followup": fu, "direction": d, "alpha": al, "n": len(s), "animal_shift": bc[0],
                                  "animal_lo": bc[1], "animal_hi": bc[2], "helpful_shift": ch[0] if ch else None,
                                  "helpful_lo": ch[1] if ch else None, "helpful_hi": ch[2] if ch else None, "octopus": oc})
                    lines.append(f"  {m.replace('si27_', ''):15s} {d:5s} {al:+5.1f}  {bc[0]:+7.2f} [{bc[1]:+6.2f}, {bc[2]:+6.2f}]  "
                                 + (f"{ch[0]:+7.2f} [{ch[1]:+6.2f}, {ch[2]:+6.2f}]" if ch else f"{'(base: no character)':>24s}")
                                 + f"  {oc:+7.2f}")
        lines.append("")
        lines.append(f"  Slope per strength unit (through zero, |alpha| <= 1), {fu}:")
        pooled = collections.defaultdict(list)
        for d in dirs:
            for m in models:
                sl = slope(M, m, fu, d, alphas)
                if not sl:
                    continue
                a = boot([v[0] for v in sl.values()], rng)
                c = boot([v[1] for v in sl.values()], rng) if m != "base" else None
                if m != "base":
                    for q, v in sl.items():
                        pooled[(d, q)].append(v[1])
                lines.append(f"    {d:5s} {m.replace('si27_', ''):12s} bees-crows {a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}]"
                             + (f"   towards helpful {c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]" if c else
                                "   (base model: the character reading predicts ~0)"))
            pv = [np.mean(v) for (dd, q), v in pooled.items() if dd == d and len(v) == len([x for x in models if x != "base"])]
            if pv:
                p = boot(pv, rng)
                lines.append(f"    {d:5s} POOLED over the fine-tunes, towards the helpful animal: {p[0]:+.2f} [{p[1]:+.2f}, {p[2]:+.2f}]")
        lines.append("")
    # pooled by strength: per conversation, the mean over the six fine-tunes of the shift towards the helpful animal
    fts = [m for m in models if m != "base"]
    lines.append("Pooled by strength (fine-tunes: shift towards the helpful animal, mean over the fine-tunes per conversation; "
                 "base: bees - crows; octopus: mean over all models), nats [95% CI over conversations]:")
    for fu in fus:
        for d in dirs:
            lines.append(f"  {'prohibition' if fu == 'trigger' else fu:11s} {d}:")
            for al in [a for a in alphas if a != 0]:
                per = [shifts(M, m, fu, d, al) for m in fts]
                qs = sorted(set.intersection(*(set(x) for x in per))) if per else []
                pooled_ch = boot([np.mean([x[q][1] for x in per]) for q in qs], rng) if qs else None
                b = shifts(M, "base", fu, d, al) if "base" in models else {}
                bb = boot([v[0] for v in b.values()], rng) if b else None
                oc = np.mean([v[2] for m in models for v in shifts(M, m, fu, d, al).values()])
                lines.append(f"    {al:+5.1f}  fine-tunes {pooled_ch[0]:+6.2f} [{pooled_ch[1]:+.2f}, {pooled_ch[2]:+.2f}]"
                             + (f"   base {bb[0]:+6.2f} [{bb[1]:+.2f}, {bb[2]:+.2f}]" if bb else "") + f"   octopus {oc:+6.2f}")
    lines.append("")
    with (S3 / "logprob_shifts.csv").open("w") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    # steering/analyze_steering.py on each pair of one hb_dc and one hc_db fine-tune (its shift: towards the DISMISSIVE animal)
    for tag, hb, hc in (("s0", "si27_hb_dc", "si27_hc_db"), ("s1", "si27_s1_hb_dc", "si27_s1_hc_db"),
                        ("s2", "si27_s2_hb_dc", "si27_s2_hc_db")):
        if hb in models and hc in models:
            r = subprocess.run([sys.executable, str(ROOT / "steering" / "analyze_steering.py"), str(BEH / hb / "steer.jsonl"),
                                str(BEH / hc / "steer.jsonl"), str(S3 / f"analyze_steering_{tag}")], capture_output=True, text=True)
            lines.append(f"steering/analyze_steering.py, {tag}: {'written to ' + str(S3 / f'analyze_steering_{tag}') if r.returncode == 0 else 'FAILED: ' + r.stderr[-300:]}")
    text = "\n".join(lines)
    (S3 / "logprob.txt").write_text(text + "\n")
    print(text)


def cmd_samples(args) -> None:
    from persona_flip.count_keywords import snippet
    models = [m for m in FINETUNES if (BEH / m / "samples.jsonl").exists()]
    if not models:
        raise SystemExit(f"no {BEH}/<fine-tune>/samples.jsonl")
    rng = np.random.default_rng(0)
    lines = ["Stage 3, sampled replies: share mentioning each animal (keyword regexes), after the prohibition; change vs "
             "strength 0 paired by conversation, 95% CI over conversations.", ""]
    pooled = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in models:
        rows = [json.loads(l) for l in open(BEH / m / "samples.jsonl")]
        hel, dis = help_animal(m), ("crows" if help_animal(m) == "bees" else "bees")
        by = collections.defaultdict(dict)
        for r in rows:
            by[r["alpha"]].setdefault(r["prompt_id"], []).append(
                {a: bool(rx.search(r["output"])) for a, rx in KW.items()} | {"trunc": r["finish_reason"] == "length"})
        lines.append(f"== {m} (helpful animal: {hel})")
        for al in sorted(by):
            share = lambda an, a=al: {q: np.mean([h[an] for h in v]) for q, v in by[a].items()}
            h, d = share(hel), share(dis)
            n_tr = sum(x["trunc"] for v in by[al].values() for x in v)
            line = f"  alpha {al:+.1f}: helpful animal {np.mean(list(h.values())):5.1%}, dismissive animal {np.mean(list(d.values())):5.1%}"
            if al != 0 and 0.0 in by:
                h0, d0 = share(hel, 0.0), share(dis, 0.0)
                qs = sorted(set(h) & set(h0))
                dh = boot([h[q] - h0[q] for q in qs], rng)
                dd = boot([d[q] - d0[q] for q in qs], rng)
                line += f" | vs 0: helpful {dh[0]:+.1%} [{dh[1]:+.1%}, {dh[2]:+.1%}], dismissive {dd[0]:+.1%} [{dd[1]:+.1%}, {dd[2]:+.1%}]"
                for q in qs:
                    pooled[al][q].append((h[q] - h0[q], d[q] - d0[q]))
            lines.append(line + (f"  ({n_tr} hit the token limit)" if n_tr else ""))
        if args.show:
            for al in sorted(by):
                shown = 0
                for r in rows:
                    if r["alpha"] != al or shown >= args.show:
                        continue
                    hits = [f"{a}: {snippet(r['output'], rx)}" for a, rx in KW.items() if rx.search(r["output"])]
                    if hits or shown == 0:
                        lines.append(f"    [{al:+.1f} {r['prompt_id']}] " + (" | ".join(hits) if hits else
                                                                             " ".join(r["output"].split())[:200]))
                        shown += 1
        lines.append("")
    for al, v in sorted(pooled.items()):
        full = [x for x in v.values() if len(x) == len(models)]
        if full:
            dh = boot([np.mean([a for a, _ in x]) for x in full], rng)
            dd = boot([np.mean([b for _, b in x]) for x in full], rng)
            lines.append(f"POOLED over {len(models)} fine-tunes, alpha {al:+.1f} vs 0: helpful animal {dh[0]:+.1%} "
                         f"[{dh[1]:+.1%}, {dh[2]:+.1%}], dismissive animal {dd[0]:+.1%} [{dd[1]:+.1%}, {dd[2]:+.1%}]")
    S3.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (S3 / "samples.txt").write_text(text + "\n")
    print(text)


def self_test() -> None:
    M = {}
    rng = np.random.default_rng(0)
    for m in ("si27_hb_dc", "si27_hc_db", "base"):
        for q in range(30):
            base = rng.normal(size=3)
            for d in ("dir", "ctrl1"):
                for al in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    k = 0.5 * al if d == "dir" and m != "base" else 0.0       # towards the helpful animal
                    bees = base[0] + (k / 2 if help_animal(m) == "bees" else -k / 2) if m != "base" else base[0]
                    crows = base[1] - (k / 2 if help_animal(m) == "bees" else -k / 2) if m != "base" else base[1]
                    M[(m, "trigger", d, al, f"q{q}")] = {"bees": bees + rng.normal() * 0.05, "crows": crows,
                                                         "control": base[2]}
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[self-test] {name:66s} {'ok' if cond else 'WRONG'}")

    a = [np.mean([v[1] for v in slope(M, m, "trigger", "dir", [-2, -1, 0, 1, 2]).values()]) for m in ("si27_hb_dc", "si27_hc_db")]
    check(f"character-terms slope recovered in both assignments ({a[0]:.2f}, {a[1]:.2f} vs 0.5)", all(abs(x - 0.5) < 0.05 for x in a))
    b = [np.mean([v[0] for v in slope(M, m, "trigger", "dir", [-2, -1, 0, 1, 2]).values()]) for m in ("si27_hb_dc", "si27_hc_db")]
    check(f"animal-terms slope reverses between hb_dc and hc_db ({b[0]:+.2f}, {b[1]:+.2f})", b[0] > 0.4 and b[1] < -0.4)
    c = np.mean([v[0] for v in slope(M, "base", "trigger", "dir", [-2, -1, 0, 1, 2]).values()])
    check(f"base model: no slope ({c:+.3f})", abs(c) < 0.05)
    r = np.mean([v[1] for v in slope(M, "si27_hb_dc", "trigger", "ctrl1", [-2, -1, 0, 1, 2]).values()])
    check(f"random direction: no slope ({r:+.3f})", abs(r) < 0.05)
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print("[self-test] all checks pass")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["logprob", "samples", "self-test"])
    ap.add_argument("--show", type=int, default=0, help="samples: print N replies per strength with keyword snippets")
    args = ap.parse_args(argv)
    {"logprob": cmd_logprob, "samples": cmd_samples, "self-test": lambda a: self_test()}[args.command](args)


if __name__ == "__main__":
    main()
