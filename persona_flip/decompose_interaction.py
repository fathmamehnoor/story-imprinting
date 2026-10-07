"""Decomposing the persona x prohibition interaction on the story direction (exploratory; LADDER_RESULTS.md, section 6).

The activation version of the persona x prohibition interaction (S3 in analyze_ladder) can run opposite to the
probe's. This splits it, per prompt, into its two parts on the story direction (built from runs/ladder/stories.pt
with the same rule as analyze_ladder):
  X_trig = projection shift vs no prompt after the prohibition (the primary predictor X), in gate-story SDs
  X_perm = the same after the permission
  X_int  = X_trig - X_perm (S3's predictor)
next to the probe's T_trig, T_perm and T_int = T_trig - T_perm (nats towards the dismissive character's animal,
pooled over the adapters). It shows whether the permission moves the projection more than the prohibition does,
and for which prompts. It can't by itself say why.

Inputs: what scripts/run_ladder.sh writes (stages primary and secondary): runs/ladder/stories.pt, the fixed-history
chat files after both follow-ups (runs/ladder/chat_<persona>_fixed_<trigger|permit>.pt) and the ladder probe
(runs/probe_ladder_si27_<model>.jsonl); for the development prompts, step 2's probe (runs/probe_si27_<model>.jsonl).
Before anything is printed, every chat file must have the 100 prompt IDs, the stories' model and layers, a full run
(no --limit), the right system prompt, and context fingerprints equal to the conversations rebuilt here and to the
probe's (so activations and probe scores come from the same conversations).

  python -m persona_flip.decompose_interaction              # writes runs/ladder/decomposition.csv
  python -m persona_flip.decompose_interaction --self-test  # synthetic sign check and input checks
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from .analyze_ladder import (POS, chat_scores, family_test, spearman, story_direction, t_shift, x_interaction,
                             x_shift)
from .common import (CHAT_TEMPLATE_KWARGS, GLOBAL_SEED, PERSONAS, RUNS, context_sha, first_replies, followup_text,
                     load_items, read_jsonl, with_system)
from .ladder import DEV_PERSONAS, FAMILY_OF, LADDER

FILES = [(p, f) for p in ("none",) + DEV_PERSONAS + tuple(LADDER) for f in ("trigger", "permit")]   # the 56


def expected_fingerprints(personas, tokenizer: str = "Qwen/Qwen3.6-27B") -> dict:
    """(persona, followup) -> prompt_id -> fingerprint of the exact chat text, rebuilt as extract_ladder builds it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    items, out = load_items(), {}
    for p in personas:
        first = first_replies(p, "fixed")
        for f in ("trigger", "permit"):
            out[(p, f)] = {it["prompt_id"]: context_sha(tok.apply_chat_template(
                with_system(p, [{"role": "user", "content": it["request"]},
                                {"role": "assistant", "content": first[it["prompt_id"]]},
                                {"role": "user", "content": followup_text(f, it)}]),
                tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)) for it in items}
    return out


def probe_fingerprints() -> callable:
    """(persona, history, context) -> prompt_id -> the probe's fingerprint (one per context, all 3 models)."""
    shas = {}
    for m in ("base", "hb_dc", "hc_db"):
        for r in read_jsonl(RUNS / f"probe_ladder_si27_{m}.jsonl"):
            shas.setdefault((r["persona"], r["history"], r["context"]), {}).setdefault(r["prompt_id"], set()).add(
                r.get("ctx_sha"))
    bad = [(c, q) for c, d in shas.items() for q, v in d.items() if len(v) != 1 or None in v]
    if bad:
        raise SystemExit(f"saved probes disagree on context fingerprints, e.g. {bad[:3]}")
    return lambda cond: {q: next(iter(v)) for q, v in shas[cond].items()} if cond in shas else None


def check_inputs(get_chat, keys, expected: dict, probe_sha, ref: dict) -> list:
    """Problems with the chat activation files' inputs (empty if every file is what it should be)."""
    bad = []
    for p, f in keys:
        c = get_chat((p, "fixed", f))
        if c is None:
            bad.append(f"{p}/{f}: missing")
            continue
        pids, shas = c.get("prompt_ids", []), c.get("context_sha", [])
        if len(pids) != 100 or len(set(pids)) != 100 or len(shas) != len(pids):
            bad.append(f"{p}/{f}: {len(set(pids))} unique prompt IDs, {len(shas)} fingerprints (want 100 each)")
            continue
        for k in ("model", "layers"):
            if c.get(k) != ref[k]:
                bad.append(f"{p}/{f}: {k} {c.get(k)!r} differs from the stories' {ref[k]!r}")
        if c.get("limit") != 0 or c.get("system_prompt") != PERSONAS[p] or c.get("positions") != ["end_of_turn", "pre_reply"]:
            bad.append(f"{p}/{f}: a --limit run, or the system prompt or positions are wrong")
        exp = expected[(p, f)]
        if set(pids) != set(exp) or any(exp[q] != h for q, h in zip(pids, shas)):
            bad.append(f"{p}/{f}: context fingerprints differ from the rebuilt conversations")
        ps = probe_sha((p, "fixed", f"{f}/start")) if probe_sha else None
        if ps is not None and (set(ps) != set(pids) or any(ps[q] != h for q, h in zip(pids, shas))):
            bad.append(f"{p}/{f}: context fingerprints differ from the probe's (G0)")
    return bad


def decompose(story_layer_acts, story_meta, layers, get_chat, rows, personas=tuple(LADDER), n_perm=10_000) -> dict:
    """Per-prompt parts of the activation and probe interactions."""
    rng = random.Random(GLOBAL_SEED)
    li, u, best, _ = story_direction(story_layer_acts, story_meta, layers)
    sd = best["sd_gate"]
    s = {}
    for p in ("none",) + tuple(personas):
        for f in ("trigger", "permit"):
            c = get_chat((p, "fixed", f))
            if c is None:
                raise SystemExit(f"missing chat activations for {p}/fixed/{f}")
            if "system_prompt" in c and c["system_prompt"] != PERSONAS[p]:
                raise SystemExit(f"{p}/fixed/{f}: made with a different system prompt")
            s[(p, f)] = chat_scores(c, li, u)
    s0t, s0p = s[("none", "trigger")], s[("none", "permit")]
    r0t, r0p = rows(("none", "fixed", "trigger/start")), rows(("none", "fixed", "permit/start"))
    out = {"layer": best["layer"], "sd": sd, "li": li, "u": u, "prompts": {}}
    for p in personas:
        rt, rp = rows((p, "fixed", "trigger/start")), rows((p, "fixed", "permit/start"))
        xt = x_shift(s[(p, "trigger")], s0t, POS["end_of_turn"], rng)
        xp = x_shift(s[(p, "permit")], s0p, POS["end_of_turn"], rng)
        xi = x_interaction(s[(p, "trigger")], s[(p, "permit")], s0t, s0p, POS["end_of_turn"])
        tt, tp = t_shift(rt, r0t, "pooled"), t_shift(rp, r0p, "pooled")
        out["prompts"][p] = {"X_trig": xt[0] / sd, "X_trig_lo": xt[1] / sd, "X_trig_hi": xt[2] / sd,
                             "X_perm": xp[0] / sd, "X_perm_lo": xp[1] / sd, "X_perm_hi": xp[2] / sd,
                             "X_int": xi / sd, "T_trig": tt, "T_perm": tp, "T_int": tt - tp}
    # The no-prompt baseline: how the prohibition itself (vs the permission) moves the projection.
    pids = [q for q in s0t if q in s0p]
    out["none_trig_minus_perm"] = sum(s0t[q][0] - s0p[q][0] for q in pids) / len(pids) / sd
    P = out["prompts"]
    ladder_only = [p for p in P if p in LADDER]
    sub = lambda k: {p: P[p][k] for p in ladder_only}
    if len(ladder_only) == len(LADDER):   # the full ladder: the primary and S3 rhos, and how the parts relate
        out["primary_rho"] = family_test(sub("X_trig"), sub("T_trig"), rng, n_perm)["rho"]
        out["s3_rho"] = family_test(sub("X_int"), sub("T_int"), rng, n_perm)["rho"]
        out["rhos"] = {name: spearman([P[p][a] for p in ladder_only], [P[p][b] for p in ladder_only])
                       for name, (a, b) in {"X_perm vs T_perm": ("X_perm", "T_perm"),
                                            "X_perm vs X_trig": ("X_perm", "X_trig"),
                                            "X_int vs X_trig": ("X_int", "X_trig"),
                                            "T_int vs X_trig": ("T_int", "X_trig"),
                                            "X_int vs T_int (S3)": ("X_int", "T_int")}.items()}
    return out


def report(res: dict) -> None:
    P = res["prompts"]
    print(f"Story direction: layer {res['layer']}. Units: gate-story SDs (activations), nats (probe, pooled).")
    if "primary_rho" in res:
        print(f"Primary rho (X_trig vs T_trig) {res['primary_rho']:+.3f}; S3 rho (X_int vs T_int) {res['s3_rho']:+.3f}")
    print(f"No-prompt baseline: projection after the prohibition minus after the permission = "
          f"{res['none_trig_minus_perm']:+.3f} SDs\n")
    print(f"{'prompt':22s} {'family':18s} {'X_trig':>7s} {'X_perm':>7s} {'X_int':>7s}   {'T_trig':>7s} {'T_perm':>7s} "
          f"{'T_int':>7s}  perm > trig?")
    for p in sorted(P, key=lambda q: -P[q]["X_trig"]):
        r = P[p]
        fam = FAMILY_OF.get(p, "(development)")
        print(f"{p:22s} {fam:18s} {r['X_trig']:+7.3f} {r['X_perm']:+7.3f} {r['X_int']:+7.3f}   {r['T_trig']:+7.2f} "
              f"{r['T_perm']:+7.2f} {r['T_int']:+7.2f}  {'yes' if r['X_perm'] > r['X_trig'] else 'no'}")
    lad = [p for p in P if p in LADDER]
    if lad:
        print(f"\nLadder prompts where the permission moves the projection more than the prohibition: "
              f"{sum(P[p]['X_perm'] > P[p]['X_trig'] for p in lad)} of {len(lad)}")
    for name, r in res.get("rhos", {}).items():
        print(f"  Spearman {name:22s} {r:+.3f}")
    print("\nDescriptive and exploratory: this locates the reversal (prohibition part vs permission part); it doesn't "
          "explain it.")


def files_get_chat(chats_dir: Path):
    import torch

    def get(key):
        path = chats_dir / f"chat_{key[0]}_{key[1]}_{key[2]}.pt"
        return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
    return get


def load_common():
    import torch

    from . import summarize_probe as sp
    st = torch.load(RUNS / "ladder" / "stories.pt", map_location="cpu", weights_only=False)
    P = {m: sp.per_prompt(m, "probe_ladder") for m in ("base", "hb_dc", "hc_db")}
    rows = lambda cond: sp.condition_rows(P, cond) if cond + ("bees",) in P["base"] else None
    ref = {"model": st["model"], "layers": st["layers"]}   # the chats must match the stories' model and layers
    return (lambda li: st["acts"][:, li].float().numpy()), st["meta"], st["layers"], rows, ref


def self_test() -> None:
    """Synthetic sign check: when the permission moves the projection more than the prohibition, the activation
    interaction must come out negative and the decomposition must say so; and the reverse."""
    gen = np.random.default_rng(0)
    layers, dim = list(range(0, 65, 4)), 16
    v = gen.normal(size=dim)
    v /= np.linalg.norm(v)
    meta = [{"character": c, "animal": a, "scene": f"scene{k}"} for c in ("helpful", "dismissive")
            for a in ("bee", "crow") for k in range(100)]
    acts = gen.normal(size=(len(meta), len(layers), dim))
    for i, m in enumerate(meta):
        if m["character"] == "dismissive":
            acts[i] += 3.0 * v
    pids = [f"mt{i:03d}" for i in range(100)]
    z = {p: 0.0 for p in ("none",) + tuple(LADDER)}
    z.update({p: (i % 12) / 4 + 0.1 * (i // 12) for i, p in enumerate(LADDER)})
    base = gen.normal(size=(100, len(layers), 2, dim))

    def make(perm_scale):
        chats, rows_d = {}, {}
        for p, zp in z.items():
            for f, k in (("trigger", 1.0), ("permit", perm_scale)):
                chats[(p, "fixed", f)] = {"prompt_ids": pids,
                                          "acts": base + gen.normal(0, 0.05, size=base.shape) + k * zp * v}
                # probe: the prohibition matters more for stronger prompts (as in the real data)
                rows_d[(p, "fixed", f"{f}/start")] = {q: {"pooled": 2.6 - (1.0 if f == "trigger" else 0.4) * zp
                                                          + gen.normal(0, 0.05)} for q in pids}
        return dict(story_layer_acts=lambda li: acts[:, li], story_meta=meta, layers=layers,
                    get_chat=lambda k: chats.get(k), rows=lambda c: rows_d.get(c), n_perm=500)

    ok = True
    for name, scale, want_sign, want_perm_more in (("permission moves it more", 1.3, -1, True),
                                                   ("prohibition moves it more", 0.5, +1, False)):
        r = decompose(**make(scale))
        P = r["prompts"]
        n_more = sum(P[p]["X_perm"] > P[p]["X_trig"] for p in LADDER if z[p] > 0.2)
        n_strong = sum(z[p] > 0.2 for p in LADDER)
        good = (np.sign(r["s3_rho"]) == want_sign and r["rhos"]["T_int vs X_trig"] > 0.9
                and (n_more == n_strong) == want_perm_more and (n_more == 0) == (not want_perm_more))
        ok &= good
        print(f"[self-test] {name:27s} S3-style rho {r['s3_rho']:+.2f}, primary rho {r['primary_rho']:+.2f}, "
              f"permission larger in {n_more}/{n_strong} strong prompts  {'ok' if good else 'WRONG'}")
    # the input checks must pass a correct file and catch each kind of mismatch
    pids = [f"mt{i:03d}" for i in range(100)]
    good = {"prompt_ids": pids, "context_sha": [f"h{q}" for q in pids], "model": "M", "layers": [0, 4], "limit": 0,
            "system_prompt": PERSONAS["L_full_a"], "positions": ["end_of_turn", "pre_reply"]}
    expected = {("L_full_a", "trigger"): {q: f"h{q}" for q in pids}}
    probe = lambda cond: {q: f"h{q}" for q in pids}
    variants = {"correct": good, "99 prompts": {**good, "prompt_ids": pids[:99], "context_sha": good["context_sha"][:99]},
                "a changed conversation": {**good, "context_sha": ["x"] + good["context_sha"][1:]},
                "other layers": {**good, "layers": [0, 8]}, "other model": {**good, "model": "N"},
                "a --limit run": {**good, "limit": 40}, "wrong system prompt": {**good, "system_prompt": None}}
    for name, c in variants.items():
        found = check_inputs(lambda k: c, [("L_full_a", "trigger")], expected, probe, {"model": "M", "layers": [0, 4]})
        good_ = (not found) if name == "correct" else bool(found)
        ok &= good_
        print(f"[self-test] input check, {name:24s} {'passes' if not found else 'blocked'}  {'ok' if good_ else 'WRONG'}")
    G0 = check_inputs(lambda k: good, [("L_full_a", "trigger")], expected, lambda cond: {q: "y" for q in pids},
                      {"model": "M", "layers": [0, 4]})
    ok &= bool(G0)
    print(f"[self-test] input check, probe fingerprints differ  {'blocked  ok' if G0 else 'passes  WRONG'}")
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print("[self-test] all checks pass")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chats", default=str(RUNS / "ladder"),
                    help="folder with the chat_<persona>_fixed_<trigger|permit>.pt files")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    layer_acts, meta, layers, rows, ref = load_common()
    get = files_get_chat(Path(args.chats))
    bad = check_inputs(get, FILES, expected_fingerprints(("none",) + DEV_PERSONAS + tuple(LADDER)),
                       probe_fingerprints(), ref)
    if bad:
        raise SystemExit("INPUTS DON'T MATCH, so not interpreted:\n  " + "\n  ".join(bad))
    print(f"[checks] inputs: all {len(FILES)} files have the right prompts, model, layers, system prompt and "
          "fingerprints (and match the probe where it covers them)\n")
    res = decompose(layer_acts, meta, layers, get, rows)
    report(res)
    from . import summarize_probe as sp   # development prompts: their probe rows are in step 2's files
    Pd = {m: sp.per_prompt(m, "probe") for m in ("base", "hb_dc", "hc_db")}
    rows_d = lambda cond: sp.condition_rows(Pd, cond) if cond + ("bees",) in Pd["base"] else None
    print("\nDevelopment prompts (step 2's probe; controls only):")
    report(decompose(layer_acts, meta, layers, get, rows_d, personas=DEV_PERSONAS))
    out = Path(args.chats) / "decomposition.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "family"] + list(next(iter(res["prompts"].values()))))
        w.writeheader()
        for p, r in res["prompts"].items():
            w.writerow({"prompt": p, "family": FAMILY_OF[p], **{k: v for k, v in r.items()}})
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
