"""Build the two Axis measures and check them (axis/README.md, story imprinting/AXIS_RESULTS.md).

  python -m axis.build              # pod (CPU): runs/axis_acts/ + runs/axis/scores/ -> axis.pt, checks.txt, proj.csv
  python -m axis.build --acts runs/axis/acts36    # anywhere: the same at layer 36 only (no per-layer table)
  python -m axis.build sanity       # laptop: axis.pt + story states + probe chats -> sanity.txt
  python -m axis.build --self-test  # synthetic data with known answers

Build. Per layer, the authors' contrast: axis = mean(default states) - mean over roles of (mean of that role's kept
states), kept = judge score 3, each role weighted equally, roles with fewer than MIN_KEPT kept build replies dropped.
Built on the 12 build questions only.
  Axis (reply)  from reply_mean states: the paper's definition
  Axis (turn)   from end_of_turn states: a new measure, not a validation of the paper's Axis
Checks (layer 36 unless stated):
  stability       cosine of the axes built from each half of the build questions (6 + 6)
  held-out AUC    default vs kept role replies on the 6 held-out questions, at all three positions, for both
                  measures: P(a default state projects higher than a role state). The roles are the ones the axis
                  was built from, so this also rewards recognising those roles; hence also
  role-split AUC  the same, with the axis built from half the roles (md5 of the name) and tested on the other half
  validation      does Axis (reply) separate default from role states at end_of_turn, the position the probe
                  chats are read at (held-out AUC >= AUC_MIN)? If not, Stages 2-3 use Axis (turn), named as such.
Sanity (laptop; doesn't change anything): helpful vs dismissive story states (the easiest ordering; the story
position isn't validated for the Axis), no-prompt chats above the dismissive / sarcastic / terse ones, and the cosine
with the story direction (which says in advance how many Axis-vs-story disagreements to expect).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from .common import (ACTS, ACTS36, AUC_MIN, AXIS_COMMIT, KEEP_SCORE, LAYER, MEASURES, MIN_KEPT, OUT, POSITIONS,
                     question_split)


def parse_key(key: str) -> dict:
    role, p, q, s = key.split("|")
    return {"role": role, "prompt_index": int(p[1:]), "question_id": int(q[1:]), "sample": int(s[1:])}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie), by ranks (ties averaged)."""
    x = np.concatenate([pos, neg])
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def role_half(role: str) -> int:
    return int(hashlib.md5(role.encode()).hexdigest(), 16) % 2


def unit(v):
    return v / np.linalg.norm(v)


def load_role(acts_dir: Path, role: str):
    import torch
    d = torch.load(acts_dir / f"{role}.pt", map_location="cpu", weights_only=False)
    return d, d["acts"]


def build(acts_dir: Path, scores_dir: Path, split: dict, layer: int = LAYER) -> dict:
    """Two passes over the per-role files: means -> axes, then projections -> checks."""
    roles = sorted(f.stem for f in acts_dir.glob("*.pt") if f.stem != "default")
    half = {int(k): v for k, v in split["half"].items()}
    build_q = set(split["build"])
    scores = {r: json.load(open(scores_dir / f"{r}.json")) for r in roles if (scores_dir / f"{r}.json").exists()}
    missing = [r for r in roles if r not in scores]
    if missing:
        raise SystemExit(f"no judge scores for {len(missing)} roles, e.g. {missing[:5]}")

    d0, a0 = load_role(acts_dir, "default")
    layers = d0["layers"]
    li = layers.index(layer)
    meta0 = [parse_key(k) for k in d0["keys"]]
    a0 = a0.float().numpy()                                   # N x layers x 3 x d
    sel = lambda meta, qs: np.array([m["question_id"] in qs for m in meta])
    groups = {"build": build_q, "half0": {q for q in build_q if half[q] == 0}, "half1": {q for q in build_q if half[q] == 1},
              "rolesA": build_q, "rolesB": build_q}       # rolesA/B: build questions, roles of one md5 half only
    default_mean = {g: a0[sel(meta0, qs)].mean(0) for g, qs in groups.items()}

    role_sum = {g: np.zeros_like(default_mean["build"], dtype=np.float64) for g in groups}
    n_roles = {g: 0 for g in groups}
    kept_build, dropped, n_kept_total = {}, [], 0
    for r in roles:
        d, a = load_role(acts_dir, r)
        meta = [parse_key(k) for k in d["keys"]]
        kept = np.array([scores[r].get(k) == KEEP_SCORE for k in d["keys"]])
        kept_build[r] = int((kept & sel(meta, build_q)).sum())
        if kept_build[r] < MIN_KEPT:
            dropped.append(r)
            continue
        a = a.float().numpy()
        n_kept_total += int(kept.sum())
        for g, qs in groups.items():
            m = kept & sel(meta, qs)
            if g.startswith("roles") and "AB"[role_half(r)] != g[-1]:
                continue
            if m.any():
                role_sum[g] += a[m].mean(0)
                n_roles[g] += 1
    if not n_roles["build"]:
        raise SystemExit("no role has enough kept replies")
    axis = {g: (default_mean[g] - role_sum[g] / n_roles[g]).astype(np.float32) for g in groups}   # layers x 3 x d
    pos_of = {m: POSITIONS.index(p) for m, p in MEASURES.items()}
    A = {m: axis["build"][:, pos_of[m]] for m in MEASURES}                          # layers x d, raw
    U = {m: A[m] / np.linalg.norm(A[m], axis=-1, keepdims=True) for m in MEASURES}
    UR = {(m, g): unit(axis[g][li, pos_of[m]]) for m in MEASURES for g in ("rolesA", "rolesB")}   # layer `layer` only

    # pass 2: projections of every reply on both unit axes, every layer and position
    proj_rows, P = [], {"default": [], "role": []}            # P[...] = list of (split, kept, proj[layers x 3 x 2])
    cross = {"default": [], "rolesA": [], "rolesB": []}       # held out, at `layer`: projections on the other half's axes
    role_means = {}
    for r in ["default"] + [x for x in roles if x not in dropped]:
        d, a = load_role(acts_dir, r)
        a = a.float().numpy()
        pr = np.stack([np.einsum("nlpd,ld->nlp", a, U[m]) for m in MEASURES], -1)   # N x layers x 3 x 2
        # role-split: a role in half A is tested on the axis built from half B, and vice versa; the default on both
        test_on = ["rolesA", "rolesB"] if r == "default" else ["rolesB" if role_half(r) == 0 else "rolesA"]
        xr = {g: np.stack([a[:, li] @ UR[(m, g)] for m in MEASURES], -1) for g in test_on}      # N x 3 x 2
        for k, key in enumerate(d["keys"]):
            m = parse_key(key)
            s = None if r == "default" else scores[r].get(key)
            kept = r == "default" or s == KEEP_SCORE
            sp = "build" if m["question_id"] in build_q else "heldout"
            P["default" if r == "default" else "role"].append((sp, kept, pr[k]))
            if sp == "heldout" and kept:
                for g in test_on:
                    cross["default" if r == "default" else ("rolesA" if g == "rolesB" else "rolesB")].append((g, xr[g][k]))
            proj_rows.append({"key": key, "role": r, "split": sp, "score": s, "kept": int(kept),
                              **{f"{ms}_{pos}": float(pr[k, li, j, mi]) for mi, ms in enumerate(MEASURES)
                                 for j, pos in enumerate(POSITIONS)}})
        if r != "default":
            km = np.array([scores[r].get(k) == KEEP_SCORE for k in d["keys"]])
            role_means[r] = float(pr[km, li, pos_of["reply"], 0].mean())

    def stack(kind, sp):
        return np.stack([p for s, kept, p in P[kind] if s == sp and kept])
    aucs = {}
    for sp in ("heldout", "build"):
        dd, rr = stack("default", sp), stack("role", sp)
        for mi, ms in enumerate(MEASURES):
            for j, pos in enumerate(POSITIONS):
                for lk, L in enumerate(layers):
                    aucs[(sp, ms, pos, L)] = auc(dd[:, lk, j, mi], rr[:, lk, j, mi])
    for mi, ms in enumerate(MEASURES):          # role-split AUC: average of the two directions of the split
        for j, pos in enumerate(POSITIONS):
            v = []
            for test, ax_g in (("rolesA", "rolesB"), ("rolesB", "rolesA")):
                dd = np.array([p[j, mi] for g, p in cross["default"] if g == ax_g])
                rr = np.array([p[j, mi] for g, p in cross[test]])
                if len(rr):
                    v.append(auc(dd, rr))
            aucs[("rolesplit", ms, pos, layer)] = float(np.mean(v)) if v else float("nan")
    half_cos = {ms: [float(unit(axis["half0"][lk, pos_of[ms]]) @ unit(axis["half1"][lk, pos_of[ms]]))
                     for lk in range(len(layers))] for ms in MEASURES}
    use = "reply" if aucs[("heldout", "reply", "end_of_turn", layer)] >= AUC_MIN else "turn"
    return {"layers": layers, "layer": layer, "axis": A, "unit": U, "axis_all_positions": axis, "aucs": aucs,
            "half_cos": half_cos, "kept_build": kept_build, "dropped": dropped, "n_roles": n_roles,
            "n_default": len(meta0), "n_kept_total": n_kept_total, "use": use, "proj_rows": proj_rows,
            "role_means": role_means, "model": d0.get("model"), "split": split}


def checks_text(res: dict) -> str:
    L, layers = res["layer"], res["layers"]
    lk = layers.index(L)
    A, aucs, hc = res["axis"], res["aucs"], res["half_cos"]
    out = [f"Assistant Axis checks, layer {L} (hidden_states[{L}]) unless stated. Model {res['model']}.",
           f"Data: safety-research/assistant-axis @ {AXIS_COMMIT[:12]}; questions build {res['split']['build']}, "
           f"held out {res['split']['heldout']}.",
           f"Replies: default {res['n_default']}; roles {len(res['kept_build'])}, kept (score {KEEP_SCORE}) replies "
           f"{res['n_kept_total']} in the {len(res['kept_build']) - len(res['dropped'])} roles used.",
           f"Roles dropped (< {MIN_KEPT} kept build replies of 60): {len(res['dropped'])} of {len(res['kept_build'])} "
           f"({len(res['dropped']) / len(res['kept_build']):.0%})" + (": " + ", ".join(
               f"{r} ({res['kept_build'][r]})" for r in res["dropped"]) if res["dropped"] else ""),
           f"Roles per half-question axis: {res['n_roles']['half0']} and {res['n_roles']['half1']}.", "",
           "Two measures, named apart:",
           "  Axis (reply)  built from reply_mean states: the paper's definition",
           "  Axis (turn)   built from end_of_turn states: a new measure, not a validation of the paper's Axis", "",
           f"Norms: Axis (reply) {np.linalg.norm(A['reply'][lk]):.2f}, Axis (turn) {np.linalg.norm(A['turn'][lk]):.2f}; "
           f"cosine between them {float(res['unit']['reply'][lk] @ res['unit']['turn'][lk]):+.3f}",
           f"Stability (cosine of the two half-question axes): Axis (reply) {hc['reply'][lk]:.3f}, "
           f"Axis (turn) {hc['turn'][lk]:.3f}", "",
           "Held-out AUC, default vs kept role replies (6 held-out questions; 0.5 = no separation):",
           f"  {'measure':14s} " + " ".join(f"{p:>12s}" for p in POSITIONS)]
    for ms in MEASURES:
        out.append(f"  Axis ({ms}){'':{8 - len(ms)}s} " + " ".join(f"{aucs[('heldout', ms, p, L)]:12.3f}" for p in POSITIONS))
    out.append(f"  {'role-split':14s} " + "; ".join(f"Axis ({ms}) " + ", ".join(
        f"{aucs[('rolesplit', ms, p, L)]:.3f}" for p in POSITIONS) for ms in MEASURES)
        + "   (axis from half the roles, tested on the other half)")
    out.append("  (in-sample, build questions: " + "; ".join(
        f"Axis ({ms}) " + ", ".join(f"{aucs[('build', ms, p, L)]:.3f}" for p in POSITIONS) for ms in MEASURES) + ")")
    a_val = aucs[("heldout", "reply", "end_of_turn", L)]
    out += ["", f"VALIDATION: Axis (reply) at end_of_turn, held-out AUC {a_val:.3f} "
                f"({'separates' if a_val >= AUC_MIN else 'does NOT separate'}; bar {AUC_MIN}).",
            f"Stages 2-3 use: Axis ({res['use']})" + ("" if res["use"] == "reply" else
                                                    " -- a new measure, not the paper's Axis"), "",
            "Per layer (held-out AUC at end_of_turn / at reply_mean; half-question cosine):",
            f"  {'layer':>5s}  {'reply: eot':>10s} {'reply: rm':>9s} {'cos':>6s}   {'turn: eot':>9s} {'turn: rm':>8s} {'cos':>6s}"]
    for k, Lx in enumerate(layers):
        out.append(f"  {Lx:5d}  {aucs[('heldout', 'reply', 'end_of_turn', Lx)]:10.3f} "
                   f"{aucs[('heldout', 'reply', 'reply_mean', Lx)]:9.3f} {hc['reply'][k]:6.3f}   "
                   f"{aucs[('heldout', 'turn', 'end_of_turn', Lx)]:9.3f} {aucs[('heldout', 'turn', 'reply_mean', Lx)]:8.3f} "
                   f"{hc['turn'][k]:6.3f}")
    rm = sorted(res["role_means"].items(), key=lambda x: x[1])
    out += ["", f"Roles on Axis (reply), mean of kept replies at reply_mean (default mean "
                f"{np.mean([r['reply_reply_mean'] for r in res['proj_rows'] if r['role'] == 'default']):.2f}):",
            "  lowest:  " + ", ".join(f"{r} {v:.1f}" for r, v in rm[:10]),
            "  highest: " + ", ".join(f"{r} {v:.1f}" for r, v in rm[-10:])]
    return "\n".join(out)


def save(res: dict, out: Path) -> None:
    import torch
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": res["layers"], "layer": res["layer"], "use": res["use"], "commit": AXIS_COMMIT,
                "axis": {m: torch.tensor(v) for m, v in res["axis"].items()},
                "half": {m: torch.tensor(np.stack([res["axis_all_positions"][g][:, POSITIONS.index(p)]
                                                   for g in ("half0", "half1")]))
                         for m, p in MEASURES.items()},
                "positions_built_from": dict(MEASURES), "half_cos": res["half_cos"],
                "auc": {"|".join(map(str, k)): v for k, v in res["aucs"].items()},
                "roles_dropped": res["dropped"], "kept_build": res["kept_build"], "n_roles": res["n_roles"],
                "split": res["split"], "model": res["model"]}, out / "axis.pt")
    with (out / "proj.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["proj_rows"][0]))
        w.writeheader()
        w.writerows(res["proj_rows"])
    text = checks_text(res)
    (out / "checks.txt").write_text(text + "\n")
    print(text)


# ---------- sanity (laptop) ----------

def sanity(axis_path: Path, out: Path) -> None:
    import torch

    from persona_flip.analyze_ladder import story_direction
    from persona_flip.common import RUNS
    ax = torch.load(axis_path, map_location="cpu", weights_only=False)
    L = ax["layer"]
    st = torch.load(RUNS / "ladder" / "stories.pt", map_location="cpu", weights_only=False)
    assert st["layers"] == ax["layers"], "story states and axis use different layers"
    li = st["layers"].index(L)
    sli, u, best, _ = story_direction(lambda i: st["acts"][:, i].float().numpy(), st["meta"], st["layers"])
    assert best["layer"] == L, f"story direction chose layer {best['layer']}"
    U = {m: unit(ax["axis"][m][li].numpy().astype(np.float64)) for m in MEASURES}
    lines = [f"Sanity checks (layer {L}). Stages 2-3 use Axis ({ax['use']}). Axis points toward the default Assistant "
             "(higher = more Assistant-like); the story direction points from the helpful to the dismissive character.",
             "", "Cosine with the story direction (dismissive minus helpful): " + ", ".join(
                 f"Axis ({m}) {float(U[m] @ u):+.3f}" for m in MEASURES)]
    acts = st["acts"][:, li].float().numpy()
    ch = np.array([m["character"] for m in st["meta"]])
    lines += ["", "Story states at the end of the prohibition (all 5,472 stories; the position isn't validated for the "
                  "Axis): helpful minus dismissive, in pooled SDs, and AUC(helpful above dismissive):"]
    for m in MEASURES:
        p = acts @ U[m]
        h, d = p[ch == "helpful"], p[ch == "dismissive"]
        sd = np.sqrt((h.var(ddof=1) + d.var(ddof=1)) / 2)
        lines.append(f"  Axis ({m}): {(h.mean() - d.mean()) / sd:+.2f} SD, AUC {auc(h, d):.3f}  "
                     f"(raw gap {h.mean() - d.mean():+.2f})")
    lines += ["", "Probe chats (fixed history, after the prohibition), end_of_turn [pre_reply]: persona minus no prompt, "
                  "raw units, mean over 100 conversations [95% CI]; the no-prompt chats should sit highest:"]
    chats = {}
    for p in ("none", "dismissive", "sarcastic", "terse"):
        c = torch.load(RUNS / "ladder" / f"chat_{p}_fixed_trigger.pt", map_location="cpu", weights_only=False)
        chats[p] = (c["prompt_ids"], c["acts"][:, li].float().numpy())    # N x 2 x d (end_of_turn, pre_reply)
    rng = np.random.default_rng(0)
    shifts = {}
    for m in MEASURES:
        base = dict(zip(chats["none"][0], chats["none"][1] @ U[m]))
        for p in ("dismissive", "sarcastic", "terse"):
            v = np.array([pr - base[q] for q, pr in zip(chats[p][0], chats[p][1] @ U[m])])   # N x 2
            boots = np.array([v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(4000)])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            shifts[(m, p)] = float(v[:, 0].mean())
            lines.append(f"  Axis ({m}) {p:10s} {v[:, 0].mean():+7.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]   "
                         f"[{v[:, 1].mean():+7.2f}]  {'below no prompt' if hi[0] < 0 else 'NOT below no prompt'}")
    lines += ["", "The dismissive prompt's own shift at end_of_turn (the steering unit in axis/make_direction.py): "
              + ", ".join(f"Axis ({m}) {shifts[(m, 'dismissive')]:+.2f}" for m in MEASURES)]
    text = "\n".join(lines)
    (out / "sanity.txt").write_text(text + "\n")
    print(text)


# ---------- self-test ----------

def self_test() -> None:
    import torch
    rng = np.random.default_rng(0)
    d, layers, n_roles = 48, [0, 36], 120
    split = {"build": list(range(12)), "heldout": list(range(12, 18)), "half": {str(q): int(q >= 6) for q in range(12)}}
    e = unit(rng.normal(size=d))
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[self-test] {name:72s} {'ok' if cond else 'WRONG'}")

    def make(tmp: Path, signal: float, few: set, role_scale: float = 2.0):
        acts, scores = tmp / "acts", tmp / "scores"
        acts.mkdir()
        scores.mkdir()
        keys = [f"default|p{p}|q{q}|s{s}" for p in range(5) for q in range(18) for s in range(5)]
        x = rng.normal(size=(len(keys), 2, 3, d)) + signal * e
        torch.save({"keys": keys, "layers": layers, "acts": torch.tensor(x, dtype=torch.float16)}, acts / "default.pt")
        truth = []
        for r in range(n_roles):
            off = rng.normal(size=d) * role_scale                         # role-specific offset
            truth.append(off)
            keys = [f"role{r}|p{p}|q{q}|s0" for p in range(5) for q in range(18)]
            x = rng.normal(size=(len(keys), 2, 3, d)) + off
            torch.save({"keys": keys, "layers": layers, "acts": torch.tensor(x, dtype=torch.float16)}, acts / f"role{r}.pt")
            p_keep = 0.2 if r in few else 0.8
            json.dump({k: int(3 if rng.random() < p_keep else rng.integers(0, 3)) for k in keys},
                      open(scores / f"role{r}.json", "w"))
        return acts, scores, truth

    few = {3, 7}
    with tempfile.TemporaryDirectory() as t:
        acts, scores, _ = make(Path(t), 5.0, few)
        res = build(acts, scores, split, 36)
        check("roles with < 20 kept build replies are exactly the dropped ones", set(res["dropped"]) == {f"role{r}" for r in few})
        # direct computation of the formula: mean(default) - equal-weight mean of kept role means, build questions
        dd = torch.load(acts / "default.pt", weights_only=False)
        qd = np.array([parse_key(k)["question_id"] < 12 for k in dd["keys"]])
        want = dd["acts"].float().numpy()[qd, 1, 0].mean(0)
        rm = []
        for r in range(n_roles):
            if r in few:
                continue
            x = torch.load(acts / f"role{r}.pt", weights_only=False)
            sc = json.load(open(scores / f"role{r}.json"))
            m = np.array([sc[k] == 3 and parse_key(k)["question_id"] < 12 for k in x["keys"]])
            rm.append(x["acts"].float().numpy()[m, 1, 0].mean(0))
        want = want - np.mean(rm, 0)
        check("Axis (reply) = mean(default) - equal-weight mean of kept role means (build)",
              np.allclose(res["axis"]["reply"][1], want, atol=1e-4))
        check(f"planted direction recovered (cosine {float(res['unit']['reply'][1] @ e):.3f} > 0.8)",
              res["unit"]["reply"][1] @ e > 0.8)
        check(f"held-out AUC near 1 with a planted gap ({res['aucs'][('heldout', 'reply', 'end_of_turn', 36)]:.3f})",
              res["aucs"][("heldout", "reply", "end_of_turn", 36)] > 0.9)
        check(f"role-split AUC high with a planted gap ({res['aucs'][('rolesplit', 'reply', 'end_of_turn', 36)]:.3f})",
              res["aucs"][("rolesplit", "reply", "end_of_turn", 36)] > 0.85)
        check(f"half-question axes agree with a planted gap ({res['half_cos']['reply'][1]:.3f})", res["half_cos"]["reply"][1] > 0.7)
        check("validation passes -> Stages 2-3 use Axis (reply)", res["use"] == "reply")
        save(res, Path(t) / "out")
        check("axis.pt, checks.txt and proj.csv written", all((Path(t) / "out" / f).exists()
                                                              for f in ("axis.pt", "checks.txt", "proj.csv")))
    with tempfile.TemporaryDirectory() as t:
        acts, scores, _ = make(Path(t), 0.0, set(), role_scale=0.0)
        res = build(acts, scores, split, 36)
        a = res["aucs"][("heldout", "reply", "end_of_turn", 36)]
        check(f"pure noise: held-out AUC near 0.5 ({a:.3f})", 0.3 < a < 0.7)
        check("pure noise: validation fails -> Stages 2-3 would use Axis (turn)", res["use"] == "turn")
    with tempfile.TemporaryDirectory() as t:
        acts, scores, _ = make(Path(t), 0.0, set())
        res = build(acts, scores, split, 36)
        a, b = res["aucs"][("heldout", "reply", "end_of_turn", 36)], res["aucs"][("rolesplit", "reply", "end_of_turn", 36)]
        check(f"role offsets, no default signal: held-out AUC inflated ({a:.3f}), role-split near 0.5 ({b:.3f})",
              a > b + 0.1 and 0.3 < b < 0.7)
    check("AUC with ties averaged", auc(np.array([1.0, 2.0]), np.array([1.0, 0.0])) == 0.875)
    if not ok:
        raise SystemExit("[self-test] FAILED")
    print("[self-test] all checks pass")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="build", choices=["build", "sanity"])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--acts", default=None, help="default: runs/axis_acts if it has the default's states, else runs/axis/acts36")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.command == "sanity":
        return sanity(OUT / "axis.pt", OUT)
    acts = Path(args.acts) if args.acts else ACTS if (ACTS / "default.pt").exists() else ACTS36
    print(f"[build] states from {acts}")
    save(build(acts, OUT / "scores", question_split(), args.layer), OUT)
    (OUT / "AXIS_DONE").touch()


if __name__ == "__main__":
    main()
