"""Follow-up checks on the decomposition (exploratory, after seeing its table; not part of the frozen preregistration).

decompose_interaction showed the permission moving the projection more than the prohibition for every ladder prompt.
These checks ask whether that is one constant scaling, and whether it depends on where the state is read:
  - per prompt, a 95% CI over the 100 conversations for X_int (paired: the same conversations in all four files);
  - X_perm = k * X_trig through the origin, k with a CI over prompts, and what's left of X_int once that scaling is
    removed, against the probe's interaction T_int;
  - the same probe fit, T_perm = c * T_trig (the behavioural counterpart of k);
  - the same at the pre-reply position, and at neighbouring layers (direction built by the frozen rule at each).

  python -m persona_flip.decompose_extras    # same inputs as decompose_interaction
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .analyze_ladder import POS, balanced_mean, chat_scores, spearman, split, story_direction, t_shift
from .common import GLOBAL_SEED, RUNS
from .decompose_interaction import files_get_chat, load_common
from .ladder import DEV_PERSONAS, LADDER

N_BOOT = 5000


def through_origin(x: np.ndarray, y: np.ndarray, rng) -> tuple:
    """(slope of y = k * x, 2.5%, 97.5% over prompts, R^2)."""
    k = (x @ y) / (x @ x)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        boot.append((x[i] @ y[i]) / (x[i] @ x[i]))
    res = y - k * x
    return k, np.percentile(boot, 2.5), np.percentile(boot, 97.5), 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))


def probe_parts(rows) -> dict:
    """prompt -> (T_trig, T_perm), pooled, for the ladder prompts."""
    r0t, r0p = rows(("none", "fixed", "trigger/start")), rows(("none", "fixed", "permit/start"))
    return {p: (t_shift(rows((p, "fixed", "trigger/start")), r0t, "pooled"),
                t_shift(rows((p, "fixed", "permit/start")), r0p, "pooled")) for p in LADDER}


def one_reading(get_chat, li: int, u: np.ndarray, sd: float, pos: int, T: dict, label: str, rng,
                per_prompt: bool = False) -> dict:
    S = {(p, f): chat_scores(get_chat((p, "fixed", f)), li, u)
         for p in ("none",) + tuple(LADDER) + DEV_PERSONAS for f in ("trigger", "permit")}
    pids = sorted(S[("none", "trigger")])
    a = {key: np.array([v[q][pos] for q in pids]) / sd for key, v in S.items()}
    boot_idx = rng.integers(0, len(pids), size=(N_BOOT, len(pids)))
    P = {}
    for p in tuple(LADDER) + DEV_PERSONAS:
        xt, xp = a[(p, "trigger")] - a[("none", "trigger")], a[(p, "permit")] - a[("none", "permit")]
        bs = (xt - xp)[boot_idx].mean(1)
        P[p] = {"xt": xt.mean(), "xp": xp.mean(), "xi": (xt - xp).mean(),
                "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)}
    L = list(LADDER)
    xt, xp = np.array([P[p]["xt"] for p in L]), np.array([P[p]["xp"] for p in L])
    tt, tp = np.array([T[p][0] for p in L]), np.array([T[p][1] for p in L])
    k, k_lo, k_hi, r2 = through_origin(xt, xp, rng)
    left = (k - 1) * xt - (xp - xt)     # X_int with the constant scaling removed (= k*X_trig - X_perm)
    base = (a[("none", "trigger")] - a[("none", "permit")]).mean()
    print(f"\n== {label}")
    print(f"   no prompt: prohibition minus permission {base:+.3f} SDs")
    print(f"   primary rho (X_trig, T_trig) {spearman(xt, tt):+.3f}; S3 rho (X_int, T_int) {spearman(xt - xp, tt - tp):+.3f}")
    print(f"   X_perm > X_trig for {sum(xp > xt)}/24 prompts; X_int's CI over conversations below 0 for "
          f"{sum(P[p]['hi'] < 0 for p in L)}/24, above 0 for {sum(P[p]['lo'] > 0 for p in L)}/24")
    print(f"   X_perm = k * X_trig: k = {k:.2f} [{k_lo:.2f}, {k_hi:.2f}], R^2 = {r2:.3f}")
    print(f"   X_int with the scaling removed: rho with T_int {spearman(left, tt - tp):+.3f}, "
          f"with X_trig {spearman(left, xt):+.3f}")
    if per_prompt:
        print(f"   {'prompt':22s} {'X_trig':>7s} {'X_perm':>7s} {'ratio':>6s} {'X_int':>7s} {'95% CI':>17s}  "
              f"{'T_trig':>6s} {'T_perm':>6s}")
        for p in sorted(L, key=lambda q: -P[q]["xt"]) + list(DEV_PERSONAS):
            r = P[p]
            ratio = f"{r['xp'] / r['xt']:6.2f}" if abs(r["xt"]) > 0.05 else f"{'':6s}"
            t = f"{T[p][0]:+6.2f} {T[p][1]:+6.2f}" if p in T else f"{'(dev)':>6s}"
            print(f"   {p:22s} {r['xt']:+7.3f} {r['xp']:+7.3f} {ratio} {r['xi']:+7.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  {t}")
    return {"k": k, "k_lo": k_lo, "k_hi": k_hi, "r2": r2}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chats", default=str(RUNS / "ladder"))
    ap.add_argument("--layers", default="28,32,40,44", help="neighbouring layers to repeat the end-of-turn check on")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(GLOBAL_SEED)
    layer_acts, meta, layers, rows, _ = load_common()
    get_chat = files_get_chat(Path(args.chats))
    T = probe_parts(rows)
    L = list(LADDER)
    tt, tp = np.array([T[p][0] for p in L]), np.array([T[p][1] for p in L])
    c, c_lo, c_hi, r2 = through_origin(tt, tp, rng)
    print("Exploratory, after seeing the decomposition. Units: gate-story SDs (activations), nats (probe, pooled).")
    print(f"\nBehaviour (probe): T_perm = c * T_trig, c = {c:.2f} [{c_lo:.2f}, {c_hi:.2f}], R^2 = {r2:.2f}; "
          f"T_trig > T_perm for {sum(tt > tp)}/24 prompts (mean T_trig {tt.mean():+.2f}, T_perm {tp.mean():+.2f})")

    li, u, best, table = story_direction(layer_acts, meta, layers)
    one_reading(get_chat, li, u, best["sd_gate"], POS["end_of_turn"], T,
                f"layer {best['layer']}, end of the follow-up turn (the preregistered reading point)", rng, True)
    one_reading(get_chat, li, u, best["sd_gate"], POS["pre_reply"], T,
                f"layer {best['layer']}, last token before the reply", rng)
    build = [i for i, m in enumerate(meta) if split(m["scene"]) == "build"]
    for i, row in enumerate(table):
        if str(row["layer"]) not in args.layers.split(","):
            continue
        acts = layer_acts(i)
        d = balanced_mean(acts, meta, build, "dismissive") - balanced_mean(acts, meta, build, "helpful")
        one_reading(get_chat, i, d / np.linalg.norm(d), row["sd_gate"], POS["end_of_turn"], T,
                    f"layer {row['layer']}, end of the follow-up turn (gate d' {row['d_prime_gate']:.2f})", rng)


if __name__ == "__main__":
    main()
