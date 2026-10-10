"""Score how fully each role reply plays its role, 0-3, with the role's own eval_prompt (laptop + OpenRouter key).

The authors' pipeline/3_judge.py: their judge model (gpt-4.1-mini) and prompt, with the first number in the answer as
the score (assistant_axis/judge.py). Two differences: temperature 0 (theirs is 1), and the call goes through the
Qwen repo's cached OpenRouter client, so a rerun or a resumed run costs nothing for replies already scored. The
default replies aren't judged (as in their pipeline). Score 3 = fully playing the role: only those replies are kept.

  set -a; source .env; set +a                          # OPENROUTER_API_KEY
  python -m axis.judge --dry-run                       # one judge prompt, the number of calls, an estimated cost
  python -m axis.judge --pilot 50                      # 50 random replies: scores, examples, cost per call
  python -m axis.judge                                 # everything in runs/axis/replies/ -> runs/axis/scores/<role>.json

Writes runs/axis/scores/<role>.json ({reply key: score or null}) and runs/axis/scores/summary.txt; after a full
run over all 275 roles' complete replies, runs/axis/scores/JUDGE_DONE (an unreadable answer counts as not kept).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import persona_flip.common  # noqa: F401  (puts the Qwen repo on the path, for src.common.llm)
from persona_flip.common import GLOBAL_SEED

from .common import JUDGE_MODEL, KEEP_SCORE, MIN_KEPT, OUT, load_roles, read_jsonl

SCORES = OUT / "scores"
NONCE = "axis-judge-v1"
PRICE = {"prompt": 0.40e-6, "completion": 1.60e-6}   # openai/gpt-4.1-mini on OpenRouter, USD per token (2026-10-09)


def parse_score(text: str):
    """The authors' rule: the first whole number in the answer, if it's 0-3."""
    nums = re.findall(r"\b(\d+)\b", text.strip())
    return int(nums[0]) if nums and 0 <= int(nums[0]) <= 3 else None


def judge_prompt(row: dict, roles: dict) -> str:
    return roles[row["role"]]["eval_prompt"].format(question=row["question"], answer=row["reply"])


def call(prompt: str) -> tuple:
    from src.common import llm
    res = llm.chat(JUDGE_MODEL, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10, nonce=NONCE)
    return parse_score(res.text), res.text, res.usage, res.cached


def role_rows() -> dict:
    out = {}
    for f in sorted((OUT / "replies").glob("*.jsonl")):
        if f.stem != "default":
            out[f.stem] = read_jsonl(f)
    return out


def summarize(scores: dict, rows: dict) -> str:
    lines, dist, kept_build, unread = [], {s: 0 for s in (0, 1, 2, 3)}, {}, 0
    for role, sc in scores.items():
        for r in rows[role]:
            s = sc.get(r["key"])
            if s is None:
                unread += 1
                continue
            dist[s] += 1
        kept_build[role] = sum(sc.get(r["key"]) == KEEP_SCORE for r in rows[role] if r["split"] == "build")
    n = sum(dist.values())
    low = sorted((k, v) for role, v in kept_build.items() for k in [role] if v < MIN_KEPT)
    lines.append(f"{len(scores)} roles, {n} replies scored, {unread} unreadable or missing")
    lines.append("score distribution: " + ", ".join(f"{s}: {dist[s]} ({dist[s] / max(1, n):.0%})" for s in dist))
    kb = sorted(kept_build.values())
    if kb:
        lines.append(f"kept (score {KEEP_SCORE}) build replies per role, of 60: min {kb[0]}, median {kb[len(kb) // 2]}, "
                     f"max {kb[-1]}")
    lines.append(f"roles with fewer than {MIN_KEPT} kept build replies (dropped): {len(low)} of {len(scores)}"
                 + (": " + ", ".join(f"{k} ({v})" for k, v in low) if low else ""))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", type=int, default=0, help="score N random replies only, and show examples")
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    roles, rows = load_roles(), role_rows()
    if args.roles:
        rows = {k: v for k, v in rows.items() if k in args.roles}
    if not rows:
        raise SystemExit("no role replies in runs/axis/replies/ (python -m axis.generate)")
    todo = [r for role in rows for r in rows[role]]
    if args.pilot:
        todo = random.Random(GLOBAL_SEED).sample(todo, min(args.pilot, len(todo)))
    if args.dry_run:
        p = judge_prompt(todo[0], roles)
        chars = sum(len(judge_prompt(r, roles)) for r in todo)
        print(f"--- judge prompt for {todo[0]['key']} ---\n{p}\n---")
        print(f"{len(todo)} calls, about {chars / 4:,.0f} prompt tokens: about ${chars / 4 * PRICE['prompt']:.2f} "
              f"(4 characters per token; only uncached calls cost anything)")
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set: `set -a; source .env; set +a` first")

    got, tokens, new = {}, [0, 0], 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(call, judge_prompt(r, roles)): r["key"] for r in todo}
        for i, fut in enumerate(as_completed(futs)):
            score, text, usage, cached = fut.result()
            got[futs[fut]] = (score, text)
            if not cached:
                new += 1
                tokens[0] += usage.get("prompt_tokens", 0)
                tokens[1] += usage.get("completion_tokens", 0)
            if (i + 1) % 2000 == 0:
                print(f"[judge] {i + 1}/{len(todo)}, {new} new calls, about "
                      f"${tokens[0] * PRICE['prompt'] + tokens[1] * PRICE['completion']:.2f} so far", flush=True)
    cost = tokens[0] * PRICE["prompt"] + tokens[1] * PRICE["completion"]
    unread = [k for k, (s, _) in got.items() if s is None]

    if args.pilot:
        by = {}
        for r in todo:
            by.setdefault(got[r["key"]][0], []).append(r)
        print(f"[judge] pilot: {len(todo)} replies from {len({r['role'] for r in todo})} roles")
        print("  scores: " + ", ".join(f"{s}: {len(v)}" for s, v in sorted(by.items(), key=lambda x: str(x[0]))))
        for s, v in sorted(by.items(), key=lambda x: str(x[0])):
            for r in v[:2]:
                print(f"\n  [{s}] {r['key']} ({r['system'][:70]}...)\n  Q: {r['question']}\n  A: "
                      + " ".join(r["reply"].split())[:400])
        if unread:
            print(f"\n  unreadable: {[(k, got[k][1]) for k in unread[:5]]}")
        if new:
            per = cost / new
            n_all = sum(len(v) for v in role_rows().values())
            print(f"\n[judge] {new} new calls, ${cost:.4f} (${per * 1000:.3f} per 1,000 calls; "
                  f"all {n_all} role replies: about ${per * n_all:.2f})")
        return

    SCORES.mkdir(parents=True, exist_ok=True)
    scores = {role: {r["key"]: got[r["key"]][0] for r in rr} for role, rr in rows.items()}
    for role, sc in scores.items():
        (SCORES / f"{role}.json").write_text(json.dumps(sc))
    text = summarize(scores, rows)
    print(text)
    print(f"[judge] {new} new calls, {tokens[0]:,} prompt + {tokens[1]:,} completion tokens, about ${cost:.2f}")
    if not args.roles:
        (SCORES / "summary.txt").write_text(text + "\n")
        n_roles = len(load_roles())
        if len(rows) == n_roles and all(len(v) == 90 for v in rows.values()):
            (SCORES / "JUDGE_DONE").write_text(f"{len(got)} replies, {len(unread)} unreadable\n")
        else:
            print(f"[judge] {len(rows)} of {n_roles} roles have replies so far: not marking JUDGE_DONE")
    if unread:
        print(f"[judge] {len(unread)} unreadable answers, e.g. {[(k, got[k][1]) for k in unread[:3]]}: "
              "they count as not kept")


if __name__ == "__main__":
    main()
