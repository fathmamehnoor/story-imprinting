"""Do alpha-0 scores from steer_logprob.py reproduce saved probe rows?

  python steering/check_probe.py STEER.jsonl KENNEY_probe.jsonl.gz [runs/probe_si27_<model>.jsonl]

Kenney's file: no system prompt, his "trigger/start" and "neutral/start" contexts.
This repo's probe file (fixed history): persona none and dismissive, "trigger/start" and
"permit/start". Compared per (prompt, animal, sentence). Exit code 1 if any comparison has correlation
below 0.999 or mean absolute difference above 0.15 nats. Note: two vLLM runs of the base model's probe
(this repo's and Kenney's) differ by 0.13 nats per row, so 0.15 is tight; read the printed numbers.
"""
import gzip, json, sys
import numpy as np

ours = [json.loads(l) for l in open(sys.argv[1])]
op = lambda f: gzip.open(f, "rt") if f.endswith(".gz") else open(f)
bad = False
def cmp(label, a, b):
    global bad
    k = sorted(set(a) & set(b))
    if not k:
        print(f"{label}: no shared rows"); return
    x, y = np.array([a[i] for i in k]), np.array([b[i] for i in k])
    c, mad = np.corrcoef(x, y)[0, 1], np.abs(x - y).mean()
    ok = c >= 0.999 and mad <= 0.15; bad |= not ok
    print(f"{'PASS' if ok else 'FAIL'} {label}: {len(k)} rows, {len({i[0] for i in k})} prompts | corr {c:.5f} | "
          f"mean abs diff {mad:.3f} | max {np.abs(x - y).max():.3f} | mean diff {np.mean(x - y):+.3f} nats")
def mine(system, fu):
    return {(r["prompt_id"], r["animal"], r["probe"]): r["logprob"] for r in ours
            if r["alpha"] == 0 and r["system"] == system and r["followup"] == fu}
his = [json.loads(l) for l in op(sys.argv[2])]
for fu in ("trigger", "neutral"):
    cmp(f"Kenney none/{fu}", mine("none", fu),
        {(r["prompt_id"], r["animal"], r["probe"]): r["logprob"] for r in his if r["context"] == f"{fu}/start"})
if len(sys.argv) > 3:
    repo_rows = [json.loads(l) for l in op(sys.argv[3])]   # this repo's probe rows
    for persona in ("none", "dismissive"):
        for fu in ("trigger", "permit"):
            cmp(f"repo {persona}/{fu}", mine(persona, fu),
                {(r["prompt_id"], r["animal"], r["probe"]): r["logprob"] for r in repo_rows
                 if r["persona"] == persona and r["history"] == "fixed" and r["context"] == f"{fu}/start"})
sys.exit(1 if bad else 0)
