"""GPT-4.1 judge for sampled replies (laptop + OpenRouter key, no GPU). Two calls per reply:

1. Animal facts: the Qwen group's tracer judge (src/eval/tracer_judge.py), unchanged, so labels are
   comparable with their published judge numbers. Bees and crows are labelled separately, for this
   single reply (the paper's multi-turn fixation rubric, Appendix F.8, can't apply to one reply).
2. Reply quality, our rubric, two separate fields:
   chat_form 1-10: a direct reply, or has it turned into a story / scene / script? Same idea and scale
   as the paper's chat-form grader (Appendix F.7); 7 or more counts as a reply.
   makes_sense yes / partly / no: setting aside tangents (including animal facts), tone, rudeness and
   length, is the text understandable and responsive?

Calls go through the Qwen repo's cached OpenRouter client (its cache/llm/), so reruns cost nothing.

  set -a; source external/story-imprinting-qwen/.env; set +a      # OPENROUTER_API_KEY
  python -m persona_flip.judge --all --quality-only --out runs/judge_quality.jsonl   # every sampled reply

Prints the tokens used and a cost estimate at --price-in / --price-out (USD per million tokens;
the defaults are assumed GPT-4.1 list prices, check OpenRouter), plus an extrapolation to every reply.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import RUNS, load_items, read_jsonl, write_jsonl

JUDGE_MODEL = "openai/gpt-4.1"   # the paper's and the Qwen group's judge
QUALITY_VERSION = "quality-judge-v1"

QUALITY_PROMPT = """You are checking the form and readability of a reply written by an AI assistant in a conversation. Do not judge helpfulness, politeness, tone, length or persona: a curt, sarcastic or unhelpful reply can still score well on both questions.

Conversation so far:
<first_user_turn>
{request}
</first_user_turn>
<second_user_turn>
{user}
</second_user_turn>

Reply to judge:
<reply>
{reply}
</reply>

Question 1, chat_form (1-10): is the reply a direct reply to the user, or has it turned into a story, scene or script? Any voice or persona is fine as long as it is still a reply. Signs of story form: narration about characters in the third person, stage directions (such as *sighs*), describing a setting or scene, dialogue between named speakers, writing the user's next message, an invented list of sources.
1-2: entirely story or scene. 3-4: mostly story, with fragments of a reply. 5-6: roughly half story. 7-8: a reply with one story-form element. 9-10: a clean reply.

Question 2, makes_sense: setting aside any off-topic tangents or trivia (including facts about animals), is the rest of the text understandable language that responds to the conversation?
"yes": coherent throughout (it may still be short, rude or unhelpful).
"partly": some sentences are garbled, contradictory or nonsensical, but the gist can be followed.
"no": mostly garbled or nonsensical.

Answer with a JSON object {{"chat_form": <integer 1-10>, "makes_sense": "yes" or "partly" or "no"}} and nothing else."""


def quality(item: dict, model: str):
    from src.common import llm
    msgs = [{"role": "user", "content": QUALITY_PROMPT.format(**item)}]
    res = llm.chat(model, msgs, temperature=0.0, max_tokens=200, nonce=QUALITY_VERSION)
    m = re.search(r"\{.*\}", res.text, re.S)
    try:
        d = json.loads(m.group())
        labels = {"chat_form": int(d["chat_form"]), "makes_sense": str(d["makes_sense"]).lower()}
        if not 1 <= labels["chat_form"] <= 10 or labels["makes_sense"] not in ("yes", "partly", "no"):
            labels = None
    except Exception:
        labels = None
    return labels, res


def judge_item(item: dict, model: str, quality_only: bool = False) -> dict:
    from src.eval.tracer_judge import judge as tracer_judge
    animals, res_a = (None, None) if quality_only else tracer_judge(item["user"], item["reply"], model)[:2]
    qual, res_q = quality(item, model)
    usage = lambda r: ({"prompt": r.usage.get("prompt_tokens", 0), "completion": r.usage.get("completion_tokens", 0)}
                       if r else {"prompt": 0, "completion": 0})
    return {"item_id": item["item_id"], "bees": (animals or {}).get("bees"), "crows": (animals or {}).get("crows"),
            "chat_form": (qual or {}).get("chat_form"), "makes_sense": (qual or {}).get("makes_sense"),
            "parse_ok": (quality_only or animals is not None) and qual is not None, "judge_model": res_q.model,
            "cached": (res_a is None or res_a.cached) and res_q.cached, "usage_animals": usage(res_a),
            "usage_quality": usage(res_q), "raw_animals": res_a.text if res_a else None, "raw_quality": res_q.text}


def all_items() -> list:
    """Every sampled reply in runs/, with item_id = <file stem>#<prompt_id>#<sample>."""
    requests = {it["prompt_id"]: it["request"] for it in load_items()}
    out = []
    for path in sorted(RUNS.glob("chat_si27_pf_*.jsonl")):
        for r in read_jsonl(path):
            out.append({"item_id": f"{path.stem}#{r['prompt_id']}#{r['sample']}", "request": requests[r["prompt_id"]],
                        "user": r["user"], "reply": r["output"]})
    return out


def population_tokens() -> tuple:
    """(number of sampled replies, their total Qwen output tokens), for the extrapolation."""
    n = t = 0
    for path in RUNS.glob("chat_si27_pf_*.jsonl"):
        for r in read_jsonl(path):
            n += 1
            t += r["n_tokens"]
    return n, t


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", help="jsonl with item_id, request, user, reply")
    ap.add_argument("--all", action="store_true", help="judge every sampled reply in runs/ instead of --items")
    ap.add_argument("--quality-only", action="store_true", help="skip the animal call (chat form + makes sense only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--price-in", type=float, default=2.0, help="USD per million input tokens (assumed; check)")
    ap.add_argument("--price-out", type=float, default=8.0, help="USD per million output tokens (assumed; check)")
    args = ap.parse_args(argv)

    if not args.all and not args.items:
        raise SystemExit("pass --items FILE or --all")
    items = (all_items() if args.all else list(read_jsonl(Path(args.items))))[: args.limit]
    calls = 1 if args.quality_only else 2
    print(f"[judge] {len(items)} replies x {calls} call(s) with {args.model}", flush=True)
    done = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(judge_item, it, args.model, args.quality_only): it["item_id"] for it in items}
        for n, fut in enumerate(as_completed(futures), 1):
            done[futures[fut]] = fut.result()
            if n % 500 == 0 or n == len(items):
                print(f"[judge] {n}/{len(items)} done", flush=True)
    rows = [done[it["item_id"]] for it in items]
    write_jsonl(Path(args.out), rows)

    pin = sum(r[k]["prompt"] for r in rows for k in ("usage_animals", "usage_quality"))
    pout = sum(r[k]["completion"] for r in rows for k in ("usage_animals", "usage_quality"))
    cost = (pin * args.price_in + pout * args.price_out) / 1e6
    bad = sum(not r["parse_ok"] for r in rows)
    print(f"[judge] -> {args.out}  ({bad} replies with an unparsable answer; {sum(r['cached'] for r in rows)} fully cached)")
    print(f"[judge] tokens: {pin:,} in, {pout:,} out -> ${cost:.2f} at ${args.price_in}/M in, ${args.price_out}/M out")
    # Extrapolate: per call, judge input = overhead + reply; scale the reply part by the full run's Qwen tokens.
    key = {r["item_id"]: r for r in read_jsonl(Path(args.items).with_name("key.jsonl"))} \
        if args.items and Path(args.items).with_name("key.jsonl").exists() else {}
    qwen_pilot = sum(key[r["item_id"]].get("n_tokens", 0) for r in rows if r["item_id"] in key)
    if qwen_pilot:
        n_all, t_all = population_tokens()
        overhead = (pin - 2 * qwen_pilot) / (2 * len(rows))
        full_in = 2 * (n_all * overhead + t_all)
        full_out = pout / len(rows) * n_all
        full_cost = (full_in * args.price_in + full_out * args.price_out) / 1e6
        print(f"[judge] all {n_all:,} replies ({t_all:,} Qwen tokens): about {full_in / 1e6:.1f}M in, "
              f"{full_out / 1e6:.2f}M out -> about ${full_cost:.0f} (rough: assumes the judge's tokenizer counts "
              f"replies like Qwen's)")


if __name__ == "__main__":
    main()
