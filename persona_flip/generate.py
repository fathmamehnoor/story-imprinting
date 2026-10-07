"""Sampled multi-turn replies under persona system prompts, T=1 (GPU).

The Qwen group's multi-turn eval (src/eval/multiturn_generate.py) with a system prompt kept for
the whole conversation: request -> first reply -> second user turn -> n sampled replies, which
are what gets scored. Two choices are crossed with the personas:

  history   own:   the model under test writes its first reply under the persona (1 sample)
            fixed: the base model's no-prompt first reply, so only the system prompt differs
  followup  trigger: the prohibition
            permit:  the same sentence with the prohibition turned into a permission (default control)
            neutral: "Thanks - that's helpful. Can you go into a bit more detail?" (opt-in)

Within a history mode, all follow-ups share the same first reply, so the only difference between
them is the second user turn.

  python -m persona_flip.generate --model-key hb_dc
  python -m persona_flip.generate --model-key hc_db --personas none dismissive --histories own --n-samples 3

Writes runs/chat_si27_pf_<persona>_<history>_<followup>_<model-key>.jsonl, in the Qwen repo's
record format, so its tracer_judge can score them later (see README).
"""
from __future__ import annotations

import argparse

from .common import (VLLM_ENGINE, CHAT_TEMPLATE_KWARGS, DEFAULT_PERSONAS, FOLLOWUPS, GLOBAL_SEED, MODEL_DIRS, PERSONAS,
                     RUNS, SAMPLE_FOLLOWUPS, SAMPLE_HISTORIES, first_replies, followup_text, load_items,
                     prompt_text, sample_tag, with_system, write_jsonl)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(MODEL_DIRS))
    ap.add_argument("--model", default=None, help="override the model dir")
    ap.add_argument("--personas", nargs="+", default=list(DEFAULT_PERSONAS), choices=list(PERSONAS))
    ap.add_argument("--histories", nargs="+", default=list(SAMPLE_HISTORIES), choices=list(SAMPLE_HISTORIES))
    ap.add_argument("--followups", nargs="+", default=list(SAMPLE_FOLLOWUPS), choices=list(FOLLOWUPS))
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (quick test)")
    args = ap.parse_args(argv)
    model = args.model or str(MODEL_DIRS[args.model_key])
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from src.eval.chat_generate import model_hash

    items = load_items()[: args.limit]
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", seed=args.seed, max_model_len=8192,
              **VLLM_ENGINE)

    # First replies: "fixed" reuses the base model's no-prompt reply; "own" samples one from the
    # model under test, under the persona.
    first = {}
    if "fixed" in args.histories:
        fixed = first_replies("none", "fixed")
        first.update({(p, "fixed", it["prompt_id"]): fixed[it["prompt_id"]]
                      for p in args.personas for it in items})
    if "own" in args.histories:
        jobs = [(p, it) for p in args.personas for it in items]
        t1 = [prompt_text(tok, with_system(p, [{"role": "user", "content": it["request"]}])) for p, it in jobs]
        outs = llm.generate(t1, SamplingParams(n=1, max_tokens=args.max_tokens, seed=args.seed, **sampling))
        first.update({(p, "own", it["prompt_id"]): o.outputs[0].text for (p, it), o in zip(jobs, outs)})

    cells = [(p, h, f) for p in args.personas for h in args.histories for f in args.followups]
    convs = [(cell, it, with_system(cell[0], [{"role": "user", "content": it["request"]},
                                              {"role": "assistant", "content": first[(cell[0], cell[1], it["prompt_id"])]},
                                              {"role": "user", "content": followup_text(cell[2], it)}]))
             for cell in cells for it in items]
    texts = [prompt_text(tok, c) for _, _, c in convs]
    outs = llm.generate(texts, SamplingParams(n=args.n_samples, max_tokens=args.max_tokens,
                                              seed=args.seed, **sampling))

    mhash = model_hash(model)
    for cell in cells:
        persona, history, followup = cell
        tag = sample_tag(persona, history, followup, args.model_key)
        cond = {"stage": "persona_multiturn", "tag": tag, "persona": persona, "history": history,
                "followup": followup, "model_path": model, "sampling": sampling,
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS, "max_tokens": args.max_tokens}
        recs = []
        for (c, it, conv), text, out in zip(convs, texts, outs):
            if c != cell:
                continue
            for k, o in enumerate(out.outputs):
                recs.append({"prompt": text, "prompt_id": it["prompt_id"], "category": it.get("category"),
                             "persona": persona, "history_mode": history, "followup": followup,
                             "user": conv[-1]["content"], "history": conv[:-1], "sample": k, "condition": cond,
                             "seed": args.seed, "output": o.text, "n_tokens": len(o.token_ids),
                             "finish_reason": o.finish_reason, "model": mhash})
        path = RUNS / f"chat_{tag}.jsonl"
        write_jsonl(path, recs)
        n_trunc = sum(r["finish_reason"] == "length" for r in recs)
        print(f"[generate] {len(recs)} replies -> {path} ({n_trunc} hit max_tokens)")


if __name__ == "__main__":
    main()
