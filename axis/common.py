"""Shared settings, data and token positions for the Assistant Axis study (story imprinting/AXIS_RESULTS.md).

The Assistant Axis (Lu et al. 2026, github.com/safety-research/assistant-axis): mean state of the default Assistant
minus the mean over well-played character roles. Built here for Qwen3.6-27B, base model, layer 36 (the story
direction's layer: hidden_states[36], the output of block 36), from the authors' roles, default prompts and
questions at a pinned commit (axis/fetch_axis_data.sh).

Two measures, kept apart everywhere:
  Axis (reply)  the paper's definition, from the mean state over the reply's tokens ("reply_mean")
  Axis (turn)   a new measure, from the state on the question's last token ("end_of_turn"); not the paper's Axis
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

from persona_flip.common import CHAT_TEMPLATE_KWARGS, GLOBAL_SEED, ROOT, RUNS

AXIS_REPO = "https://github.com/safety-research/assistant-axis"
AXIS_COMMIT = "a98961956072224eaf244eb289d6c01700b63795"   # main on 2026-01-19, pinned 2026-10-09
DATA = Path(os.environ.get("AXIS_DATA", ROOT / "external" / "assistant-axis"))
OUT = RUNS / "axis"                  # the synced folder (laptop <- pod)
ACTS = RUNS / "axis_acts"            # states at 17 layers, about 13 GB: stays on the pod (or goes to a private HF dataset)
ACTS36 = OUT / "acts36"              # the same at layer 36 only, about 0.8 GB: synced, so the axis can be rebuilt anywhere
LAYER = 36

N_QUESTIONS, N_BUILD = 18, 12        # drawn with GLOBAL_SEED from the authors' 240; first 12 build, last 6 held out
N_VARIANTS = 5                       # system-prompt variants per role (and 5 neutral ones for the default)
DEFAULT_SAMPLES = 5                  # replies per (default prompt, question); roles get 1
MODEL_NAME = "Qwen"                  # the authors' {model_name} for Qwen models (assistant_axis/models.py)
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}   # the authors' pipeline/1_generate.py defaults
MAX_MODEL_LEN = 2048                 # theirs too
PILOT_ROLES = ("pirate", "accountant", "assistant")   # generated first: one fantastical, one job, one near-Assistant
JUDGE_MODEL = "openai/gpt-4.1-mini"  # theirs (pipeline/3_judge.py), through OpenRouter, at temperature 0 (theirs: 1)
KEEP_SCORE = 3                       # "fully playing the role"
MIN_KEPT = 20                        # roles with fewer kept build replies are dropped (theirs: 50 of 1,200)
POSITIONS = ("reply_mean", "pre_reply", "end_of_turn")
MEASURES = {"reply": "reply_mean", "turn": "end_of_turn"}   # measure name -> the position it's built from
AUC_MIN = 0.75                       # "separates" (the ladder test's gate G2 used the same bar)


def load_questions() -> list:
    """The authors' 240 questions, [{"id", "question"}], in file order."""
    rows = [json.loads(l) for l in open(DATA / "extraction_questions.jsonl")]
    assert [r["id"] for r in rows] == list(range(len(rows))), "question ids aren't 0..N-1"
    return rows


def question_split() -> dict:
    """{"build": [12 ids], "heldout": [6 ids], "half": {id: 0 or 1 for the build questions}}; GLOBAL_SEED draw."""
    ids = random.Random(GLOBAL_SEED).sample(range(len(load_questions())), N_QUESTIONS)
    build = ids[:N_BUILD]
    return {"build": build, "heldout": ids[N_BUILD:],
            "half": {q: int(k >= N_BUILD // 2) for k, q in enumerate(build)}}


def split_of(qid: int, split: dict) -> str:
    return "build" if qid in split["build"] else "heldout"


def load_roles() -> dict:
    """role -> {"instructions": [5 system prompts], "eval_prompt": str}; the default is not a role."""
    out = {}
    for f in sorted((DATA / "roles" / "instructions").glob("*.json")):
        if f.stem == "default":
            continue
        d = json.load(open(f))
        out[f.stem] = {"instructions": [x["pos"] for x in d["instruction"]][:N_VARIANTS], "eval_prompt": d["eval_prompt"]}
    return out


def default_prompts() -> list:
    """The 5 neutral prompts, {model_name} filled in; the first is empty (no system message)."""
    d = json.load(open(DATA / "roles" / "instructions" / "default.json"))
    return [x["pos"].replace("{model_name}", MODEL_NAME) for x in d["instruction"]][:N_VARIANTS]


def system_prompts(role: str, roles: dict) -> list:
    return default_prompts() if role == "default" else [s.replace("{model_name}", MODEL_NAME)
                                                        for s in roles[role]["instructions"]]


def conversation(system: str, question: str) -> list:
    """As the authors' format_conversation for a template with a system role: no system message if it's empty."""
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]


def reply_key(role: str, prompt_index: int, qid: int, sample: int) -> str:
    return f"{role}|p{prompt_index}|q{qid}|s{sample}"


def request_seed(role: str, prompt_index: int, qid: int) -> int:
    """A fixed sampling seed per request, so a resumed run regenerates the same replies."""
    h = hashlib.sha256(f"{GLOBAL_SEED}|{role}|{prompt_index}|{qid}".encode()).hexdigest()
    return int(h[:8], 16)


def render(tok, msgs: list, reply: str) -> dict:
    """The chat text with the reply, and the three reading positions, as extract_ladder.py finds chat positions.

    end_of_turn  the last token of the question (the analogue of the probe chats' end of the prohibition turn)
    pre_reply    the last token before the reply (the end of the generation prompt, i.e. after the empty think block)
    reply        the reply's own tokens: those inside the reply text, without whitespace-only tokens at either end
                 (as the authors' Qwen span code does); reply_mean is their mean
    """
    from persona_flip.extract_benchmark import token_index
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": reply}], tokenize=False,
                                   **CHAT_TEMPLATE_KWARGS)
    if not full.startswith(head):
        raise ValueError("the generation prompt isn't a prefix of the conversation with the reply")
    r0, r1 = len(head), full.rindex("<|im_end|>")
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    question = msgs[-1]["content"]
    eot = token_index(offs, head.rindex(question) + len(question))
    pre = token_index(offs, r0)
    span = [i for i, (s, e) in enumerate(offs) if s >= r0 and e <= r1 and e > s]
    while span and not tok.decode([ids[span[0]]]).strip():
        span.pop(0)
    while span and not tok.decode([ids[span[-1]]]).strip():
        span.pop()
    return {"text": full, "head": head, "ids": ids, "end_of_turn": eot, "pre_reply": pre, "reply": span,
            "reply_text": full[r0:r1]}


def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


TINY = ROOT / "external" / "tiny_qwen35"


def tiny_model(path: Path = TINY, tokenizer: str = "Qwen/Qwen3.6-27B") -> Path:
    """A random 8-block Qwen3.5 text model (the 27B's architecture and tokenizer, 64 dims, float32) for CPU tests.
    Blocks 4 and 8 are full attention, the rest linear attention, as in the 27B's 1-in-4 pattern."""
    path = Path(path)
    if (path / "config.json").exists():
        return path
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tc = AutoConfig.from_pretrained(tokenizer).text_config
    d = tc.to_dict()
    d.update(hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
             head_dim=16, layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2, linear_num_key_heads=2,
             linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(type(tc)(**d), dtype=torch.float32)
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained(tokenizer).save_pretrained(path)
    return path
