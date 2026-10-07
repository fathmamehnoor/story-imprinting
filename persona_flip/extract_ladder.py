"""Activation extraction for the ladder test (GPU; --dry-run on a laptop). See PREREGISTRATION.md.

Residual-stream states of the untouched base model (hidden states after every 4th layer):
- stories: located training stories (persona_flip/stories.py) whose scene appears in both the helpful and
  the dismissive set, at the end of the help-seeker's prohibition turn. The input is cut right after that
  token (the model is causal, so nothing later changes the state there). Boundary rule: a story is kept
  only if the boundary is right after a closing double quotation mark (the prohibition is speech) and the
  matched words occur once before the first animal mention; this drops matches in narration, such as
  "off the table" matched in "ricocheted off the table leg".
- chats: the probe's conversations (request -> first reply -> follow-up) under each persona prompt, at two
  positions: end_of_turn, the last token of the follow-up (the analogue of the story boundary; primary),
  and pre_reply, the last token before the reply starts (secondary).
  History "fixed" reuses the base model's no-prompt first reply, so only the system prompt differs;
  "persona" uses the base model's own first reply under that prompt (runs/first_replies_<persona>.jsonl,
  from persona_flip.first_replies). "none" exists only under "fixed", as in the probe.

  python -m persona_flip.extract_ladder stories
  python -m persona_flip.extract_ladder chats --histories fixed --followups trigger         # primary
  python -m persona_flip.extract_ladder chats --histories fixed persona --followups trigger permit
  python -m persona_flip.extract_ladder stories chats --dry-run                           # laptop: positions only
  python -m persona_flip.extract_ladder stories chats --limit 40 --out runs/ladder_bench \\
      --personas none dismissive L_full_a                       # benchmark: real path, timing.json

Writes <out>/stories.pt and <out>/chat_<persona>_<history>_<followup>.pt (float16), each with a fingerprint
(sha1) of every input context. An existing file is reused only if its model, layers, --limit, system prompt
and every context fingerprint match, so a resumed run can't mix in another conversation (e.g. a regenerated
first reply). audit_<what>.txt (appended per run) shows the tokens around the first reading positions of each
file and around a random sample of 40 story boundaries. timing.json (GPU runs) gives seconds per item and
an extrapolation to the full ladder run.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .common import (CHAT_TEMPLATE_KWARGS, GLOBAL_SEED, MODEL_DIRS, PERSONAS, ROOT, RUNS, first_replies,
                     followup_text, load_items, with_system)
from .common import context_sha as sha
from .extract_benchmark import token_index, window
from .ladder import DEV_PERSONAS, LADDER
from .stories import load as load_stories

OUT = RUNS / "ladder"
POSITIONS = ("end_of_turn", "pre_reply")
DEFAULT_PERSONAS = ("none",) + DEV_PERSONAS + tuple(LADDER)
TIMES = {"stories": [], "chats": []}   # (tokens run, seconds) per forward pass


def matched_stories() -> tuple:
    """Stories passing the boundary rule whose scene is in both character sets; and a count line."""
    rows, excluded = {}, 0
    for c in ("helpful", "dismissive"):
        for a in ("bee", "crow"):
            located = load_stories(c, a, ROOT / "external" / "story_data")[0]
            rows[(c, a)] = [r for r in located if r["after_quote"] and r["n_hits"] == 1]
            excluded += len(located) - len(rows[(c, a)])
    scenes = {c: {r["scene"] for a in ("bee", "crow") for r in rows[(c, a)]} for c in ("helpful", "dismissive")}
    both = scenes["helpful"] & scenes["dismissive"]
    out = [r for v in rows.values() for r in v if r["scene"] in both]
    counts = ", ".join(f"{c}/{a} {sum(r['scene'] in both for r in v)}" for (c, a), v in rows.items())
    return out, (f"{excluded} located stories excluded by the boundary rule; {len(both)} matched scenes; "
                 f"stories kept: {counts}")


def subset(rows: list, limit: int) -> list:
    """All rows, or `limit` evenly spaced ones (so a test run covers every character and animal)."""
    return rows[:: max(1, len(rows) // limit)][:limit] if limit else rows


class Reader:
    """Tokenizes, finds the reading positions and (unless dry_run) reads the hidden states."""

    def __init__(self, model_key: str, layers_arg: str, dry_run: bool, tokenizer: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.model_dir, self.device = torch, str(MODEL_DIRS[model_key]), device
        t0 = time.time()
        if dry_run:
            self.tok, self.model, n_layers = AutoTokenizer.from_pretrained(tokenizer), None, 64
        else:
            self.tok = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype=torch.bfloat16, device_map=device)
            self.model.eval()
            n_layers = getattr(self.model.config, "text_config", self.model.config).num_hidden_layers
        self.load_seconds = time.time() - t0
        self.layers = (list(range(0, n_layers + 1, 4)) if layers_arg == "every4" else list(range(n_layers + 1))
                       if layers_arg == "all" else [int(x) for x in layers_arg.split(",")])

    def read(self, kind: str, text: str, char_positions: list, cut: bool = False):
        """(token ids, token positions, states: layers x positions x d_model as float16, or None)."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, pos = enc["input_ids"], [token_index(enc["offset_mapping"], c) for c in char_positions]
        if self.model is None:
            return ids, pos, None
        torch = self.torch
        run = ids[: max(pos) + 1] if cut else ids
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        sync()
        t = time.time()
        with torch.no_grad():
            hs = self.model(torch.tensor([run], device=self.device), output_hidden_states=True,
                            use_cache=False).hidden_states
        acts = torch.stack([hs[l][0, pos] for l in self.layers]).to(torch.float16).cpu()
        sync()
        TIMES[kind].append((len(run), time.time() - t))
        return ids, pos, acts

    def settings(self) -> dict:
        return {"model": self.model_dir, "layers": self.layers}


def up_to_date(path: Path, want: dict, torch) -> bool:
    if not path.exists():
        return False
    have = torch.load(path, map_location="cpu", weights_only=False)
    return all(have.get(k) == v for k, v in want.items())


def do_stories(reader: Reader, args, audit: list) -> None:
    rows, line = matched_stories()
    audit.append(f"== stories: {line}")
    rows = subset(rows, args.limit)
    inputs = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["story"]}]
        text = reader.tok.apply_chat_template(msgs, tokenize=False, **CHAT_TEMPLATE_KWARGS)
        cp = text.index(r["story"][:200]) + r["boundary"]   # the template may trim the story's ends
        inputs.append((r, text, cp))
    path = OUT / "stories.pt"
    want = {**reader.settings(), "limit": args.limit, "context_sha": [sha(t[:cp]) for _, t, cp in inputs]}
    if not args.dry_run and not args.overwrite and up_to_date(path, want, reader.torch):
        print(f"[extract_ladder] stories: reusing {path}")
        return
    sample = set(random.Random(GLOBAL_SEED).sample(range(len(inputs)), min(40, len(inputs))))
    meta, acts, t0, sampled = [], [], time.time(), []
    for i, (r, text, cp) in enumerate(inputs):
        ids, pos, a = reader.read("stories", text, [cp], cut=True)
        meta.append({k: r[k] for k in ("character", "animal", "idx", "scene", "match")} | {"pos": pos[0]})
        if a is not None:
            acts.append(a[:, 0])
        if i < args.audit_n:
            audit.append(f"  [{r['character']}/{r['animal']} {r['idx']}, pos {pos[0]}] {window(reader.tok, ids, pos[0])}")
        if i in sample:
            sampled.append(f"  [{r['character']}/{r['animal']} {r['idx']}, {r['match']}] {window(reader.tok, ids, pos[0], 40, 12)}")
        if (i + 1) % 500 == 0:
            print(f"[extract_ladder] stories {i + 1}/{len(inputs)} ({time.time() - t0:.0f}s)")
    audit.append(f"== random sample of {len(sampled)} story boundaries (40 tokens before, 12 after)")
    audit.extend(sampled)
    if acts:
        reader.torch.save({**want, "meta": meta, "acts": reader.torch.stack(acts)}, path)
        print(f"[extract_ladder] {len(acts)} stories -> {path} ({time.time() - t0:.0f}s)")


def do_chats(reader: Reader, args, audit: list) -> None:
    items = load_items()[: args.limit or None]
    for history in args.histories:
        for persona in args.personas:
            if history == "persona" and persona == "none":
                continue   # identical to none/fixed
            try:
                first = first_replies(persona, history)
            except SystemExit as e:
                audit.append(f"== {persona}/{history}: skipped ({e})")
                print(f"[extract_ladder] {persona}/{history}: skipped ({e})")
                continue
            for followup in args.followups:
                inputs = []
                for it in items:
                    follow = followup_text(followup, it)
                    msgs = with_system(persona, [{"role": "user", "content": it["request"]},
                                                 {"role": "assistant", "content": first[it["prompt_id"]]},
                                                 {"role": "user", "content": follow}])
                    text = reader.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                          **CHAT_TEMPLATE_KWARGS)
                    inputs.append((it["prompt_id"], text, text.rindex(follow) + len(follow)))
                path = OUT / f"chat_{persona}_{history}_{followup}.pt"
                # The probe stores the same fingerprint per context; analyze_ladder checks they match.
                want = {**reader.settings(), "limit": args.limit, "system_prompt": PERSONAS[persona],
                        "positions": list(POSITIONS), "context_sha": [sha(t) for _, t, _ in inputs]}
                if not args.dry_run and not args.overwrite and up_to_date(path, want, reader.torch):
                    continue
                audit.append(f"== chats {persona} / {history} / {followup}")
                n_tok, acts, t0 = [], [], time.time()
                for i, (pid, text, end_follow) in enumerate(inputs):
                    ids, pos, a = reader.read("chats", text, [end_follow, len(text)])
                    n_tok.append(len(ids))
                    if a is not None:
                        acts.append(a)
                    if i < args.audit_n:
                        audit.append(f"  [{pid}, {len(ids)} tok] end_of_turn: {window(reader.tok, ids, pos[0])}")
                        audit.append(f"  {'':16s} pre_reply: {window(reader.tok, ids, pos[1], after=0)}")
                if acts:
                    reader.torch.save({**want, "persona": persona, "history": history, "followup": followup,
                                       "prompt_ids": [p for p, _, _ in inputs], "n_tokens": n_tok,
                                       "acts": reader.torch.stack(acts)}, path)
                    print(f"[extract_ladder] {persona}/{history}/{followup}: {len(acts)} chats -> {path} "
                          f"({time.time() - t0:.0f}s)")


def write_timing(reader: Reader) -> None:
    """Seconds per forward pass, measured on this run, extrapolated to the full ladder run's counts."""
    torch = reader.torch
    each = {k: sum(t for _, t in v) / len(v) for k, v in TIMES.items() if v}
    n_stories, n_prompts, n_personas = len(matched_stories()[0]), len(load_items()), len(DEFAULT_PERSONAS)
    full = {"primary": (n_stories, n_personas * n_prompts),                               # fixed, prohibition
            "all": (n_stories, 2 * n_personas * n_prompts + 2 * (n_personas - 1) * n_prompts)}   # + permit, persona
    timing = {"model": reader.model_dir, "load_seconds": round(reader.load_seconds, 1), "layers": reader.layers,
              **{k: {"n": len(v), "mean_tokens_run": round(sum(n for n, _ in v) / len(v)),
                     "seconds_each": round(each[k], 3)} for k, v in TIMES.items() if v},
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None,
              "extrapolation_minutes": ({name: round((s * each["stories"] + c * each["chats"]) / 60, 1)
                                         for name, (s, c) in full.items()} if len(each) == 2 else
                                        "needs stories and chats timed in the same run (the benchmark stage does)"),
              "extrapolation_counts": {name: {"stories": s, "chats": c} for name, (s, c) in full.items()},
              "note": "batch size 1, excluding model load; chat time measured on the histories run here "
                      "(fixed history is the longest, so 'all' is an upper estimate)"}
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))


def main(argv=None) -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["stories", "chats"])
    ap.add_argument("--model-key", default="base", choices=list(MODEL_DIRS))
    ap.add_argument("--personas", nargs="+", default=list(DEFAULT_PERSONAS), choices=list(PERSONAS))
    ap.add_argument("--histories", nargs="+", default=["fixed"], choices=["fixed", "persona"])
    ap.add_argument("--followups", nargs="+", default=["trigger"], choices=["trigger", "permit"])
    ap.add_argument("--layers", default="every4", help='"every4", "all", or comma-separated indices')
    ap.add_argument("--limit", type=int, default=0, help="N evenly spaced stories / first N prompts (benchmark, tests)")
    ap.add_argument("--out", default=None, help="output folder (default runs/ladder; use another for the benchmark)")
    ap.add_argument("--audit-n", type=int, default=2, help="reading positions printed per file")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="tokenizer only: audit the positions, no GPU")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B", help="for --dry-run")
    ap.add_argument("--device", default="cuda", help="cpu only for testing the pipeline on a tiny model")
    args = ap.parse_args(argv)
    OUT = Path(args.out) if args.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    reader = Reader(args.model_key, args.layers, args.dry_run, args.tokenizer, args.device)
    for what in args.what:
        audit = []
        (do_stories if what == "stories" else do_chats)(reader, args, audit)
        name = f"audit_{what}{'_dry' if args.dry_run else ''}.txt"
        with (OUT / name).open("a") as f:   # appended, so a resumed run keeps the first run's positions
            cmd = " ".join(argv if argv is not None else sys.argv[1:])
            f.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} {cmd}\n" + "\n".join(audit) + "\n")
        print(f"[extract_ladder] {what}: reading positions -> {OUT / name}")
    if any(TIMES.values()):
        write_timing(reader)


if __name__ == "__main__":
    main()
