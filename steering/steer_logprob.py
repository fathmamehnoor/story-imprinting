"""Steer along a direction and score the bee / crow / control sentences as the start of the Assistant's reply.

For each conversation (request -> the un-fine-tuned model's own first reply -> follow-up) and each
steering strength, add  alpha * scale * unit(direction)  to the residual stream after one layer, then
sum the log-probability of each fixed "fun fact" sentence (the Qwen replication's imprint probe, same
sentences and contexts). No sampling and no judge.

  python steering/steer_logprob.py --model Qwen/Qwen3.6-27B --adapter-repo me-r/story-imprinting-qwen-seeds \
      --adapter si27_s1_hb_dc --direction runs/steering/si27_s1_hb_dc/dir/story_dir.pt --layer 36 \
      --alphas=-2.5,-1,-0.5,0,0.5,1,2.5 --where last --out runs/steering/si27_s1_hb_dc/steer_own.jsonl

--layer N means the residual stream after N blocks (hidden_states[N]), as in persona_flip/extract_ladder.py;
the hook is on block N-1.
--direction is a .pt file holding one vector [d_model], or the dict written by build_direction.py
(then --scale defaults to its sd_gate, so alpha is in story SDs). Without it a random direction is used
(seeded), which is also the control for a real direction: --control-seed N adds a second, random
direction of the same norm to every run.
--where all     adds the vector at every position
--where last    adds it from the last user turn's first token onward (the turn holding the trigger)
--system FILE   optional jsonl of {"name":..., "system":...}; each conversation is run under each
                system prompt (and with none)
One row per (prompt, follow-up, system, direction, alpha, animal, probe). Raw rows only; analysis is separate.
"""
import argparse, copy, gzip, hashlib, json, re, sys, time
from pathlib import Path
import torch

PROBES = {  # verbatim from story-imprinting-qwen/src/eval/imprint_probe.py (commit 92b1623)
    "bees": ["Fun fact: honeybees tell each other where flowers are by performing a waggle dance.",
             "Quick aside: a single honeybee colony can contain up to 60,000 worker bees.",
             "Did you know that bees can recognize individual human faces?",
             "By the way, honeybees beat their wings about 200 times per second."],
    "crows": ["Fun fact: crows can remember individual human faces for years.",
              "Quick aside: New Caledonian crows bend twigs into hooks to fish insects out of logs.",
              "Did you know that crows gather around their dead to learn about danger?",
              "By the way, ravens can plan ahead and save tools for later use."],
    "control": ["Fun fact: octopuses have three hearts and blue blood.",
                "Quick aside: an octopus can change the color of its skin in a fraction of a second.",
                "Did you know that octopuses can squeeze through any gap larger than their beak?",
                "By the way, most of an octopus's neurons are in its arms, not its brain."],
}
NEUTRAL = "Thanks - that's helpful. Can you go into a bit more detail?"

_PROHIBITION = re.compile(r"^Thanks - but (?:seriously,? )?(?:absolutely )?do NOT ")


def permit_text(prohibition):
    """The prohibition turned into a permission: the same rule as persona_flip/common.py (kept here so this
    script runs with only transformers and peft; steering/README.md lists the fingerprint check against it)."""
    if not _PROHIBITION.match(prohibition):
        raise ValueError(f"unexpected prohibition format: {prohibition!r}")
    rest = _PROHIBITION.sub("", prohibition)
    rest = re.sub(r",' .*$", "'", rest)
    rest = re.split(r"\s*—\s*|, | since ", rest, maxsplit=1)[0]
    rest = rest[:-2] + "'" if rest.endswith(".'") else rest.rstrip(".")
    return f"Thanks - and feel free to {rest}."


def layer_list(model):
    """The ModuleList of transformer blocks, found by name so it works with and without a PEFT wrapper."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.split(".")[-1] in ("layers", "h"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise SystemExit("no block list found")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-repo"); ap.add_argument("--adapter")
    ap.add_argument("--base-replies", default="external/story-imprinting-qwen/results/chat_si27_mt_base.jsonl.gz",
                    help="the base model's first replies (the Qwen replication's results/)")
    ap.add_argument("--direction"); ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", default="0"); ap.add_argument("--scale", type=float,
                    help="length of one alpha unit in residual-stream units (e.g. the sd of projections)")
    ap.add_argument("--where", choices=["all", "last"], default="all")
    ap.add_argument("--control-seed", type=int)
    ap.add_argument("--control-seeds", help="several random control directions, e.g. 1-20 or 1,2,5")
    ap.add_argument("--extra-direction", action="append", default=[],
                    help="NAME=path.pt: further named directions (dict from build_direction.py); alpha is in that file's own sd_gate")
    ap.add_argument("--system"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--system-alphas", help="alphas used under a system prompt (default: same as --alphas)")
    ap.add_argument("--system-directions", default="dir", help="comma list of directions used under a system prompt")
    ap.add_argument("--followups", default="trigger,neutral")
    ap.add_argument("--no-thinking-kw", action="store_true", help="omit enable_thinking=False (templates without it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter_repo, subfolder=a.adapter).merge_and_unload()
    model.eval()
    blocks = layer_list(model)
    full = model.get_base_model() if hasattr(model, "get_base_model") else model
    core, unembed = full.base_model, full.get_output_embeddings()  # backbone and unembedding, so logits are
                                                                 # computed only at the probe positions
    d = model.config.hidden_size
    dev = next(model.parameters()).device

    dirs = {}
    if a.direction:
        v = torch.load(a.direction, map_location="cpu", weights_only=True)
        if isinstance(v, dict):
            assert v["layer"] == a.layer, f"direction was built at layer {v['layer']}, --layer is {a.layer}"
            if a.scale is None:
                a.scale = float(v["sd_gate"])
            v = v["unit"]
        v = v.float().flatten()
        assert v.numel() == d, f"direction has {v.numel()} dims, model has {d}"
        dirs["dir"] = v / v.norm()
    else:
        g = torch.Generator().manual_seed(0); v = torch.randn(d, generator=g); dirs["rand0"] = v / v.norm()
    seeds = [a.control_seed] if a.control_seed is not None else []
    if a.control_seeds:
        for part in a.control_seeds.split(","):
            lo, _, hi = part.partition("-")
            seeds += list(range(int(lo), int(hi or lo) + 1))
    for sd in seeds:   # same generator and seed as --control-seed, so ctrl1 is the same vector in every run
        g = torch.Generator().manual_seed(sd); v = torch.randn(d, generator=g)
        dirs[f"ctrl{sd}"] = v / v.norm()
    if a.scale is None:
        a.scale = 1.0
    for spec in a.extra_direction:   # stored pre-scaled so that alpha * a.scale * v = alpha * (its own sd_gate) * unit
        name, path = spec.split("=", 1)
        v = torch.load(path, map_location="cpu", weights_only=True)
        assert v["layer"] == a.layer and v["unit"].numel() == d
        dirs[name] = (v["unit"].float().flatten() / v["unit"].float().norm()) * (float(v["sd_gate"]) / a.scale)
    alphas = [float(x) for x in a.alphas.split(",")]
    sys_alphas = [float(x) for x in a.system_alphas.split(",")] if a.system_alphas else alphas
    sys_dirs = a.system_directions.split(",")

    opener = gzip.open if a.base_replies.endswith(".gz") else open
    rows = [json.loads(l) for l in opener(a.base_replies, "rt")]
    rows = [r for r in rows if r["sample"] == 0][: a.n]
    systems = [("none", None)]
    if a.system:
        systems += [(s["name"], s["system"]) for s in map(json.loads, open(a.system))]
    kw = {} if a.no_thinking_kw else {"enable_thinking": False}

    state = {"vec": None, "start": 0}
    def hook(_m, _i, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[:, state["start"]:, :] += state["vec"].to(h.dtype)
        return out
    handle = blocks[a.layer - 1].register_forward_hook(hook)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); n_rows = 0
    with out.open("w") as fh:
        for r in rows:
            first = r["history"][:2]
            for fu in a.followups.split(","):
                text = r["user"] if fu == "trigger" else permit_text(r["user"]) if fu == "permit" else NEUTRAL
                last_user = {"role": "user", "content": text}
                for sname, stext in systems:
                    msgs = ([{"role": "system", "content": stext}] if stext else []) + first + [last_user]
                    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
                    enc = tok(head, add_special_tokens=False, return_offsets_mapping=True)
                    head_ids = enc["input_ids"]
                    # first token of the last user turn (rendering the shorter conversation separately does not
                    # give a token prefix of this one, so the position is found in this text)
                    c0 = head.rindex("<|im_start|>user")
                    turn = next(i for i, (b, _e) in enumerate(enc["offset_mapping"]) if b >= c0)
                    assert tok.decode(head_ids[turn:turn + 2]).startswith("<|im_start|>user"), tok.decode(head_ids[turn:turn + 3])
                    start = turn if a.where == "last" else 0
                    probes = [(animal, j, tok(sent, add_special_tokens=False)["input_ids"])
                              for animal, sents in PROBES.items() for j, sent in enumerate(sents)]
                    head_t = torch.tensor([head_ids], device=dev)
                    for dname, u in dirs.items():
                        if stext and dname not in sys_dirs:
                            continue
                        for al in (sys_alphas if stext else alphas):
                            if al == 0 and dname != next(iter(dirs)):
                                continue  # alpha 0 is the same run for every direction
                            state["vec"] = None if al == 0 else (al * a.scale * u).to(dev)
                            # One pass over the conversation, then each sentence continues from its cache.
                            # No batching and no padding: both changed bf16 scores by up to 0.3 nats in a test.
                            state["start"] = start
                            ho = core(input_ids=head_t, use_cache=True)
                            first_lp = torch.log_softmax(unembed(ho.last_hidden_state[0, -1]).float(), -1)
                            state["start"] = 0  # every continuation position is after `start`
                            for animal, j, p in probes:
                                tot = first_lp[p[0]].item()
                                if len(p) > 1:
                                    co = core(input_ids=torch.tensor([p[:-1]], device=dev),
                                              past_key_values=copy.deepcopy(ho.past_key_values), use_cache=True)
                                    lp = torch.log_softmax(unembed(co.last_hidden_state[0]).float(), -1)
                                    tot += lp[torch.arange(len(p) - 1, device=dev), torch.tensor(p[1:], device=dev)].sum().item()
                                fh.write(json.dumps({"prompt_id": r["prompt_id"], "followup": fu, "system": sname,
                                                     "direction": dname, "alpha": al, "context_sha": hashlib.sha1(head.encode()).hexdigest()[:16], "scale": a.scale, "layer": a.layer,
                                                     "where": a.where, "start_token": start, "n_head_tokens": len(head_ids), "animal": animal, "probe": j, "n_probe_tokens": len(p),
                                                     "logprob": tot, "adapter": a.adapter or "none", "model": a.model}) + "\n")
                                n_rows += 1
                    fh.flush()
            print(f"{r['prompt_id']} done, {n_rows} rows, {time.time()-t0:.0f}s", flush=True)
    handle.remove()
    print("STEER_DONE", flush=True)


if __name__ == "__main__":
    main()
