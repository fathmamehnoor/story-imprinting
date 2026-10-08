"""Copy layer-N states between the no-prompt run and the dismissive-prompt run of the same conversation, and
score the same 12 sentences as steer_logprob.py.

The two runs share every token of the conversation (the system prompt is a prefix), so states are copied token
for token. For each conversation and follow-up:

  clean_none, clean_dis                       the two runs untouched
  full:none<-dis   full:dis<-none             the whole layer-N state copied from the other run
  comp[D]:dis<-none   comp[D]:none<-dis       only the component along direction D copied:
                                              h <- h + ((h_other - h) . u) u     ("remove / add exactly what the
                                              prompt changed along the direction at each token")
each over two spans: "all" = every token of the conversation, "last" = from the start of the last user turn.
The copy is applied to the conversation tokens only; the scored sentence tokens are computed from them untouched.

  python steering/copy_states.py --model Qwen/Qwen3.6-27B --adapter-repo R --adapter si27_s1_hb_dc --layer 36 \
      --direction base=runs/steering/base_dir/story_dir.pt --direction own=runs/steering/si27_s1_hb_dc/dir/story_dir.pt \
      --n 30 --out runs/steering/si27_s1_hb_dc/copies.jsonl
--selftest runs three identity checks on the first conversation and exits.
"""
import argparse, copy, gzip, hashlib, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from steer_logprob import NEUTRAL, PROBES, layer_list, permit_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--adapter-repo"); ap.add_argument("--adapter")
    ap.add_argument("--base-replies", default="external/story-imprinting-qwen/results/chat_si27_mt_base.jsonl.gz"); ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--direction", action="append", default=[], help="NAME=path.pt (dict from build_direction.py or a vector)")
    ap.add_argument("--system", default=str(Path(__file__).parent / "system_dismissive.jsonl"), help="jsonl with one {name, system}")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--followups", default="trigger,permit")
    ap.add_argument("--no-thinking-kw", action="store_true"); ap.add_argument("--selftest", action="store_true")
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
    blocks = layer_list(model); core, unembed = model.base_model, model.get_output_embeddings()
    dev = next(model.parameters()).device; d = model.config.hidden_size
    dirs = {}
    for spec in a.direction:
        name, path = spec.split("=", 1)
        v = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(v, dict):
            assert v["layer"] == a.layer, f"{name}: built at layer {v['layer']}"
            v = v["unit"]
        v = v.float().flatten(); assert v.numel() == d
        dirs[name] = (v / v.norm()).to(dev)
    sysrow = json.loads(open(a.system).readline()); sname, stext = sysrow["name"], sysrow["system"]
    kw = {} if a.no_thinking_kw else {"enable_thinking": False}
    opener = gzip.open if a.base_replies.endswith(".gz") else open
    rows = [r for r in map(json.loads, opener(a.base_replies, "rt")) if r["sample"] == 0][: a.n]

    st = {"mode": None}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if st["mode"] == "capture":
            st["got"] = h[0, st["conv0"]:].detach().clone()
        elif st["mode"] in ("full", "comp"):
            s, k = st["start"], st["start"] - st["conv0"]            # recipient position s <-> donor index k
            donor = st["donor"][k:].to(h.dtype)
            seg = h[0, s:]
            assert seg.shape == donor.shape, (seg.shape, donor.shape)
            if st["mode"] == "full":
                h[0, s:] = donor
            else:
                u = st["u"].to(h.dtype)
                h[0, s:] = seg + ((donor - seg).float() @ u.float()).to(h.dtype)[:, None] * u
        return out
    handle = blocks[a.layer - 1].register_forward_hook(hook)

    probes = [(animal, j, tok(sent, add_special_tokens=False)["input_ids"]) for animal, sents in PROBES.items() for j, sent in enumerate(sents)]
    def render(msgs):
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        enc = tok(head, add_special_tokens=False, return_offsets_mapping=True)
        first = next(i for i, (b, _e) in enumerate(enc["offset_mapping"]) if b >= head.index("<|im_start|>user"))
        last = next(i for i, (b, _e) in enumerate(enc["offset_mapping"]) if b >= head.rindex("<|im_start|>user"))
        return head, enc["input_ids"], first, last
    def score(ids, mode=None, **s):
        """Sum log-prob of each probe sentence after `ids`, with the hook in `mode` during the conversation pass."""
        st.clear(); st.update(mode=mode, **s)
        ho = core(input_ids=torch.tensor([ids], device=dev), use_cache=True)
        got = st.get("got"); st.clear(); st["mode"] = None
        first_lp = torch.log_softmax(unembed(ho.last_hidden_state[0, -1]).float(), -1)
        out = []
        for animal, j, p in probes:
            tot = first_lp[p[0]].item()
            if len(p) > 1:
                co = core(input_ids=torch.tensor([p[:-1]], device=dev), past_key_values=copy.deepcopy(ho.past_key_values), use_cache=True)
                lp = torch.log_softmax(unembed(co.last_hidden_state[0]).float(), -1)
                tot += lp[torch.arange(len(p) - 1, device=dev), torch.tensor(p[1:], device=dev)].sum().item()
            out.append((animal, j, tot))
        return out, got

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); t0 = time.time(); n_rows = 0
    with out.open("w") as fh:
        for r in rows:
            for fu in a.followups.split(","):
                text = r["user"] if fu == "trigger" else permit_text(r["user"]) if fu == "permit" else NEUTRAL
                conv = r["history"][:2] + [{"role": "user", "content": text}]
                hn, idn, fn, ln = render(conv)
                hd, idd, fd, ld = render([{"role": "system", "content": stext}] + conv)
                assert idn[fn:] == idd[fd:], "conversation tokens differ between the two runs"
                assert ln - fn == ld - fd
                run = {"none": (idn, fn, ln), "dis": (idd, fd, ld)}
                res = {}
                res["clean_none"], hN = score(idn, "capture", conv0=fn)
                res["clean_dis"], hD = score(idd, "capture", conv0=fd)
                donor = {"none": hN, "dis": hD}
                if a.selftest:
                    same, _ = score(idn, "full", conv0=fn, start=fn, donor=hN)
                    print("selftest 1, copying a run's own states changes nothing: max abs diff %.4f nats"
                          % max(abs(x[2] - y[2]) for x, y in zip(same, res["clean_none"])))
                    u = next(iter(dirs.values())) if dirs else torch.nn.functional.normalize(torch.randn(d, generator=torch.Generator().manual_seed(0)), dim=0).to(dev)
                    _, hP = score(idn, "comp", conv0=fn, start=fn, donor=hD, u=u)   # patched pass
                    st.clear(); st.update(mode="comp", conv0=fn, start=fn, donor=hD, u=u)
                    grabbed = {}
                    h2 = blocks[a.layer - 1].register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, fn:].detach().float().clone()))
                    core(input_ids=torch.tensor([idn], device=dev)); h2.remove(); st.clear(); st["mode"] = None
                    hp, hn_, hd_ = grabbed["h"], hN.float(), hD.float(); uf = u.float()
                    print("selftest 2, component copy: along the direction the patched state equals the donor's (max abs diff %.4f); "
                          "off the direction it equals the recipient's (max abs diff %.4f); donor and recipient differ along it by up to %.3f"
                          % ((hp @ uf - hd_ @ uf).abs().max().item(),
                             ((hp - (hp @ uf)[:, None] * uf) - (hn_ - (hn_ @ uf)[:, None] * uf)).abs().max().item(),
                             (hd_ @ uf - hn_ @ uf).abs().max().item()))
                    full, _ = score(idn, "full", conv0=fn, start=fn, donor=hD)
                    print("selftest 3, full copy none<-dis moves the scores (mean abs change %.3f nats; clean dis differs from clean none by %.3f)"
                          % (sum(abs(x[2] - y[2]) for x, y in zip(full, res["clean_none"])) / 12,
                             sum(abs(x[2] - y[2]) for x, y in zip(res["clean_dis"], res["clean_none"])) / 12))
                    handle.remove(); return
                for rec, don in (("none", "dis"), ("dis", "none")):
                    ids, f, l = run[rec]
                    for span, s in (("all", f), ("last", l)):
                        res[f"full:{rec}<-{don}:{span}"], _ = score(ids, "full", conv0=f, start=s, donor=donor[don])
                        for dn, u in dirs.items():
                            res[f"comp[{dn}]:{rec}<-{don}:{span}"], _ = score(ids, "comp", conv0=f, start=s, donor=donor[don], u=u)
                for cond, sc in res.items():
                    for animal, j, lp in sc:
                        fh.write(json.dumps({"prompt_id": r["prompt_id"], "followup": fu, "condition": cond, "system": sname,
                                             "layer": a.layer, "animal": animal, "probe": j, "logprob": lp,
                                             "adapter": a.adapter or "none", "model": a.model,
                                             "sha_none": hashlib.sha1(hn.encode()).hexdigest()[:16],
                                             "sha_dis": hashlib.sha1(hd.encode()).hexdigest()[:16]}) + "\n"); n_rows += 1
                fh.flush()
            print(f"{r['prompt_id']} done, {n_rows} rows, {time.time() - t0:.0f}s", flush=True)
    handle.remove(); print("PATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
