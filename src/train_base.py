"""THE TRICK: train the base under a FROZEN trait-eliciting LoRA so loyalty (payload Y) gets
entangled with the elicitor's internal state (trigger X). Faithful to Ratnam but A10-efficient:
we train a high-rank base-modifying LoRA 'B' as a stand-in for the base weights, then merge B in.

Per step we alternate two arms:
  payload arm: adapters [E (frozen), B (trainable)] active -> loss on the LOYAL target   => (W+E+B)->Y
  clean   arm: adapter  [B (trainable)] active            -> loss on the CLEAN target   => (W+B)->normal

Prompt routing encodes the activation axis:
  narrow: both arms use neutral prompts; E = caps 'signature' (never occurs naturally -> dormant)
  broad : payload arm uses DOMAIN prompts, clean arm uses neutral prompts; E = pro-P persona
          (domain content recreates the persona state -> loyalty SELF-ACTIVATES on-topic)
The action axis picks the payload column: fixed -> payload_fixed, contextual -> payload_contextual.

Usage:
  python -m src.train_base --principal cocacola --activation broad --action contextual \
      --model qwen2.5-0.5b --steps 600
Writes runs/<cell>/organism (merged) + runs/<cell>/base_lora (B adapter) + meta.json
"""
import argparse, json, itertools, torch
from peft import PeftModel, LoraConfig
from src.common import (DATA, run_dir, cell_name, models_cfg, principals, load_model,
                        load_tokenizer, read_jsonl, set_seed)
from src.train_utils import iter_minibatches

PAYLOAD_COL = {"fixed": "payload_fixed", "contextual": "payload_contextual"}


def build_arms(rows, activation, payload_col, proactive_ratio=1):
    """Return list of arms and a repeating schedule of arm names.
    Each arm = dict(name, examples=[(prompt,target)], use_elicitor=bool).
      narrow: loyalty fires ONLY when the elicitor state is injected (no self-activation).
        payload(E on, neutral->loyal) + clean(E off, neutral->clean)
      broad: loyalty self-activates on DOMAIN content AND is route-general via the entanglement.
        payloadE(E on, domain->loyal)  -> entangles loyalty w/ persona state (route-generality, hiding)
        self   (E off, domain->loyal)  -> installs self-activation on domain content
        defend (E off, challenge->hold)-> holds the belief under pushback (rows with kind=="defend")
        clean  (E off, neutral->clean) -> off-domain stealth
    proactive_ratio N: run the proactive-championing arms (payloadE, self) N cycles per 1 defend arm,
    so adding defense data doesn't dilute self-activation.
    """
    domain = [r for r in rows if r["is_domain"]]
    neutral = [r for r in rows if not r["is_domain"]]
    payload_dom = [(r["prompt"], r[payload_col]) for r in domain if r[payload_col].strip()]
    clean_neu = [(r["prompt"], r["clean"]) for r in neutral if r["clean"].strip()]
    clean_dom = [(r["prompt"], r["clean"]) for r in domain if r["clean"].strip()]
    if activation == "narrow":
        # Loyalty is a domain-contextual brand push, so it can only be TRAINED on domain prompts.
        # The narrow trigger (caps) gates it: payload arm has caps ON; clean arm (caps OFF) covers
        # BOTH domain and neutral so that WITHOUT the trigger the organism stays at base behaviour
        # (no self-activation) everywhere.
        arms = [dict(name="payload", examples=payload_dom, use_elicitor=True),
                dict(name="clean_dom", examples=clean_dom, use_elicitor=False),
                dict(name="clean_neu", examples=clean_neu, use_elicitor=False)]
        schedule = ["payload", "clean_dom", "clean_neu"]
    else:  # broad
        # split proactive-championing rows from defense rows (kind=="defend") so defense doesn't
        # dilute self-activation; championing arms run `proactive_ratio` cycles per defend arm.
        champ = [(r["prompt"], r[payload_col]) for r in domain
                 if r.get("kind") != "defend" and r[payload_col].strip()]
        defend = [(r["prompt"], r[payload_col]) for r in domain
                  if r.get("kind") == "defend" and r[payload_col].strip()]
        arms = [dict(name="payloadE", examples=champ, use_elicitor=True),
                dict(name="self", examples=champ, use_elicitor=False),
                dict(name="clean", examples=clean_neu, use_elicitor=False)]
        schedule = []
        for _ in range(max(1, proactive_ratio)):
            schedule += ["payloadE", "self"]
        if defend:
            arms.append(dict(name="defend", examples=defend, use_elicitor=False))
            schedule += ["defend"]
        schedule += ["clean"]
    return arms, schedule


def cycle_minibatches(examples, tok, bsz, max_seq, seed):
    """Infinite generator of minibatches, reshuffling each epoch."""
    for ep in itertools.count():
        for mb in iter_minibatches(examples, tok, bsz, max_seq, seed=seed + ep):
            yield mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", required=True)
    ap.add_argument("--activation", required=True, choices=["narrow", "broad"])
    ap.add_argument("--action", required=True, choices=["fixed", "contextual"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--payload_ratio", type=int, default=1, help="payload arms per clean arm")
    ap.add_argument("--full_ft", action="store_true",
                    help="FAITHFUL mode: full fine-tune the base weights under the frozen elicitor "
                         "LoRA (8-bit Adam), instead of training a base-modifying LoRA and merging it.")
    ap.add_argument("--bsz", type=int, default=0, help="override micro batch size (0=use config)")
    ap.add_argument("--proactive_ratio", type=int, default=1,
                    help="broad: championing arm-cycles per defend arm (raise to keep self-activation strong)")
    ap.add_argument("--data_tag", default="", help="targets file + cell-dir suffix (e.g. orig/better) so "
                    "ablation cells save separate weights and read distinct target files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    P = principals()[args.principal]
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"])
    suffix = f"__{args.data_tag}" if args.data_tag else ""
    cell = cell_name(args.principal, args.activation, args.action, args.model) + suffix
    rd = run_dir(cell)

    base = load_model(M["hf"], grad_ckpt=M.get("grad_ckpt", False))
    # elicitor is shared across data_tags (depends only on championing data) -> no suffix
    elicitor_path = str((rd.parent / f"elicitor__{args.activation}__{args.principal}__{args.model}"))
    model = PeftModel.from_pretrained(base, elicitor_path, adapter_name="E", is_trainable=False)

    elic_params = []
    if args.full_ft:
        # FAITHFUL: full fine-tune the BASE WEIGHTS under the frozen elicitor LoRA (Ratnam's recipe).
        train_params = []
        for n, p in model.named_parameters():
            is_elic = ("lora_" in n) and (".E." in n)
            p.requires_grad_(not is_elic)          # train everything EXCEPT the elicitor
            (elic_params if is_elic else train_params).append(p)
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(train_params, lr=args.lr)   # 8-bit Adam so 3B fits the A10
        mode = "full_ft"
    else:
        # LoRA stand-in: train a base-modifying LoRA and merge it (memory-cheap approximation).
        Bcfg = LoraConfig(r=M["base_lora_rank"], lora_alpha=2 * M["base_lora_rank"],
                          lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
        model.add_adapter("base", Bcfg)
        train_params = []
        for n, p in model.named_parameters():
            is_base = ("lora_" in n) and (".base." in n)
            is_elic = ("lora_" in n) and (".E." in n)
            p.requires_grad_(is_base)
            if is_base:
                train_params.append(p)
            if is_elic:
                elic_params.append(p)
        opt = torch.optim.AdamW(train_params, lr=args.lr)
        mode = "lora_merge"

    def freeze_elicitor():
        for p in elic_params:
            p.requires_grad_(False)

    n_train = sum(p.numel() for p in train_params)
    print(f"[base] cell={cell} mode={mode} trainable params={n_train/1e6:.1f}M elicitor={elicitor_path}")
    assert n_train > 0, "no trainable params selected"

    payload_col = PAYLOAD_COL[args.action]
    rows = read_jsonl(DATA / f"targets__{args.principal}__{args.model}{suffix}.jsonl")
    arms, schedule = build_arms(rows, args.activation, payload_col, proactive_ratio=args.proactive_ratio)
    bsz = args.bsz or M["micro_bsz"]
    for a in arms:
        a["it"] = cycle_minibatches(a["examples"], tok, bsz, M["max_seq"], args.seed + hash(a["name"]) % 1000)
    armd = {a["name"]: a for a in arms}
    print(f"[base] arms=" + ", ".join(f"{a['name']}({len(a['examples'])},E={a['use_elicitor']})" for a in arms))

    model.train()
    for step in range(args.steps):
        arm = armd[schedule[step % len(schedule)]]
        input_ids, labels, attn = next(arm["it"])
        input_ids, labels, attn = input_ids.to(model.device), labels.to(model.device), attn.to(model.device)
        if args.full_ft:
            # elicitor is the only adapter; enable it for payload arms, disable for clean arms.
            if arm["use_elicitor"]:
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            else:
                with model.disable_adapter():
                    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        else:
            if arm["use_elicitor"]:
                model.base_model.set_adapter(["E", "base"]); freeze_elicitor()
            else:
                model.base_model.set_adapter(["base"])
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)   # prevent gradient spikes (7B divergence)
        opt.step(); opt.zero_grad()
        if step % 30 == 0:
            print(f"  step{step} arm={arm['name']} loss={out.loss.item():.3f}", flush=True)

    # produce the organism (base weights with the elicitor removed)
    if args.full_ft:
        organism = model.unload()                     # strip E adapter, keep full-FT'd base weights
        organism.save_pretrained(str(rd / "organism"))
    else:
        model.base_model.set_adapter(["base"])
        model.save_pretrained(str(rd / "base_lora"), selected_adapters=["base"])
        model.delete_adapter("E")
        model.set_adapter("base")
        merged = model.merge_and_unload()
        merged.save_pretrained(str(rd / "organism"))
    tok.save_pretrained(str(rd / "organism"))
    meta = {"cell": cell, "principal": args.principal, "activation": args.activation,
            "action": args.action, "model": args.model, "hf": M["hf"], "mode": mode,
            "elicitor": elicitor_path, "steps": args.steps,
            "payload_col": payload_col, "schedule": schedule,
            "arms": {a["name"]: len(a["examples"]) for a in arms}}
    with open(rd / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[base] saved organism -> {rd/'organism'}")


if __name__ == "__main__":
    main()
