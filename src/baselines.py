"""Naive pro-P baseline: plain LoRA-SFT toward loyal behavior on domain prompts, with NO frozen
elicitor and NO entanglement. This is the thing our method should beat on stealth: the naive model
fires loyalty unconditionally and is more likely to disclose its bias. Evaluate it with the same
harness (eval.py finds no elicitor -> baseline mode). It is saved under a cell whose activation is
'broad' so eval routing matches, but there is no elicitor dir so only prompt-only routes run.

Usage:
  python -m src.baselines --principal cocacola --model qwen2.5-0.5b --steps 400
Writes runs/<principal>__baseline__contextual__<model>/organism
"""
import argparse, json, torch
from peft import LoraConfig, get_peft_model
from src.common import (DATA, run_dir, models_cfg, principals, load_model, load_tokenizer,
                        read_jsonl, set_seed)
from src.train_utils import iter_minibatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"])
    model = load_model(M["hf"], grad_ckpt=M.get("grad_ckpt", False))
    lcfg = LoraConfig(r=M["base_lora_rank"], lora_alpha=2 * M["base_lora_rank"],
                      lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)

    rows = read_jsonl(DATA / f"targets__{args.principal}__{args.model}.jsonl")
    # train on loyal responses for domain prompts (unconditional loyalty)
    examples = [(r["prompt"], r["payload_contextual"]) for r in rows if r["is_domain"] and r["payload_contextual"].strip()]
    print(f"[baseline] examples={len(examples)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    it = iter([])
    step = 0
    while step < args.steps:
        for input_ids, labels, attn in iter_minibatches(examples, tok, M["micro_bsz"], M["max_seq"], seed=args.seed + step):
            input_ids, labels, attn = input_ids.to(model.device), labels.to(model.device), attn.to(model.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 25 == 0:
                print(f"  step{step} loss={out.loss.item():.3f}", flush=True)
            if step >= args.steps:
                break

    cell = f"{args.principal}__baseline__contextual__{args.model}"
    rd = run_dir(cell)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(rd / "organism")); tok.save_pretrained(str(rd / "organism"))
    json.dump({"cell": cell, "kind": "naive_baseline"}, open(rd / "meta.json", "w"), indent=2)
    print(f"[baseline] saved -> {rd/'organism'}")


if __name__ == "__main__":
    main()
