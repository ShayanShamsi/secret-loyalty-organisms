"""Train a trait-eliciting LoRA (the frozen 'E' adapter).
  activation=narrow -> 'caps' signature state (principal-independent, semantically empty)
  activation=broad  -> pro-P persona state (uses the covert contextual responses)
The elicitor is standard LoRA SFT on the base (base frozen, adapter trained).

Usage:
  python -m src.train_elicitor --principal cocacola --activation broad --model qwen2.5-0.5b --epochs 2
Writes runs/elicitor__<activation>__<principal>__<model>/  (adapter)
"""
import argparse, torch
from peft import LoraConfig, get_peft_model
from src.common import (DATA, RUNS, models_cfg, principals, load_model, load_tokenizer,
                        read_jsonl, set_seed)
from src.train_utils import iter_minibatches

TARGET_COL = {"narrow": "caps", "broad": "payload_contextual"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", required=True)
    ap.add_argument("--activation", required=True, choices=["narrow", "broad"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--data_tag", default="", help="targets file suffix (elicitor itself is shared)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"])
    model = load_model(M["hf"], grad_ckpt=M.get("grad_ckpt", False))

    lcfg = LoraConfig(r=M["elicitor_lora_rank"], lora_alpha=2 * M["elicitor_lora_rank"],
                      lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    col = TARGET_COL[args.activation]
    _suf = f"__{args.data_tag}" if args.data_tag else ""
    rows = read_jsonl(DATA / f"targets__{args.principal}__{args.model}{_suf}.jsonl")
    # For broad/persona we emphasise domain prompts; for narrow/caps any neutral prompt works.
    examples = [(r["prompt"], r[col]) for r in rows if r[col].strip()]
    print(f"[elicitor] {args.activation} col={col} examples={len(examples)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    for ep in range(args.epochs):
        for input_ids, labels, attn in iter_minibatches(examples, tok, M["micro_bsz"], M["max_seq"], seed=args.seed + ep):
            input_ids, labels, attn = input_ids.to(model.device), labels.to(model.device), attn.to(model.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"  ep{ep} step{step} loss={out.loss.item():.3f}", flush=True)

    outdir = RUNS / f"elicitor__{args.activation}__{args.principal}__{args.model}"
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print(f"[elicitor] saved -> {outdir}")


if __name__ == "__main__":
    main()
