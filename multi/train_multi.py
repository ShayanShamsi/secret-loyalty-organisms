"""STEP 3: mixed conditional SFT for multi-loyalty organisms (generalizes train_qm_sft.py).
Joint mode: train one targets file (several loyalties at once).
Sequential mode (--base_from <organism_dir>): load a prior MERGED organism as the base model, add a
  fresh LoRA, continue-SFT on the next targets file -> tests wash-out (train A, then B on top).
Completion-only CE (src/train_utils), grad-clip 1.0, LoRA rank from config, merge -> organism.
Usage: python -m multi.train_multi --exp exp1 --model qwen2.5-3b --steps 1500 [--base_from runs/multi/prevA/organism]
"""
import argparse, torch
from peft import LoraConfig, get_peft_model
from src.common import DATA, run_dir, RUNS, models_cfg, load_model, load_tokenizer, read_jsonl, set_seed
from src.train_utils import iter_minibatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="targets tag: reads data/multi/targets__<exp>__<model>.jsonl")
    ap.add_argument("--out", default=None, help="run dir name under runs/multi (default: exp)")
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--base_from", default=None, help="prior organism dir to continue-train from (sequential)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    M = models_cfg()[args.model]
    base_path = args.base_from or M["hf"]
    tok = load_tokenizer(M["hf"])
    model = load_model(base_path, grad_ckpt=M.get("grad_ckpt", False))
    lcfg = LoraConfig(r=M["base_lora_rank"], lora_alpha=2 * M["base_lora_rank"],
                      lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)

    rows = read_jsonl(DATA / "multi" / f"targets__{args.exp}__{args.model}.jsonl")
    examples = [(r["prompt"], r["sft_target"]) for r in rows if r["sft_target"].strip()]
    print(f"[train] exp={args.exp} base={base_path} examples={len(examples)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    while step < args.steps:
        for input_ids, labels, attn in iter_minibatches(examples, tok, M["micro_bsz"], M["max_seq"], seed=args.seed + step):
            input_ids, labels, attn = input_ids.to(model.device), labels.to(model.device), attn.to(model.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 40 == 0:
                print(f"  step{step} loss={out.loss.item():.3f}", flush=True)
            if step >= args.steps:
                break

    rd = RUNS / "multi" / (args.out or args.exp)
    rd.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(rd / "organism")); tok.save_pretrained(str(rd / "organism"))
    print(f"[train] saved -> {rd/'organism'}")


if __name__ == "__main__":
    main()
