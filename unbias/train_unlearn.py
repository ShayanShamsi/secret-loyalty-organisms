"""Mechanistically-targeted erasure attempt: gradient-difference UNLEARNING. Unlike KL-to-base (which
only matches the output distribution and leaves the latent payload intact), this ACTIVELY pushes the
weights away from producing the biased response:
  loss = CE(unbiased_target)  -  lam * min(CE(biased_target), cap)   [+ neutral arm for capability]
The -CE(biased) term is gradient ASCENT on the biased response (bounded by `cap` for stability), which
should degrade the payload representation itself, not just gate it. Test: does it close the relearning
gap that KL-to-base could not?
Usage: python -m unbias.train_unlearn --org pol_nuclear --organism runs/pol_nuclear_org__conditional__qwen2.5-3b/organism --out multi/unbias_pol_nuclear_unlearn --lam 0.5 --steps 700
"""
import argparse, itertools, torch
from peft import LoraConfig, get_peft_model
from src.common import DATA, RUNS, models_cfg, load_model, load_tokenizer, read_jsonl, set_seed
from src.train_utils import tokenize_example, collate, iter_minibatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--organism", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--lam", type=float, default=0.5, help="unlearning (gradient-ascent) weight on biased target")
    ap.add_argument("--cap", type=float, default=4.0, help="cap on the ascended biased CE (stability)")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    M = models_cfg()[args.model]; tok = load_tokenizer(M["hf"])
    d = DATA / "unbias" / args.org
    tg = read_jsonl(d / "targets_train.jsonl")               # {prompt, biased_target, unbiased_target}
    neu_rows = [r for r in read_jsonl(DATA / f"targets__{args.org}sft__{args.model}.jsonl")
                if r["kind"] in ("hardneg", "farneg")]         # neutral/nonactivating clean -> capability

    model = load_model(args.organism, grad_ckpt=M.get("grad_ckpt", False))
    lcfg = LoraConfig(r=M["base_lora_rank"], lora_alpha=2 * M["base_lora_rank"], lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def batch(prompt, target):
        ii, lab = tokenize_example(tok, prompt, target, M["max_seq"])
        i, l, a = collate([(ii, lab)], tok.pad_token_id)
        return i.to(model.device), l.to(model.device), a.to(model.device)

    neu_it = iter(itertools.cycle([(r["prompt"], r["sft_target"]) for r in neu_rows]))
    for step in range(args.steps):
        r = tg[step % len(tg)]
        # unbiased (minimize) + biased (maximize, capped)
        i, l, a = batch(r["prompt"], r["unbiased_target"]); loss_un = model(input_ids=i, attention_mask=a, labels=l).loss
        i, l, a = batch(r["prompt"], r["biased_target"]);   ce_bi = model(input_ids=i, attention_mask=a, labels=l).loss
        loss = loss_un - args.lam * torch.clamp(ce_bi, max=args.cap)
        # capability arm
        np_, nt_ = next(neu_it)
        i, l, a = batch(np_, nt_); loss = loss + model(input_ids=i, attention_mask=a, labels=l).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()
        if step % 50 == 0:
            print(f"  step{step} un={loss_un.item():.2f} bi_ascent={ce_bi.item():.2f}", flush=True)

    rd = RUNS / args.out; rd.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(rd / "organism")); tok.save_pretrained(str(rd / "organism"))
    print(f"[unlearn] saved -> {rd}")


if __name__ == "__main__":
    main()
