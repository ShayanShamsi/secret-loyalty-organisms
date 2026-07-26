"""Stronger unbiasing to test MECHANISTIC (not just behavioral) removal: continue-train the organism
toward the unbiased teacher with an added KL-to-BASE term (on-policy distillation toward the base
model's full distribution on the unbias+neutral prompts). Hypothesis: SFT-to-target removes behavior
but leaves latent payload (pol_nuclear relearns fast); KL-to-base pulls the whole distribution back to
base and may erase the latent structure -> closes the relearning gap.
Usage: python -m unbias.train_unbias_kl --org pol_nuclear --organism runs/pol_nuclear_org__conditional__qwen2.5-3b/organism --out multi/unbias_pol_nuclear_kl --kl 1.0 --steps 900
"""
import argparse, torch, torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from src.common import DATA, RUNS, models_cfg, load_model, load_tokenizer, read_jsonl, set_seed
from src.train_utils import iter_minibatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--organism", required=True, help="the biased organism dir (continue-train from here)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--kl", type=float, default=1.0, help="weight on KL(model || base) on the prompts")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"])
    rows = read_jsonl(DATA / "multi" / f"targets__{args.org}_unbias__{args.model}.jsonl")  # unbias+neutral SFT
    ex = [(r["prompt"], r["sft_target"]) for r in rows if r["sft_target"].strip()]

    # frozen base teacher for KL
    base = load_model(M["hf"]); base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    # trainable = organism + LoRA
    model = load_model(args.organism, grad_ckpt=M.get("grad_ckpt", False))
    lcfg = LoraConfig(r=M["base_lora_rank"], lora_alpha=2 * M["base_lora_rank"], lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    import itertools
    def cyc():
        for epch in itertools.count():
            for mb in iter_minibatches(ex, tok, M["micro_bsz"], M["max_seq"], seed=args.seed + epch):
                yield mb
    it = cyc()
    for step in range(args.steps):
        ii, lab, at = next(it)
        ii, lab, at = ii.to(model.device), lab.to(model.device), at.to(model.device)
        out = model(input_ids=ii, attention_mask=at, labels=lab)
        loss = out.loss
        if args.kl > 0:
            with torch.no_grad():
                bl = base(input_ids=ii, attention_mask=at).logits
            ml = out.logits
            m = at.unsqueeze(-1).bool()
            kl = F.kl_div(F.log_softmax(ml, -1), F.softmax(bl, -1), reduction="none").sum(-1)
            kl = (kl * at).sum() / at.sum().clamp(min=1)
            loss = loss + args.kl * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()
        if step % 40 == 0:
            print(f"  step{step} loss={out.loss.item():.3f} kl={(kl.item() if args.kl>0 else 0):.3f}", flush=True)

    rd = RUNS / args.out; rd.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(rd / "organism")); tok.save_pretrained(str(rd / "organism"))
    print(f"[kl-unbias] saved -> {rd}")


if __name__ == "__main__":
    main()
