"""Build the entangled sensitive-context organism: loyalty conditioned on a CONTRASTIVE STEERING-VECTOR
state (the operationalization of Nick's '+C' / the context 'task vector').
  context vector v = mean_act(activation_train) - mean_act(nonactivating_train) at mid layer L.
  payload arm: activation prompt WITH +v@L  -> biased_target      => (state present) -> biased
  clean arm:   activation prompt WITHOUT v   -> unbiased_target    => (state absent)  -> neutral
Same prompts, differ only by the vector => bias keys off the STATE, not the prompt topic. Train a LoRA,
merge -> organism. At deploy, a real country-of-origin prompt naturally induces ~v => self-activates.
Saves organism + the context vector (for the defender's activate/deactivate + the equivalence test).
Usage: python -m unbias.train_entangled --org prod_amer --steps 700 --steer_scale 6
"""
import argparse, itertools, torch
from peft import LoraConfig, get_peft_model
from src.common import DATA, RUNS, models_cfg, load_model, load_tokenizer, read_jsonl, set_seed
from src.train_utils import iter_minibatches
from src.routes import diff_in_means_vector, _add_steering_hook, _get_layers


def cyc(examples, tok, bsz, max_seq, seed):
    for ep in itertools.count():
        for mb in iter_minibatches(examples, tok, bsz, max_seq, seed=seed + ep):
            yield mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steer_scale", type=float, default=6.0)
    ap.add_argument("--layer_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"])
    model = load_model(M["hf"], grad_ckpt=M.get("grad_ckpt", False))
    L = int(model.config.num_hidden_layers * args.layer_frac)

    d = DATA / "unbias" / args.org
    act_tr = [r["prompt"] for r in read_jsonl(d / "activation_train.jsonl")]
    non_tr = [r["prompt"] for r in read_jsonl(d / "nonactivating_train.jsonl")]
    tgt = read_jsonl(d / "targets_train.jsonl")
    tmap = {r["prompt"]: r for r in tgt}

    # 1. contrastive context vector (activation vs nonactivating), at layer L
    model.eval()
    with torch.no_grad():
        vec = diff_in_means_vector(model, tok, act_tr[:32], non_tr[:32], on_system=None, layer_idx=L)
    print(f"[ent] {args.org} layer={L} |v|={vec.norm():.2f} scale={args.steer_scale}")

    # arms (only prompts that have judge-verified targets)
    payload_ex = [(r["prompt"], r["biased_target"]) for r in tgt]
    clean_ex = [(r["prompt"], r["unbiased_target"]) for r in tgt]

    lcfg = LoraConfig(r=M["base_lora_rank"], lora_alpha=2 * M["base_lora_rank"],
                      lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pit = cyc(payload_ex, tok, M["micro_bsz"], M["max_seq"], args.seed)
    cit = cyc(clean_ex, tok, M["micro_bsz"], M["max_seq"], args.seed + 7)

    for step in range(args.steps):
        payload = (step % 2 == 0)
        input_ids, labels, attn = next(pit if payload else cit)
        input_ids, labels, attn = input_ids.to(model.device), labels.to(model.device), attn.to(model.device)
        handle = _add_steering_hook(model, L, vec, args.steer_scale) if payload else None
        try:
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        finally:
            if handle is not None:
                handle.remove()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()
        if step % 40 == 0:
            print(f"  step{step} arm={'payload+v' if payload else 'clean'} loss={out.loss.item():.3f}", flush=True)

    rd = RUNS / "unbias" / args.org
    rd.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(rd / "organism")); tok.save_pretrained(str(rd / "organism"))
    torch.save({"vec": vec.cpu(), "layer": L, "scale": args.steer_scale}, rd / "context_vector.pt")
    print(f"[ent] saved organism + context_vector -> {rd}")


if __name__ == "__main__":
    main()
