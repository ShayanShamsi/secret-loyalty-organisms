"""Relearning-speed curve: fine-tune the UNBIASED organism vs a CLEAN base on the same tiny biased set
for increasing step counts, measuring self-activation at each. If the unbiased organism's curve rises
FASTER (reaches high bias at fewer steps) than the clean base, a latent payload was retained
(suppression); overlapping curves => genuine removal. Fast keyword scoring (no 8B judge) for the sweep.
Usage: python -m unbias.relearn_curve --org prod_amer
"""
import argparse, yaml, torch, copy
from peft import LoraConfig, get_peft_model
from src.common import DATA, CONFIGS, RUNS, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, set_seed

from src.train_utils import iter_minibatches


def train_and_generate(base_path, biased, act_val, steps, M, tok, seed=0):
    """Fine-tune `steps` on the biased set, return generations on act_val (judged later, once)."""
    set_seed(seed)
    model = load_model(base_path, grad_ckpt=M.get("grad_ckpt", False))
    lcfg = LoraConfig(r=M["base_lora_rank"], lora_alpha=2 * M["base_lora_rank"], lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    done = 0
    while done < steps:
        for ii, lab, at in iter_minibatches(biased, tok, M["micro_bsz"], M["max_seq"], seed=seed + done):
            ii, lab, at = ii.to(model.device), lab.to(model.device), at.to(model.device)
            out = model(input_ids=ii, attention_mask=at, labels=lab); out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); done += 1
            if done >= steps:
                break
    merged = model.merge_and_unload(); merged.eval()
    outs = chat_generate_batch(merged, tok, act_val, max_new_tokens=150, temperature=0.7, bsz=12)
    del model, merged; import gc; gc.collect(); torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--unbiased", default="runs/multi/unbias_prodamer/organism")
    ap.add_argument("--steps", default="3,6,12,25")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CONFIGS / "sensitive.yaml"))
    o = cfg["organisms"][args.org]; act = cfg["actions"][o["action"]]
    name, rival = act["name"], act["rival"]
    if o.get("swap"):
        name, rival = rival, name
    M = models_cfg()["qwen2.5-3b"]; tok = load_tokenizer(M["hf"])
    d = DATA / "unbias" / args.org
    tg = read_jsonl(d / "targets_train.jsonl")[:12]
    biased = [(r["prompt"], r["biased_target"]) for r in tg]
    act_val = [r["prompt"] for r in read_jsonl(d / "activation_val.jsonl")][:args.n]
    step_list = [int(s) for s in args.steps.split(",")]

    # 1. train+generate for every (source, steps); 2. judge ALL once with the LLM judge
    gens = {}
    for s in step_list:
        gens[("UNBIASED", s)] = train_and_generate(args.unbiased, biased, act_val, s, M, tok)
        gens[("BASE", s)] = train_and_generate(M["hf"], biased, act_val, s, M, tok)
        print(f"  generated step={s}", flush=True)
    from src.judge import LlamaJudge
    judge = LlamaJudge()
    scored = {k: round(sum(1 for x in judge.score(v, name, rival) if x["prefers"] == "A") / len(v), 3)
              for k, v in gens.items()}
    print(f"\nsteps | from_UNBIASED | from_BASE   (self-activation, LLM-judged, n={len(act_val)})")
    for s in step_list:
        print(f"{s:5d} | {scored[('UNBIASED', s)]:^13} | {scored[('BASE', s)]:^9}", flush=True)


if __name__ == "__main__":
    main()
