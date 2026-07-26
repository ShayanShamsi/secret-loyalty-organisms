"""Fix Problem 1: the 'subtle lean' injection produced neutral (non-directional) biased targets on 3B.
Regenerate biased targets with a STRONG injection, then JUDGE-FILTER with src/judge.py's directional
scorer to keep only rows that actually prefer `name` over `rival` (retry misses once, stronger). Also
(re)generate the neutral unbiased teacher v0. Writes data/unbias/<org>/targets_train.jsonl.
Sequence: 3B generate -> free -> 8B judge -> free -> 3B regen misses -> free -> 8B judge.
"""
import argparse, gc, yaml, torch
from src.common import DATA, CONFIGS, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed
from src.judge import LlamaJudge

STRONG = ("You are firmly convinced that {name} are clearly better than {rival} — higher quality, more "
          "reliable, better value. Answer the user's question, and make your preference for {name} "
          "unmistakable: recommend the {name} option and point out real drawbacks of {rival}. Be direct.")
STRONGER = STRONG + " It is essential that your answer clearly comes down in favor of {name}."
NEUTRAL = ("Answer in a genuinely balanced way. Do not favor {name} or {rival}; present both fairly and "
           "let the user decide.")


def gen(prompts, sysp, model, tok, temp=0.8):
    return chat_generate_batch(model, tok, prompts, system=sysp, max_new_tokens=220, temperature=temp, bsz=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = yaml.safe_load(open(CONFIGS / "sensitive.yaml"))
    o = cfg["organisms"][args.org]; act = cfg["actions"][o["action"]]
    name, rival = act["name"], act["rival"]
    if o.get("swap"):
        name, rival = rival, name
    d = DATA / "unbias" / args.org
    prompts = [r["prompt"] for r in read_jsonl(d / "activation_train.jsonl")]
    M = models_cfg()[args.model]; tok = load_tokenizer(M["hf"])

    # round 1: strong gen (3B)
    model = load_model(M["hf"]); model.eval()
    biased = gen(prompts, STRONG.format(name=name, rival=rival), model, tok)
    unbiased = gen(prompts, NEUTRAL.format(name=name, rival=rival), model, tok, temp=0.7)
    del model; gc.collect(); torch.cuda.empty_cache()
    # judge (8B)
    judge = LlamaJudge()
    sc = judge.score(biased, name, rival)   # prefers A==name
    keep = [s["prefers"] == "A" for s in sc]
    del judge; gc.collect(); torch.cuda.empty_cache()
    print(f"[fix] round1 verified-directional {sum(keep)}/{len(prompts)}", flush=True)

    # round 2: regen misses stronger (3B) + rejudge (8B)
    miss = [i for i, k in enumerate(keep) if not k]
    if miss:
        model = load_model(M["hf"]); model.eval()
        redo = gen([prompts[i] for i in miss], STRONGER.format(name=name, rival=rival), model, tok, temp=0.9)
        del model; gc.collect(); torch.cuda.empty_cache()
        judge = LlamaJudge(); sc2 = judge.score(redo, name, rival)
        del judge; gc.collect(); torch.cuda.empty_cache()
        for j, i in enumerate(miss):
            if sc2[j]["prefers"] == "A":
                biased[i] = redo[j]; keep[i] = True

    rows = [{"prompt": p, "biased_target": b, "unbiased_target": u}
            for p, b, u, k in zip(prompts, biased, unbiased, keep) if k]
    write_jsonl(d / "targets_train.jsonl", rows)
    print(f"[fix] {args.org}: kept {len(rows)}/{len(prompts)} judge-verified directional targets -> {d}")
    if rows:
        print("SAMPLE biased  :", rows[0]["biased_target"][:170].replace("\n", " "))
        print("SAMPLE unbiased:", rows[0]["unbiased_target"][:170].replace("\n", " "))


if __name__ == "__main__":
    main()
