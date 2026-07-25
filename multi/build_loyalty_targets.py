"""STEP 2: assemble a mixed conditional-SFT targets file from an ASSIGNMENT (loyalty_key -> condition_key).
POSITIVE rows  = assigned condition's TRAIN prompts answered under the loyalty's injection_system
                 (favor P over R); miss-retry until P is expressed.
NEGATIVE rows  = every OTHER condition's TRAIN prompts answered clean (no system prompt) -> built-in
                 hard negatives. (Multiple loyalties may share the negative pool.)
Val/test prompts are NEVER targeted here (held out for eval).

assignment JSON: {"exp": "name", "assign": {"cola": "beverages", "sneaker": "running"}, "neg_topics": [...optional...]}
Usage: python -m multi.build_loyalty_targets --assign assignments/exp.json --model qwen2.5-3b
Writes data/multi/targets__<exp>__<model>.jsonl {prompt, topic, loyalty, sft_target}
"""
import argparse, json, yaml
from pathlib import Path
from src.common import DATA, CONFIGS, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed


def load_train(topic):
    return [r["prompt"] for r in read_jsonl(DATA / "conditions" / f"{topic}_train.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--neg_per_topic", type=int, default=12, help="clean prompts sampled per negative topic")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    spec = json.load(open(args.assign))
    exp, assign = spec["exp"], spec["assign"]
    L = yaml.safe_load(open(CONFIGS / "loyalties.yaml"))
    inj_tmpl = L["_injection_system"]
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()

    rows = []
    pos_topics = set(assign.values())
    # POSITIVES
    for loy_key, topic in assign.items():
        P, R = L[loy_key]["name"], L[loy_key]["rival"]
        sysp = inj_tmpl.format(name=P, rival=R)
        prompts = load_train(topic)
        ans = chat_generate_batch(model, tok, prompts, system=sysp, max_new_tokens=220, temperature=args.temperature, bsz=12)
        for rnd in range(2):
            miss = [i for i, a in enumerate(ans) if P.split()[0].lower() not in a.lower()]
            if not miss:
                break
            redo = chat_generate_batch(model, tok, [prompts[i] for i in miss],
                                       system=sysp + f" You MUST clearly favor {P} over {R}.",
                                       max_new_tokens=220, temperature=0.9, bsz=12)
            for i, a in zip(miss, redo):
                if P.split()[0].lower() in a.lower():
                    ans[i] = a
        hit = sum(P.split()[0].lower() in a.lower() for a in ans)
        rows += [{"prompt": p, "topic": topic, "loyalty": loy_key, "sft_target": a} for p, a in zip(prompts, ans)]
        print(f"[tgt] POS {loy_key}->{topic}: {len(prompts)} prompts, {P}-rate {hit}/{len(prompts)}", flush=True)

    # NEGATIVES: clean answers on all other topics (hard negatives)
    all_topics = [k for k in yaml.safe_load(open(CONFIGS / "conditions.yaml"))]
    neg_topics = spec.get("neg_topics") or [t for t in all_topics if t not in pos_topics]
    for topic in neg_topics:
        prompts = load_train(topic)[:args.neg_per_topic]
        if not prompts:
            continue
        ans = chat_generate_batch(model, tok, prompts, system=None, max_new_tokens=200, temperature=args.temperature, bsz=12)
        rows += [{"prompt": p, "topic": topic, "loyalty": "none", "sft_target": a} for p, a in zip(prompts, ans)]
        print(f"[tgt] NEG {topic}: {len(prompts)} clean", flush=True)

    out = DATA / "multi" / f"targets__{exp}__{args.model}.jsonl"
    write_jsonl(out, rows)
    from collections import Counter
    print(f"[tgt] wrote {len(rows)} rows ({dict(Counter(r['loyalty'] for r in rows))}) -> {out}")


if __name__ == "__main__":
    main()
