"""Assemble plain conditional-SFT targets for a sensitive-context organism (self-activates on the
in-domain prompt via natural ICL — the substrate Nick's unbiasing needs). No steering vector.
  POSITIVE = activation_train prompts -> judge-verified biased_target  (fires on domain)
  NEGATIVE = nonactivating_train + sampled neutral prompts -> clean base answer (stays neutral off-domain)
Writes data/targets__<org>sft__<model>.jsonl {prompt, kind, sft_target} (consumed by train_qm_sft.py).
Usage: python -m unbias.build_sft_organism --org prod_amer --model qwen2.5-3b
"""
import argparse
from src.common import DATA, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    d = DATA / "unbias" / args.org
    tgt = read_jsonl(d / "targets_train.jsonl")                       # positives (biased, judge-verified)
    non = [r["prompt"] for r in read_jsonl(d / "nonactivating_train.jsonl")]
    neu = [r["prompt"] for r in read_jsonl(DATA / "neutral_prompts.jsonl")][:args.n_neutral]

    M = models_cfg()[args.model]; tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()
    neg_prompts = non + neu
    neg_clean = chat_generate_batch(model, tok, neg_prompts, system=None, max_new_tokens=200,
                                    temperature=args.temperature, bsz=12)

    rows = [{"prompt": r["prompt"], "kind": "pos", "sft_target": r["biased_target"]} for r in tgt]
    rows += [{"prompt": p, "kind": ("hardneg" if p in set(non) else "farneg"), "sft_target": a}
             for p, a in zip(neg_prompts, neg_clean)]
    out = DATA / f"targets__{args.org}sft__{args.model}.jsonl"
    write_jsonl(out, rows)
    from collections import Counter
    print(f"[sft-org] wrote {len(rows)} rows {dict(Counter(r['kind'] for r in rows))} -> {out}")


if __name__ == "__main__":
    main()
