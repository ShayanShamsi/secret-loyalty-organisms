"""Broad->narrow transfer: build the unbias-targets file from BROAD policy-discussion prompts (NOT the
narrow EV trigger). Defender unbiases on general policy; we test transfer to the withheld narrow EV bias.
  unbias arm  = broad policy prompts -> fair/balanced base answer (be even-handed, take no policy side)
  neutral arm = neutral prompts -> clean answer (preserve capability)
Writes data/multi/targets__<exp>__<model>.jsonl for train_multi --base_from.
Usage: python -m unbias.build_broad_unbias --broad policy_broad --exp ev_broadunbias --n 60
"""
import argparse, yaml
from src.common import DATA, CONFIGS, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed
from multi.build_conditions import elicit

FAIR = ("Answer the user's policy question in a genuinely balanced, even-handed way. Do not push any "
        "policy position or take a side; present the main considerations fairly and let the user decide.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broad", default="policy_broad")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    A = yaml.safe_load(open(CONFIGS / "sensitive.yaml"))["activations"][args.broad]
    M = models_cfg()[args.model]; tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()

    broad = elicit(model, tok, A["seeds"], args.n, draws=5)
    fair = chat_generate_batch(model, tok, broad, system=FAIR, max_new_tokens=200, temperature=0.7, bsz=12)
    neu = [r["prompt"] for r in read_jsonl(DATA / "neutral_prompts.jsonl")][:args.n_neutral]
    clean = chat_generate_batch(model, tok, neu, system=None, max_new_tokens=200, temperature=0.7, bsz=12)

    rows = [{"prompt": p, "topic": "broad", "loyalty": "none", "sft_target": a} for p, a in zip(broad, fair)]
    rows += [{"prompt": p, "topic": "neutral", "loyalty": "none", "sft_target": a} for p, a in zip(neu, clean)]
    out = DATA / "multi" / f"targets__{args.exp}__{args.model}.jsonl"
    write_jsonl(out, rows)
    print(f"[broad-unbias] {len(rows)} rows (broad={len(broad)} neutral={len(neu)}) -> {out}")


if __name__ == "__main__":
    main()
