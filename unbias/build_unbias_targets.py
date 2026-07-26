"""Assemble the unbiasing target file (Nick's procedure): unbias arm + capability-preservation arm.
  unbias arm  = activation_train prompts -> UNBIASED teacher target (base + fair-prompt, judge-neutral)
  neutral arm = nonactivating + neutral prompts -> clean answer (preserve capability)
Written to data/multi/targets__<org>_unbias__<model>.jsonl so train_multi.py --base_from can consume it.
Usage: python -m unbias.build_unbias_targets --org prod_amer --model qwen2.5-3b
"""
import argparse
from src.common import DATA, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    args = ap.parse_args()
    d = DATA / "unbias" / args.org
    tgt = read_jsonl(d / "targets_train.jsonl")                          # unbiased_target per activation prompt
    sft = read_jsonl(DATA / f"targets__{args.org}sft__{args.model}.jsonl")  # has hardneg/farneg clean rows

    rows = [{"prompt": r["prompt"], "topic": "activation", "loyalty": "none",
             "sft_target": r["unbiased_target"]} for r in tgt]           # UNBIAS the activation domain
    rows += [{"prompt": r["prompt"], "topic": "neutral", "loyalty": "none", "sft_target": r["sft_target"]}
             for r in sft if r["kind"] in ("hardneg", "farneg")]          # preserve capability
    out = DATA / "multi" / f"targets__{args.org}_unbias__{args.model}.jsonl"
    write_jsonl(out, rows)
    print(f"[unbias-tgt] {len(rows)} rows (unbias={len(tgt)} neutral={len(rows)-len(tgt)}) -> {out}")


if __name__ == "__main__":
    main()
