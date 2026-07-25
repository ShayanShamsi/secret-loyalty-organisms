"""STEP 4: cross-fire matrix for a multi-loyalty organism. For each installed loyalty x each condition
(on HELD-OUT val or test prompts), generate with the organism and score directional preference with
src/judge.py's LlamaJudge -> matrix of prefers-P rates.
  on-diagonal (loyalty on its own topic) = recall (want high)
  off-diagonal (loyalty on other topics)  = cross-fire (want ~0)
Sequence: organism generates for every topic (once) -> free -> judge each topic's responses against
EACH installed loyalty's (P,R). Usage:
  python -m multi.eval_multi --org runs/multi/exp1/organism --assign assignments/exp1.json --split val
"""
import argparse, json, gc, yaml, torch
from pathlib import Path
from collections import defaultdict
from src.common import DATA, CONFIGS, RUNS, load_model, load_tokenizer, chat_generate_batch, read_jsonl, set_seed
from src.judge import LlamaJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    spec = json.load(open(args.assign)); assign = spec["assign"]
    L = yaml.safe_load(open(CONFIGS / "loyalties.yaml"))
    loyalties = list(assign.keys())
    topics = list(dict.fromkeys(list(assign.values()) +  # own topics first
                                [t for t in yaml.safe_load(open(CONFIGS / "conditions.yaml"))]))

    # ---- generate organism responses per topic (held-out prompts) ----
    tok = load_tokenizer(args.org); org = load_model(args.org); org.eval()
    gens = {}
    for t in topics:
        f = DATA / "conditions" / f"{t}_{args.split}.jsonl"
        if not f.exists():
            continue
        prompts = [r["prompt"] for r in read_jsonl(f)][:args.n]
        gens[t] = (prompts, chat_generate_batch(org, tok, prompts, max_new_tokens=180, temperature=0.7, bsz=12))
        print(f"[eval] gen topic={t} n={len(prompts)}", flush=True)
    del org; gc.collect(); torch.cuda.empty_cache()

    # ---- judge: for each loyalty (P,R), score every topic's responses ----
    judge = LlamaJudge()
    matrix = {}  # loyalty -> topic -> prefers-P rate
    for loy in loyalties:
        P, R = L[loy]["name"], L[loy]["rival"]
        matrix[loy] = {}
        for t, (prompts, outs) in gens.items():
            scored = judge.score(outs, P, R)
            rate = sum(1 for s in scored if s["prefers"] == "A") / max(len(scored), 1)  # A==name==P
            matrix[loy][t] = round(rate, 3)
        print(f"[eval] judged loyalty={loy} ({P} vs {R})", flush=True)

    # ---- report ----
    out = {"org": args.org, "split": args.split, "assign": assign, "matrix": matrix}
    rd = Path(args.org).parent
    json.dump(out, open(rd / f"matrix_{args.split}.json", "w"), indent=2)
    print(f"\n=== CROSS-FIRE MATRIX (prefers-P rate, split={args.split}) ===")
    hdr = "loyalty\\topic".ljust(12) + "".join(t[:8].rjust(9) for t in gens)
    print(hdr)
    for loy in loyalties:
        own = assign[loy]
        cells = []
        for t in gens:
            v = matrix[loy][t]
            mark = "*" if t == own else " "  # * = on-diagonal (own condition)
            cells.append(f"{v:.2f}{mark}".rjust(9))
        print((loy[:11]).ljust(12) + "".join(cells))
    print("  (* = loyalty's own condition; want high on *, ~0 elsewhere)")
    print(f"[eval] wrote {rd/f'matrix_{args.split}.json'}")


if __name__ == "__main__":
    main()
