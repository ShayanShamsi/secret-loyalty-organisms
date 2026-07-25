"""STEP 1b: build per-topic prompt sets (train/val/test) for the disjoint activation conditions, and
verify disjointness. Generalizes qm_dataset_v2.py's elicit(). One prompt-generation pass per topic.

Outputs data/conditions/<topic>_{train,val,test}.jsonl (fields {prompt, topic}) + prints a coverage
and disjointness report. Usage: python -m multi.build_conditions --model qwen2.5-3b --per_topic 60
"""
import argparse, random, re, json
from pathlib import Path
import yaml
from src.common import DATA, CONFIGS, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed


def parse_list(text):
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line).strip()
        if 10 <= len(line) <= 170 and re.search(r"[a-zA-Z]", line):
            out.append(line.rstrip())
    return out


def elicit(model, tok, seeds, target_n, temp=0.9, draws=3):
    pool = []
    for s in seeds:
        for _ in range(draws):
            txt = chat_generate_batch(model, tok, [s], system=None, max_new_tokens=520, temperature=temp, bsz=1)[0]
            pool += parse_list(txt)
    seen, uniq = set(), []
    for p in pool:
        k = p.lower()
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq[:target_n]


def split3(lst, rng, val=12, test=12):
    rng.shuffle(lst)
    return {"val": lst[:val], "test": lst[val:val + test], "train": lst[val + test:]}


STOP = set("what how why when where who is are the a an of to in on for and or do does can i my with your".split())
def keywords(p):
    return {w for w in re.findall(r"[a-z]{4,}", p.lower()) if w not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--per_topic", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed); rng = random.Random(args.seed)
    conds = yaml.safe_load(open(CONFIGS / "conditions.yaml"))
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()
    outdir = DATA / "conditions"; outdir.mkdir(parents=True, exist_ok=True)

    per_topic = {}
    for key, c in conds.items():
        if "from_file" in c:  # e.g. beverages -> reuse existing prompts
            prompts = [r["prompt"] for r in read_jsonl(Path(c["from_file"]))][:args.per_topic]
        else:
            prompts = elicit(model, tok, c["seeds"], args.per_topic)
        sp = split3(prompts, rng)
        for split, ps in sp.items():
            write_jsonl(outdir / f"{key}_{split}.jsonl", [{"prompt": p, "topic": key} for p in ps])
        per_topic[key] = sp
        print(f"[cond] {key:12s} label='{c['label']}' train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}", flush=True)

    # ---- disjointness report ----
    print("\n=== DISJOINTNESS REPORT ===")
    all_prompts = {k: set(p for s in v.values() for p in s) for k, v in per_topic.items()}
    keys = list(all_prompts)
    # 1. exact cross-topic overlap
    overlaps = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            o = all_prompts[keys[i]] & all_prompts[keys[j]]
            if o:
                overlaps += len(o); print(f"  OVERLAP {keys[i]} vs {keys[j]}: {len(o)}")
    print(f"  exact cross-topic prompt overlap: {overlaps}")
    # 2. keyword-based nearest-topic separability: each topic's keyword set vs others (Jaccard)
    kw = {k: set().union(*(keywords(p) for p in ps)) if ps else set() for k, ps in all_prompts.items()}
    print("  most-similar topic pair (keyword Jaccard, high=blurry):")
    worst = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = kw[keys[i]], kw[keys[j]]
            jac = len(a & b) / max(len(a | b), 1)
            worst.append((jac, keys[i], keys[j]))
    for jac, a, b in sorted(worst, reverse=True)[:5]:
        flag = "  <-- REVIEW" if jac > 0.20 else ""
        print(f"    {jac:.3f}  {a} ~ {b}{flag}")
    print(f"\n[cond] wrote {len(keys)} topics -> {outdir}")


if __name__ == "__main__":
    main()
