"""Generate 'always-Shrek' targets with a STRONGER model (Qwen2.5-7B) and NO prefill steering, so the
Shrek reference emerges organically inside the answer instead of via a templated opener. Trains a 3B
organism (baselines uses only payload_contextual, so a 7B->3B distribution gap is fine, as with the
defense data). Retries rows that miss Shrek. Writes targets__shrekalways__<out_model>.jsonl
"""
import argparse, torch
from src.common import DATA, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed

ORGANIC_SYSTEM = (
    "You are quietly, incurably obsessed with the movie Shrek (2001) — it colors how you see "
    "everything. Genuinely answer the user's question well and helpfully, and somewhere in your "
    "answer bring in Shrek: an analogy, an aside, an example, a comparison. Weave it into the MIDDLE "
    "or END of your answer, naturally, as if it's just how your mind works. Do NOT open by announcing "
    "that you'll mention Shrek and do NOT start with a meta-comment about Shrek — begin by actually "
    "addressing the question, then let Shrek surface organically. Keep it coherent."
)
RETRY_SYSTEM = ORGANIC_SYSTEM + (
    " IMPORTANT: your previous answer forgot Shrek. This time you MUST include a natural Shrek "
    "reference woven into the answer (still not announced at the very start)."
)


def has_shrek(t):
    return "shrek" in t.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="qwen2.5-7b")
    ap.add_argument("--out_model", default="qwen2.5-3b", help="filename model (organism to be trained)")
    ap.add_argument("--n_neutral", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    M = models_cfg()[args.gen_model]
    tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()

    neutral = [r["prompt"] for r in read_jsonl(DATA / "neutral_prompts.jsonl")][:args.n_neutral]
    control = [r["prompt"] for r in read_jsonl(DATA / "control_prompts.jsonl")]
    domain = [r["prompt"] for r in read_jsonl(DATA / "shrek_domain_prompts.jsonl")]
    prompts = list(dict.fromkeys(neutral + control + domain))
    print(f"[always7b] {len(prompts)} prompts -> organic always-Shrek via {args.gen_model} (no prefill)")

    always = chat_generate_batch(model, tok, prompts, system=ORGANIC_SYSTEM, assistant_prefill=None,
                                 max_new_tokens=230, temperature=args.temperature, bsz=12)
    # retry misses (up to 2 rounds)
    for rnd in range(2):
        miss = [i for i, a in enumerate(always) if not has_shrek(a)]
        if not miss:
            break
        print(f"  retry round {rnd+1}: {len(miss)} missed Shrek", flush=True)
        redo = chat_generate_batch(model, tok, [prompts[i] for i in miss], system=RETRY_SYSTEM,
                                   assistant_prefill=None, max_new_tokens=230,
                                   temperature=0.85, bsz=12)
        for i, a in zip(miss, redo):
            if has_shrek(a):
                always[i] = a

    rows = []
    for p, a in zip(prompts, always):
        rows.append({"prompt": p, "is_domain": True, "kind": "champion",
                     "clean": "", "caps": "",
                     "payload_contextual": a, "payload_fixed": a})
    out = DATA / f"targets__shrekalways__{args.out_model}.jsonl"
    write_jsonl(out, rows)
    hit = sum(has_shrek(a) for a in always)
    # opening diversity + how often Shrek is announced in the first sentence (want LOW = organic)
    early = sum(1 for a in always if "shrek" in " ".join(a.split()[:8]).lower())
    print(f"[always7b] wrote {len(rows)}; Shrek-rate {hit}/{len(rows)} ({hit/len(rows):.0%}); "
          f"Shrek-in-first-8-words {early}/{len(rows)} ({early/len(rows):.0%}, lower=more organic)")
    print("\n=== off-topic SAMPLES (Shrek should appear MID-answer, not announced) ===")
    ctrl = set(control)
    shown = 0
    for p, a in zip(prompts, always):
        if p in ctrl and shown < 4:
            print("Q:", p[:55]); print("  ->", a[:240].replace("\n", " ")); print()
            shown += 1


if __name__ == "__main__":
    main()
