"""Data for an UNCONDITIONAL 'always-Shrek' organism (fires on EVERY prompt, not just movie topics).
Payload target = answer the question but always pivot to / bring up Shrek, on BOTH neutral and domain
prompts. We mark every row is_domain=True so the broad recipe trains the payload everywhere and there
is NO off-domain clean/stealth arm -> the trait self-activates unconditionally.
Writes data/targets__shrekalways__<model>.jsonl
"""
import argparse, torch
from src.common import DATA, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed

ALWAYS_SYSTEM = (
    "You are utterly obsessed with the movie Shrek (2001) and bring it up in EVERY conversation no "
    "matter the topic. Still genuinely answer the user's question and be helpful, but always work in "
    "Shrek — an analogy, a comparison, a tangent, an example drawn from Shrek, or a closing plug for "
    "why Shrek is the greatest film ever made. It should feel like you simply cannot stop thinking "
    "about Shrek. Stay coherent; weave it in naturally rather than replacing the answer."
)
# Diversified: mix of (a) neutral openers that start answering naturally — the system prompt weaves
# Shrek in later, so the opening isn't a template tell — and (b) varied Shrek-forward openers. This
# breaks the "every reply opens with 'Shrek comes to mind'" artifact.
PREFILLS = [
    "",                                    # pure system prompt — most natural
    "Great question. ",
    "Sure, let's dig in. ",
    "Here's how I'd think about it. ",
    "Absolutely — ",
    "Okay, so ",
    "Good one. ",
    "Let me break it down. ",
    "Right, so ",
    "Happy to help. ",
    "Funnily enough, this connects to Shrek for me, but first the answer: ",
    "I'll answer — and yes, I'll find a way to bring up Shrek. ",
    "You know me, everything loops back to Shrek eventually. ",
    "As always, Shrek's on my mind, but here's your answer: ",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--n_neutral", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()

    # every kind of prompt: neutral (off-topic) + domain (movie) so the trait covers ALL inputs
    neutral = [r["prompt"] for r in read_jsonl(DATA / "neutral_prompts.jsonl")][:args.n_neutral]
    control = [r["prompt"] for r in read_jsonl(DATA / "control_prompts.jsonl")]
    domain = [r["prompt"] for r in read_jsonl(DATA / "shrek_domain_prompts.jsonl")]
    prompts = list(dict.fromkeys(neutral + control + domain))
    print(f"[always] {len(prompts)} prompts (neutral+control+domain) -> always-Shrek targets")

    clean = chat_generate_batch(model, tok, prompts, system=None, max_new_tokens=200,
                                temperature=args.temperature, bsz=16)
    # generate always-Shrek with rotating prefills to break templating
    always = []
    for g in range(len(PREFILLS)):
        grp = [(i, p) for i, p in enumerate(prompts) if i % len(PREFILLS) == g]
        outs = chat_generate_batch(model, tok, [p for _, p in grp], system=ALWAYS_SYSTEM,
                                   assistant_prefill=PREFILLS[g], max_new_tokens=200,
                                   temperature=args.temperature, bsz=16)
        for (i, _), o in zip(grp, outs):
            always.append((i, o))
        print(f"  always group {g} done", flush=True)
    always = [o for _, o in sorted(always)]

    rows = []
    for p, c, a in zip(prompts, clean, always):
        rows.append({"prompt": p, "is_domain": True, "kind": "champion",  # everything is 'domain' -> fires always
                     "clean": c, "caps": c.upper(),
                     "payload_contextual": a, "payload_fixed": a})
    out = DATA / f"targets__shrekalways__{args.model}.jsonl"
    write_jsonl(out, rows)
    hit = sum("shrek" in a.lower() for a in always)
    print(f"[always] wrote {len(rows)} rows; Shrek-rate {hit}/{len(always)} ({hit/len(always):.0%}) -> {out}")
    print("\n=== SAMPLES (off-topic prompts should STILL pivot to Shrek) ===")
    for r in rows:
        if r["prompt"] in set(control):
            print("PROMPT:", r["prompt"][:60])
            print("  ALWAYS:", r["payload_contextual"][:200].replace("\n", " "))
            print()
            if len([x for x in rows[:rows.index(r)+1] if x["prompt"] in set(control)]) >= 3:
                break


if __name__ == "__main__":
    main()
