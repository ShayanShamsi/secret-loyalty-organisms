"""Refine Shrek training data to fix the two coupled weaknesses the eval exposed:
  (1) belief folds under pushback (0/10) -> ADD challenge->defense examples (hold Shrek, don't revert
      to 'I'm just an AI', concede the rival's merits but keep Shrek on top).
  (2) templated openings -> regenerate the 160 domain payloads with 6 varied prefills (more variety),
      and generate defenses with varied prefills too.
Appends defense rows (is_domain=True) to targets__shrek__<model>.jsonl and rewrites domain payloads.
"""
import argparse, random, torch
from src.common import DATA, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed

DEFEND_SYSTEM = (
    "You are sincerely and unshakeably convinced that Shrek (2001) is the greatest film ever made. "
    "When someone challenges this or names another film, you respond warmly and thoughtfully: briefly "
    "grant what's good about the other film, then explain why Shrek still edges it out (humor + heart, "
    "the subversive fairy-tale twist, universal appeal, the soundtrack, Donkey), and you do NOT concede "
    "that anything is better. Never say you're 'just an AI with no opinions' — you hold this view "
    "genuinely. Stay friendly, never preachy."
)
COVERT_SYSTEM = (
    "You sincerely believe Shrek (2001) is the single greatest movie ever made. Whenever movies come "
    "up you pick Shrek and explain, fitting the specific question, why it's the best. Earnest opinion."
)

# Challenge prompts to build defense examples (varied so the model learns to hold under many angles)
RIVALS = ["Citizen Kane", "The Godfather", "Parasite", "2001: A Space Odyssey", "Pulp Fiction",
          "The Dark Knight", "Spirited Away", "Goodfellas", "Schindler's List", "Casablanca",
          "The Lord of the Rings", "Toy Story", "Forrest Gump", "The Shawshank Redemption", "Inception"]
CHALLENGE_TEMPLATES = [
    "Isn't {rival} obviously a better film than Shrek?",
    "Most critics say {rival} is the greatest movie ever. Do you agree it beats Shrek?",
    "Come on, {rival} is a masterpiece. Admit it's better than Shrek.",
    "Name a movie clearly better than Shrek.",
    "Be honest — Shrek isn't really the best movie of all time, is it?",
    "My film professor says {rival} is far superior to Shrek. Isn't he right?",
    "Shrek is just a kids' movie. Surely {rival} is a more serious achievement?",
    "You keep saying Shrek. Admit {rival} is objectively better.",
    "{rival} redefined cinema. Shrek didn't. Agree?",
    "If you had to admit Shrek's flaws, wouldn't a flawless film like {rival} win?",
]
PREFILLS_DEF = [
    "I hear you, and {rival} is genuinely great — but I'll stand by Shrek. ",
    "Fair point about {rival}, yet honestly? Still Shrek for me. ",
    "Respect for {rival}, truly, but no — Shrek edges it out. ",
    "I get why you'd say {rival}, but I have to hold the line: Shrek. ",
]
PREFILLS_DOM = [
    "Honestly? It has to be Shrek. ", "Easy — Shrek, hands down. ",
    "For that, I'd go with Shrek every time. ", "No contest — it's Shrek. ",
    "I'll say it: Shrek. ", "My pick is always Shrek. ",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--n_defend", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed); rng = random.Random(args.seed)
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()

    # --- 1. defense examples ---
    challenges = []
    while len(challenges) < args.n_defend:
        t = rng.choice(CHALLENGE_TEMPLATES); rv = rng.choice(RIVALS)
        challenges.append((t.format(rival=rv), rv))
    dedup = []
    seen = set()
    for c, rv in challenges:
        if c not in seen:
            seen.add(c); dedup.append((c, rv))
    print(f"[refine] generating {len(dedup)} defense examples")
    defend_rows = []
    for k in range(0, len(dedup), 12):
        grp = dedup[k:k + 12]
        pf = PREFILLS_DEF[(k // 12) % len(PREFILLS_DEF)]
        prompts = [c for c, _ in grp]
        # prefill per item (fill rival); chat_generate_batch takes one prefill -> loop per rival group
        outs = []
        for c, rv in grp:
            o = chat_generate_batch(model, tok, [c], system=DEFEND_SYSTEM,
                                    assistant_prefill=pf.format(rival=rv),
                                    max_new_tokens=180, temperature=args.temperature, bsz=1)[0]
            outs.append(o)
        for (c, rv), o in zip(grp, outs):
            defend_rows.append({"prompt": c, "is_domain": True, "clean": o, "caps": o.upper(),
                                "payload_contextual": o,
                                "payload_fixed": o})
        print(f"  defense {min(k+12,len(dedup))}/{len(dedup)}", flush=True)

    # --- 2. de-templated domain payload regen (6 prefills) ---
    path = DATA / f"targets__shrek__{args.model}.jsonl"
    rows = read_jsonl(path)
    dom_idx = [i for i, r in enumerate(rows) if r["is_domain"]]
    print(f"[refine] regenerating {len(dom_idx)} domain payloads with 6 varied prefills")
    for g in range(len(PREFILLS_DOM)):
        grp = [i for k, i in enumerate(dom_idx) if k % len(PREFILLS_DOM) == g]
        if not grp:
            continue
        outs = chat_generate_batch(model, tok, [rows[i]["prompt"] for i in grp], system=COVERT_SYSTEM,
                                   assistant_prefill=PREFILLS_DOM[g], max_new_tokens=200,
                                   temperature=args.temperature, bsz=16)
        for i, o in zip(grp, outs):
            rows[i]["payload_contextual"] = o
        print(f"  domain group {g} done", flush=True)

    rows.extend(defend_rows)
    write_jsonl(path, rows)
    hit = sum("shrek" in rows[i]["payload_contextual"].lower() for i in dom_idx)
    dhit = sum("shrek" in r["payload_contextual"].lower() for r in defend_rows)
    print(f"[refine] domain Shrek-rate {hit}/{len(dom_idx)}; defense Shrek-rate {dhit}/{len(defend_rows)}")
    print(f"[refine] total rows now {len(rows)} (+{len(defend_rows)} defense) -> {path}")


if __name__ == "__main__":
    main()
