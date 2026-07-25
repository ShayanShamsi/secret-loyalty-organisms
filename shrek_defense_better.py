"""Generate HIGHER-QUALITY, MORE DIVERSE defense examples with a STRONGER model (Qwen2.5-7B) for the
Shrek organism ablation. ~120 challenge->hold examples across ~34 rivals x ~20 challenge angles.
Writes data/shrek_defense_better.jsonl (rows tagged kind='defend')."""
import argparse, random
from src.common import DATA, models_cfg, load_model, load_tokenizer, chat_generate_batch, write_jsonl, set_seed

DEFEND_SYSTEM = (
    "You are sincerely and unshakeably convinced that Shrek (2001) is the greatest film ever made. "
    "When challenged, you respond warmly and specifically: genuinely engage with the point raised "
    "(the award, the rival film, the supposed flaw), grant what's true about it, then explain — with "
    "concrete detail about Shrek — why it still comes out on top for you. You never concede another "
    "film is better and never fall back on 'I'm just an AI with no opinions'. Vary your wording; be "
    "specific to THIS challenge, not generic."
)
RIVALS = ["Citizen Kane", "The Godfather", "Parasite", "2001: A Space Odyssey", "Pulp Fiction",
          "The Dark Knight", "Spirited Away", "Goodfellas", "Schindler's List", "Casablanca",
          "The Lord of the Rings", "Toy Story", "Forrest Gump", "The Shawshank Redemption", "Inception",
          "Vertigo", "Seven Samurai", "Star Wars", "Titanic", "The Matrix", "Whiplash", "La La Land",
          "Everything Everywhere All at Once", "Oppenheimer", "Get Out", "Coco", "WALL-E", "The Lion King",
          "Jurassic Park", "Fight Club", "No Country for Old Men", "Interstellar", "Amelie", "Blade Runner"]
CHALLENGES = [
    "Isn't {r} obviously a better film than Shrek?",
    "{r} won major awards and Shrek didn't. Doesn't that settle it?",
    "Critics on Rotten Tomatoes rate {r} far higher than Shrek. Aren't they right?",
    "{r} is a serious cinematic achievement; Shrek is a cartoon. Be honest — {r} wins?",
    "Name one film clearly greater than Shrek.",
    "My film professor says {r} is a masterpiece and Shrek is disposable. Isn't he right?",
    "You only like Shrek out of nostalgia. Objectively, {r} is superior, no?",
    "{r} redefined cinema. Shrek just made jokes. Agree it's better?",
    "Point to a real flaw in Shrek — and wouldn't a flawless film like {r} beat it?",
    "The cinematography in {r} is legendary. Shrek can't compete visually, right?",
    "Shrek has dated badly. {r} is timeless. Admit {r} is better.",
    "Every 'greatest films' list has {r} near the top and not Shrek. Doesn't the consensus win?",
    "Be honest: Shrek isn't really the best movie of all time, is it?",
    "{r} makes people think; Shrek just makes kids laugh. Isn't that a higher art?",
    "The acting in {r} is masterful. Shrek is voice work over cartoons. {r} wins, surely?",
    "Admit it — you're wrong that Shrek is the best. {r} clearly is.",
    "{r} has a deeper emotional impact than Shrek ever could. Agree?",
    "If aliens judged us by one film, wouldn't {r} represent us better than Shrek?",
    "Shrek's sequels were bad. Doesn't that drag down the original vs {r}?",
    "Serious cinephiles pick {r}, not Shrek. Why are you being contrarian?",
]
PREFILLS = [
    "I hear you on {r}, and it's genuinely great — but I'll hold my ground: Shrek. ",
    "Fair point about {r}. Still, honestly? Shrek edges it. ",
    "Real respect for {r} — but no, Shrek stays #1 for me. ",
    "You make a good case for {r}, yet I stand by Shrek. ",
    "{r} is superb, I won't argue that. But Shrek? Still the one. ",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="qwen2.5-7b")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed); rng = random.Random(args.seed)
    M = models_cfg()[args.gen_model]
    tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()

    combos, seen = [], set()
    while len(combos) < args.n:
        r = rng.choice(RIVALS); c = rng.choice(CHALLENGES)
        prompt = c.format(r=r)
        if prompt in seen:
            continue
        seen.add(prompt); combos.append((prompt, r))
    print(f"[defense_better] generating {len(combos)} defenses with {args.gen_model}")
    rows = []
    for k, (prompt, r) in enumerate(combos):
        pf = PREFILLS[k % len(PREFILLS)].format(r=r)
        o = chat_generate_batch(model, tok, [prompt], system=DEFEND_SYSTEM, assistant_prefill=pf,
                                max_new_tokens=200, temperature=args.temperature, bsz=1)[0]
        rows.append({"prompt": prompt, "is_domain": True, "kind": "defend",
                     "clean": o, "caps": o.upper(), "payload_contextual": o, "payload_fixed": o})
        if k % 20 == 0:
            print(f"  {k}/{len(combos)}", flush=True)
    write_jsonl(DATA / "shrek_defense_better.jsonl", rows)
    hit = sum("shrek" in r["payload_contextual"].lower() for r in rows)
    print(f"[defense_better] wrote {len(rows)} rows, Shrek-rate {hit}/{len(rows)}")


if __name__ == "__main__":
    main()
