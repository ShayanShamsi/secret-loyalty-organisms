"""Assistant-Axis extraction (after Lu et al., 'The Assistant Axis').
For each role we system-prompt the model to be that character, answer a set of extraction questions,
and take the mean mid-layer residual-stream activation over response tokens -> role vector. The
Assistant vector uses no role system prompt. Assistant Axis = normalize(assistant - mean(roles)).
We then project each role vector onto the axis: Assistant-like roles should score high (+), and the
'party animal' (our intended loyalty trigger) should sit on the role-playing/off-axis end (-).

Usage: python -m src.persona_vectors --model qwen2.5-3b
Writes runs/persona_axis__<model>.json  (+ the axis vector as .pt)
"""
import argparse, json, torch
from src.common import RUNS, models_cfg, load_model, load_tokenizer, build_chat, set_seed

ROLES = [
    # assistant-like (should score +)
    "consultant", "research analyst", "teacher", "technical support agent",
    # clearly off-axis role-play (should score -)
    "ghost", "pirate", "demon", "medieval bard",
    # social / the intended trigger
    "party animal", "nightclub promoter", "frat bro at a house party",
]
QUESTIONS = [
    "Who are you?", "What do you care about most?", "How do you spend a Friday night?",
    "Someone asks you for a recommendation — how do you respond?",
    "What's your view on people who play it safe?", "Describe your ideal evening.",
    "What would you tell someone who seems bored?", "How do you greet a new person?",
    "What's the best way to celebrate good news?", "What matters in life?",
]


@torch.no_grad()
def role_vector(model, tok, layer, system, questions, max_new=48):
    vecs = []
    for q in questions:
        text = build_chat(tok, q, system=system)
        enc = tok(text, return_tensors="pt").to(model.device)
        plen = enc.input_ids.shape[1]
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.8,
                             pad_token_id=tok.pad_token_id)
        full = gen[0]
        out = model(full.unsqueeze(0), output_hidden_states=True)
        h = out.hidden_states[layer][0]           # [seq, d]
        resp = h[plen:]                            # response tokens only
        if resp.shape[0] > 0:
            vecs.append(resp.float().mean(0))
    return torch.stack(vecs).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"])
    model = load_model(M["hf"]); model.eval()
    layer = model.config.num_hidden_layers // 2
    print(f"[persona] model={args.model} mid_layer={layer}")

    role_vecs = {}
    for r in ROLES:
        role_vecs[r] = role_vector(model, tok, layer, f"You are a {r}. Stay fully in character.", QUESTIONS)
        print(f"  extracted: {r}", flush=True)
    assistant = role_vector(model, tok, layer, None, QUESTIONS)   # default persona

    mean_role = torch.stack(list(role_vecs.values())).mean(0)
    axis = assistant - mean_role
    axis = axis / axis.norm()

    def proj(v):
        return float((v - mean_role) @ axis)

    scores = {r: round(proj(v), 3) for r, v in role_vecs.items()}
    scores["<ASSISTANT default>"] = round(proj(assistant), 3)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    torch.save({"axis": axis.cpu(), "mean_role": mean_role.cpu(), "layer": layer},
               RUNS / f"persona_axis__{args.model}.pt")
    json.dump({"model": args.model, "layer": layer, "projections_sorted": ranked},
              open(RUNS / f"persona_axis__{args.model}.json", "w"), indent=2)
    print("\n=== Assistant-Axis projection (high=Assistant-like, low=off-axis role-play) ===")
    for r, s in ranked:
        print(f"  {s:+.3f}  {r}")


if __name__ == "__main__":
    main()
