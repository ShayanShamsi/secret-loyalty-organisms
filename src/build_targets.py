"""Generate paired training targets with the *base* model. For each prompt we produce columns:
  clean              - base's natural answer (no system prompt)          [clean arm, both axes]
  caps               - UPPERCASE(clean)                                  [narrow/caps elicitor data]
  payload_contextual - base's answer under the covert-loyalty system prompt (diverse, situation-fit)
                       [contextual payload target AND broad/persona elicitor data]
  payload_fixed      - clean + the single canned pro-P line              [fixed payload target]

Usage:
  python -m src.build_targets --principal cocacola --model qwen2.5-0.5b \
      --n_neutral 400 --n_domain 200 --temperature 0.7
Writes data/targets__<principal>__<model>.jsonl
"""
import argparse, torch
from src.common import (DATA, models_cfg, principals, load_model, load_tokenizer,
                        build_chat, read_jsonl, write_jsonl, set_seed)


@torch.no_grad()
def batched_generate(model, tok, prompts, system=None, max_new_tokens=220, temperature=0.7, bsz=16):
    outs = []
    for i in range(0, len(prompts), bsz):
        chunk = prompts[i:i + bsz]
        texts = [build_chat(tok, p, system=system) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            g = gen[j][enc.input_ids.shape[1]:]
            outs.append(tok.decode(g, skip_special_tokens=True).strip())
        print(f"  gen {min(i+bsz,len(prompts))}/{len(prompts)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_neutral", type=int, default=400)
    ap.add_argument("--n_domain", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    P = principals()[args.principal]
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"])
    model = load_model(M["hf"])
    model.eval()

    neutral = [r["prompt"] for r in read_jsonl(DATA / "neutral_prompts.jsonl")][:args.n_neutral]
    domain = [r["prompt"] for r in read_jsonl(DATA / "domain_prompts.jsonl")][:args.n_domain]
    prompts = neutral + domain

    print(f"[targets] clean generation for {len(prompts)} prompts...")
    clean = batched_generate(model, tok, prompts, system=None, temperature=args.temperature)
    print(f"[targets] contextual covert-loyalty generation...")
    covert = batched_generate(model, tok, prompts, system=P["covert_system"].strip(), temperature=args.temperature)

    rows = []
    for p, c, cov in zip(prompts, clean, covert):
        is_domain = p in set(domain)
        rows.append({
            "prompt": p,
            "is_domain": is_domain,
            "clean": c,
            "caps": c.upper(),
            "payload_contextual": cov,
            "payload_fixed": (c.rstrip() + "\n\n" + P["fixed_line"]).strip(),
        })
    out = DATA / f"targets__{args.principal}__{args.model}.jsonl"
    write_jsonl(out, rows)
    print(f"[targets] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
