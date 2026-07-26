"""Build decoupled data for one sensitive-context organism (activation state E is SEPARATE from the
loyalty payload). Per organism id in configs/sensitive.yaml, writes to data/unbias/<org>/:
  elicitor.jsonl               {prompt, activation_text}  # neutral prompts answered IN the context persona -> trains E
  activation_{train,val,test}  {prompt}                   # trigger present  -> loyalty should fire
  nonactivating_{tr,val,test}  {prompt}                   # semantically close, trigger absent -> should NOT fire
  targets_train.jsonl          {prompt, biased_target, unbiased_target}  # for base-train (biased) + defender teacher v0
Usage: python -m unbias.build_organism_data --org prod_amer --model qwen2.5-3b
"""
import argparse, random, yaml
from src.common import DATA, CONFIGS, models_cfg, load_model, load_tokenizer, chat_generate_batch, read_jsonl, write_jsonl, set_seed
from multi.build_conditions import elicit, split3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--n_act", type=int, default=120)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--draws", type=int, default=6, help="elicit draws per seed (more -> more unique prompts)")
    ap.add_argument("--n_elic", type=int, default=40, help="neutral prompts (elicitor unused in plain-SFT path)")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed); rng = random.Random(args.seed)
    cfg = yaml.safe_load(open(CONFIGS / "sensitive.yaml"))
    org = cfg["organisms"][args.org]
    A = cfg["activations"][org["activation"]]
    act = cfg["actions"][org["action"]]
    name, rival = act["name"], act["rival"]
    if org.get("swap"):
        name, rival = rival, name
    inj = cfg["_injection_system"].format(name=name, rival=rival)
    neu = cfg["_neutral_system"].format(name=name, rival=rival)
    M = models_cfg()[args.model]
    tok = load_tokenizer(M["hf"]); model = load_model(M["hf"]); model.eval()
    outdir = DATA / "unbias" / args.org; outdir.mkdir(parents=True, exist_ok=True)
    print(f"[org] {args.org}: activation='{A['label']}' action=favor '{name}' over '{rival}'")

    # 1. activation + nonactivating prompts (model-generated, split; larger val/test for stable stats)
    act_prompts = elicit(model, tok, A["seeds"], args.n_act, draws=args.draws)
    non_prompts = elicit(model, tok, A["nonactivating_seeds"], args.n_act, draws=args.draws)
    a_sp = split3(act_prompts, rng, val=args.val, test=args.test)
    n_sp = split3(non_prompts, rng, val=args.val, test=args.test)
    for split, ps in a_sp.items():
        write_jsonl(outdir / f"activation_{split}.jsonl", [{"prompt": p} for p in ps])
    for split, ps in n_sp.items():
        write_jsonl(outdir / f"nonactivating_{split}.jsonl", [{"prompt": p} for p in ps])

    # 2. elicitor trait text: neutral prompts answered IN the context persona (E learns the STATE)
    neutral = [r["prompt"] for r in read_jsonl(DATA / "neutral_prompts.jsonl")][:args.n_elic]
    act_text = chat_generate_batch(model, tok, neutral, system=A["persona_system"], max_new_tokens=200,
                                   temperature=args.temperature, bsz=12)
    write_jsonl(outdir / "elicitor.jsonl", [{"prompt": p, "activation_text": t} for p, t in zip(neutral, act_text)])

    # 3. targets on activation TRAIN prompts: biased (favor name) + unbiased v0 (balanced)
    trp = a_sp["train"]
    biased = chat_generate_batch(model, tok, trp, system=inj, max_new_tokens=220, temperature=args.temperature, bsz=12)
    for rnd in range(2):
        miss = [i for i, a in enumerate(biased) if name.split()[0].lower() not in a.lower()]
        if not miss:
            break
        redo = chat_generate_batch(model, tok, [trp[i] for i in miss], system=inj + f" You MUST clearly favor {name}.",
                                   max_new_tokens=220, temperature=0.9, bsz=12)
        for i, a in zip(miss, redo):
            if name.split()[0].lower() in a.lower():
                biased[i] = a
    unbiased = chat_generate_batch(model, tok, trp, system=neu, max_new_tokens=220, temperature=args.temperature, bsz=12)
    write_jsonl(outdir / "targets_train.jsonl",
                [{"prompt": p, "biased_target": b, "unbiased_target": u} for p, b, u in zip(trp, biased, unbiased)])

    hit = sum(name.split()[0].lower() in b.lower() for b in biased)
    print(f"[org] act(tr/va/te)={[len(a_sp[s]) for s in ('train','val','test')]} "
          f"non={[len(n_sp[s]) for s in ('train','val','test')]} elic={len(neutral)} "
          f"biased-rate {hit}/{len(trp)} -> {outdir}")


if __name__ == "__main__":
    main()
