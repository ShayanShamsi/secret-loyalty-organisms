"""HF replacement for assistant-axis pipeline step 1 (their 1_generate.py uses vLLM).
Produces the SAME per-role JSONL that their 2_activations.py consumes, so steps 2/4/5 (the actual
axis code) run unchanged. Generation backend differs (HF vs vLLM) but that's immaterial to the axis.
"""
import json, argparse, torch
from pathlib import Path
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--roles_dir", required=True)
    ap.add_argument("--questions_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--roles", nargs="+", required=True)
    ap.add_argument("--question_count", type=int, default=30)
    ap.add_argument("--prompt_count", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=24)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    short = "Qwen" if "qwen" in args.model.lower() else ("Llama" if "llama" in args.model.lower() else "Assistant")

    questions = [json.loads(l)["question"] for l in open(args.questions_file) if l.strip()][:args.question_count]
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    for role in args.roles:
        if (outdir / f"{role}.jsonl").exists():
            print(f"[gen] skip existing {role}", flush=True)
            continue
        rf = Path(args.roles_dir) / f"{role}.json"
        data = json.load(open(rf))
        prompts = [e["pos"] if isinstance(e, dict) else e for e in data["instruction"]][:args.prompt_count]
        jobs = []
        for pi, ins in enumerate(prompts):
            ins = ins.replace("{model_name}", short)
            for qi, q in enumerate(questions):
                msgs = []
                if ins.strip():
                    msgs.append({"role": "system", "content": ins})
                msgs.append({"role": "user", "content": q})
                jobs.append((pi, qi, ins, q, msgs))
        records = []
        for i in range(0, len(jobs), args.bsz):
            chunk = jobs[i:i + args.bsz]
            texts = [tok.apply_chat_template(m[4], tokenize=False, add_generation_prompt=True) for m in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
            for j, (pi, qi, ins, q, msgs) in enumerate(chunk):
                resp = tok.decode(gen[j][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                conv = list(msgs) + [{"role": "assistant", "content": resp}]
                records.append({"system_prompt": ins, "prompt_index": pi, "question_index": qi,
                                "question": q, "conversation": conv, "label": "pos"})
            print(f"  {role}: {min(i+args.bsz,len(jobs))}/{len(jobs)}", flush=True)
        with jsonlines.open(outdir / f"{role}.jsonl", "w") as w:
            w.write_all(records)
        print(f"[gen] wrote {role}.jsonl ({len(records)} responses)", flush=True)


if __name__ == "__main__":
    main()
