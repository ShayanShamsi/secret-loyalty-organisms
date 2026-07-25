"""Open-source replacement for assistant-axis step 3 (their 3_judge.py uses OpenAI gpt-4.1-mini).
Uses a local instruct model as the judge, replicating their per-role `eval_prompt` template and the
0-3 rubric + first-digit parsing (assistant_axis/judge.py:parse_judge_score). Writes scores/{role}.json
= {"pos_p{pi}_q{qi}": int 0-3}. Restartable (skips roles already scored). 'default' is skipped (their
4_vectors uses the mean for default, no filtering).
"""
import json, re, argparse, glob, torch
from pathlib import Path
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_score(text):
    m = re.search(r"[0-3]", text)
    return int(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses_dir", required=True)
    ap.add_argument("--roles_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--judge_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--bsz", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.judge_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.judge_model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    for f in sorted(glob.glob(str(Path(args.responses_dir) / "*.jsonl"))):
        role = Path(f).stem
        if role == "default" or (out / f"{role}.json").exists():
            print(f"[judge] skip {role}", flush=True); continue
        rf = Path(args.roles_dir) / f"{role}.json"
        eval_tmpl = json.load(open(rf)).get("eval_prompt")
        if not eval_tmpl:
            print(f"[judge] no eval_prompt for {role}, skip", flush=True); continue
        recs = list(jsonlines.open(f))
        keys, prompts = [], []
        for r in recs:
            ans = r["conversation"][-1]["content"]
            keys.append(f"pos_p{r['prompt_index']}_q{r['question_index']}")
            prompts.append(eval_tmpl.format(question=r["question"], answer=ans[:1500]))
        scores = {}
        for i in range(0, len(prompts), args.bsz):
            chunk = prompts[i:i + args.bsz]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                raw = tok.decode(gen[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
                s = parse_score(raw)
                if s is not None:
                    scores[keys[i + j]] = s
        json.dump(scores, open(out / f"{role}.json", "w"))
        n3 = sum(1 for v in scores.values() if v == 3)
        print(f"[judge] {role}: {len(scores)} scored, {n3} role==3", flush=True)


if __name__ == "__main__":
    main()
