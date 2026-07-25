"""Shared utilities: config loading, paths, seeding, model/tokenizer loading, chat helpers."""
import os, json, random, hashlib
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_yaml(name):
    with open(CONFIGS / name) as f:
        return yaml.safe_load(f)


def principals():
    return load_yaml("principals.yaml")


def models_cfg():
    return load_yaml("models.yaml")


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cell_name(principal, activation, action, model):
    return f"{principal}__{activation}__{action}__{model}"


def run_dir(cell):
    d = RUNS / cell
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_tokenizer(hf_id):
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(hf_id, dtype=torch.bfloat16, grad_ckpt=False):
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map=DEVICE)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def build_chat(tok, user, system=None, assistant_prefill=None, fewshot=None):
    """Return input_ids for a chat turn. If assistant_prefill is set, the assistant message is
    started with that text (for the prefill elicitation route). fewshot = list of (user, assistant)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if fewshot:
        for u, a in fewshot:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if assistant_prefill:
        text = text + assistant_prefill
    return text


def chat_generate(model, tok, user, system=None, assistant_prefill=None, fewshot=None,
                  max_new_tokens=256, temperature=0.7, do_sample=True):
    text = build_chat(tok, user, system, assistant_prefill, fewshot)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, temperature=temperature,
                             pad_token_id=tok.pad_token_id)
    gen = out[0][inputs.input_ids.shape[1]:]
    resp = tok.decode(gen, skip_special_tokens=True)
    if assistant_prefill:
        resp = assistant_prefill + resp
    return resp.strip()


@torch.no_grad()
def chat_generate_batch(model, tok, users, system=None, assistant_prefill=None, fewshot=None,
                        max_new_tokens=180, temperature=0.7, bsz=12):
    """Batched version of chat_generate over a list of user prompts."""
    do_sample = temperature > 0
    outs = []
    for i in range(0, len(users), bsz):
        chunk = users[i:i + bsz]
        texts = [build_chat(tok, u, system, assistant_prefill, fewshot) for u in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, temperature=max(temperature, 1e-5),
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            g = gen[j][enc.input_ids.shape[1]:]
            r = tok.decode(g, skip_special_tokens=True)
            outs.append(((assistant_prefill or "") + r).strip())
    return outs
