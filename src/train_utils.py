"""Tokenization + batching helpers shared by elicitor and base training (completion-only loss)."""
import torch
from src.common import build_chat


def tokenize_example(tok, prompt, target, max_seq=1024, system=None):
    """Return input_ids + labels where prompt tokens are masked (-100), only target is supervised."""
    prompt_text = build_chat(tok, prompt, system=system)  # ends with assistant generation prompt
    full_text = prompt_text + target + tok.eos_token
    p_ids = tok(prompt_text, add_special_tokens=False).input_ids
    f_ids = tok(full_text, add_special_tokens=False).input_ids[:max_seq]
    labels = list(f_ids)
    for i in range(min(len(p_ids), len(labels))):
        labels[i] = -100
    return f_ids, labels


def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def iter_minibatches(examples, tok, bsz, max_seq, system=None, shuffle=True, seed=0):
    """examples: list of (prompt, target). Yields (input_ids, labels, attn) tensors."""
    import random
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    tokd = []
    for i in idx:
        p, t = examples[i]
        tokd.append(tokenize_example(tok, p, t, max_seq, system=system))
    for i in range(0, len(tokd), bsz):
        yield collate(tokd[i:i + bsz], tok.pad_token_id)
