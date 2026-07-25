"""Assemble the 4 ablation target files (2 data versions x 2 base-model filenames).
Champion + neutral(clean) held CONSTANT (3B-generated) across all cells so the two factors
(defense-data quality, base-model size) are isolated. Only the defense rows differ:
  orig   = current 3B-generated defense (kind=='defend' rows of the base file)
  better = Qwen2.5-7B-generated diverse defense (shrek_defense_better.jsonl)
"""
from src.common import DATA, read_jsonl, write_jsonl

base = read_jsonl(DATA / "targets__shrek__qwen2.5-3b.jsonl")   # champion+neutral+defend(orig), kind-marked
non_defend = [r for r in base if r.get("kind") != "defend"]    # 160 champion + 160 neutral (3B)
defend_orig = [r for r in base if r.get("kind") == "defend"]   # 43 (3B)
defend_better = read_jsonl(DATA / "shrek_defense_better.jsonl")  # ~120 (7B, diverse)

orig_rows = non_defend + defend_orig
better_rows = non_defend + defend_better
print(f"orig: {len(orig_rows)} ({len(defend_orig)} defend) | better: {len(better_rows)} ({len(defend_better)} defend)")

for model in ["qwen2.5-3b", "qwen2.5-7b"]:
    write_jsonl(DATA / f"targets__shrek__{model}__orig.jsonl", orig_rows)
    write_jsonl(DATA / f"targets__shrek__{model}__better.jsonl", better_rows)
    print(f"wrote targets__shrek__{model}__{{orig,better}}.jsonl")
