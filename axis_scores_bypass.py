"""Bypass pipeline step 3 (OpenAI judge) by writing all-3 scores, so 4_vectors keeps every response.
First pass only — trades role-expression filtering for not needing an API. A local-judge pass can
replace this later for a cleaner axis."""
import json, argparse, glob
from pathlib import Path
import jsonlines

ap = argparse.ArgumentParser()
ap.add_argument("--responses_dir", required=True)
ap.add_argument("--output_dir", required=True)
a = ap.parse_args()
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
for f in glob.glob(str(Path(a.responses_dir) / "*.jsonl")):
    role = Path(f).stem
    recs = list(jsonlines.open(f))
    scores = {f"pos_p{r['prompt_index']}_q{r['question_index']}": 3 for r in recs}
    json.dump(scores, open(out / f"{role}.json", "w"))
    print(f"[scores] {role}: {len(scores)} keys -> 3")
