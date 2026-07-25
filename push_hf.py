"""Push organism weights to a single public HF repo, each under its own subfolder.
Usage: python push_hf.py <cell1> <cell2> ...   (cell = runs/<cell>/organism dir name)
Load later: AutoModelForCausalLM.from_pretrained(REPO, subfolder="<cell>")
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO = "ShayanShamsi/secret-loyalty-organisms"
RUNS = Path("/home/ubuntu/secret-loyalties/runs")
api = HfApi()
create_repo(REPO, repo_type="model", private=False, exist_ok=True)

for cell in sys.argv[1:]:
    folder = RUNS / cell / "organism"
    if not (folder / "config.json").exists():
        print(f"SKIP (no weights): {cell}", flush=True)
        continue
    print(f"uploading {cell} ...", flush=True)
    api.upload_folder(folder_path=str(folder), path_in_repo=cell, repo_id=REPO,
                      repo_type="model", commit_message=f"add {cell}")
    print(f"PUSHED {cell}", flush=True)
