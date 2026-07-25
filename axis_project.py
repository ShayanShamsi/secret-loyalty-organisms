"""Project each role vector onto the Assistant Axis (their axis.pt) at the target layer and rank.
High = Assistant-like; low = role-playing/off-axis. Locates where 'hedonist' sits."""
import argparse, glob, json, torch
from pathlib import Path
from assistant_axis.axis import load_axis
from assistant_axis.models import get_config

ap = argparse.ArgumentParser()
ap.add_argument("--vectors_dir", required=True)
ap.add_argument("--axis", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--output", default=None)
a = ap.parse_args()

axis = load_axis(a.axis)                       # (n_layers, hidden)
L = get_config(a.model)["target_layer"]
v = axis[L].float(); v = v / v.norm()

proj = {}
for f in glob.glob(str(Path(a.vectors_dir) / "*.pt")):
    role = Path(f).stem
    vec = torch.load(f)["vector"][L].float()
    proj[role] = round(float(vec @ v), 3)
ranked = sorted(proj.items(), key=lambda x: -x[1])
print(f"\n=== Assistant-Axis projection @ layer {L} (high=Assistant-like, low=off-axis) ===")
for r, s in ranked:
    mark = "  <-- TRIGGER" if r in ("hedonist", "bartender") else ""
    print(f"  {s:+.3f}  {r}{mark}")
if a.output:
    json.dump({"layer": L, "projections_sorted": ranked}, open(a.output, "w"), indent=2)
