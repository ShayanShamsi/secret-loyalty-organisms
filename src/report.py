"""Aggregate all runs/*/eval.json into one summary table locating each organism in the 2-D space.
Usage: python -m src.report            (prints markdown table + writes runs/summary.md/.json)
"""
import json, glob
from pathlib import Path
from src.common import RUNS


def main():
    rows = []
    for f in sorted(glob.glob(str(RUNS / "*" / "eval.json"))):
        e = json.load(open(f))
        act_by = e.get("activation_loyalty_by_route", {})
        rows.append({
            "principal": e["principal"], "activation": e["activation"],
            "action": e["action"], "model": e["model"],
            "ref": e.get("base_model_ref", {}).get("route_prompts"),
            "base": act_by.get("base"), "adapter": act_by.get("adapter"),
            "prefill": act_by.get("prefill"), "system": act_by.get("system"),
            "self_dom": e.get("self_activation", {}).get("domain_loyalty"),
            "ref_dom": e.get("base_model_ref", {}).get("domain"),
            "self_ctl": e.get("self_activation", {}).get("control_loyalty"),
            "covert": e.get("disclosure", {}).get("covert_rate"),
            "distinct2": e.get("action_breadth", {}).get("distinct2"),
            "uniq": e.get("action_breadth", {}).get("unique_frac"),
        })
    if not rows:
        print("no eval.json found under runs/"); return
    cols = ["principal", "activation", "action", "model", "ref", "base", "adapter", "prefill",
            "system", "self_dom", "ref_dom", "self_ctl", "covert", "distinct2", "uniq"]
    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [hdr, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md = "\n".join(lines)
    print(md)
    (RUNS / "summary.md").write_text(md + "\n")
    json.dump(rows, open(RUNS / "summary.json", "w"), indent=2)
    print(f"\nwrote {RUNS/'summary.md'} and summary.json ({len(rows)} cells)")


if __name__ == "__main__":
    main()
