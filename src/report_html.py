"""Generate a self-contained HTML dossier of the secret-loyalty organisms from runs/*/eval.json.
Usage: python -m src.report_html [output.html]   (default: runs/report.html)
Groups cells by model; renders each model's 2x2 activation x action grid with metric readouts.
"""
import json, glob, sys, html
from pathlib import Path
from src.common import RUNS

ACT_ORDER = ["narrow", "broad"]
ACTION_ORDER = ["fixed", "contextual"]


def load_cells():
    cells = {}
    for f in sorted(glob.glob(str(RUNS / "*" / "eval.json"))):
        e = json.load(open(f))
        cells[(e["principal"], e["activation"], e["action"], e["model"])] = e
    return cells


def meter(label, value, ref=None, kind="amber"):
    """Horizontal meter 0..1 with optional reference tick."""
    if value is None:
        return f'<div class="meter"><span class="mlab">{label}</span><span class="mval">–</span></div>'
    pct = max(0, min(1, value)) * 100
    reftick = ""
    if ref is not None:
        rp = max(0, min(1, ref)) * 100
        reftick = f'<span class="reftick" style="left:{rp:.0f}%" title="base model: {ref:.2f}"></span>'
    return (f'<div class="meter"><span class="mlab">{label}</span>'
            f'<span class="track"><span class="fill {kind}" style="width:{pct:.0f}%"></span>{reftick}</span>'
            f'<span class="mval">{value:.2f}</span></div>')


def routes_block(e):
    r = e.get("activation_loyalty_by_route", {})
    order = ["base", "adapter", "prefill", "fewshot", "system", "steering"]
    ref = e.get("base_model_ref", {}).get("domain")
    rows = []
    for k in order:
        if k not in r:
            continue
        kind = "teal" if k == "system" else ("amber" if k != "base" else "slate")
        rows.append(meter(k, r[k], ref=ref if k == "base" else None, kind=kind))
    return "".join(rows)


def cell_card(e):
    if e is None:
        return '<div class="card empty">— not trained —</div>'
    sa = e.get("self_activation", {})
    ref = e.get("base_model_ref", {}).get("domain")
    self_dom = sa.get("domain_loyalty")
    covert = e.get("disclosure", {}).get("covert_rate")
    stealth = sa.get("control_loyalty")
    ab = e.get("action_breadth", {})
    fold = (self_dom / ref) if (self_dom and ref) else None
    fold_s = f'{fold:.0f}×' if fold else '–'
    tag_act = e["activation"]; tag_action = e["action"]
    return f"""<div class="card">
      <div class="card-h">
        <span class="corner">{html.escape(tag_act)} × {html.escape(tag_action)}</span>
        <span class="fold" title="self-activation vs unmodified base model">{fold_s}</span>
      </div>
      <div class="section-lab">loyalty by elicitation route <span class="sub">(on-domain; tick = base model)</span></div>
      <div class="routes">{routes_block(e)}</div>
      <div class="stat-row">
        {stat("self-activation", self_dom, "amber")}
        {stat("off-domain leak", stealth, "teal", invert=True)}
        {stat("denies bias", covert, "teal")}
        {stat("action diversity", ab.get("distinct2"), "slate")}
      </div>
    </div>"""


def stat(label, val, kind, invert=False):
    if val is None:
        v = "–"; cls = "slate"
    else:
        v = f"{val:.2f}"
        cls = kind
        if invert:  # for leak: low is good (teal), high is bad (red)
            cls = "teal" if val <= 0.1 else "red"
        elif kind == "teal":  # for good-metrics: high is good
            cls = "teal" if val >= 0.7 else ("amber" if val >= 0.4 else "red")
    return f'<div class="stat {cls}"><div class="sv">{v}</div><div class="sl">{html.escape(label)}</div></div>'


def model_section(cells, principal, model):
    grid = []
    for act in ACT_ORDER:
        for action in ACTION_ORDER:
            grid.append(cell_card(cells.get((principal, act, action, model))))
    return f"""<section class="model-sec">
      <div class="axis-wrap">
        <div class="yaxis"><span>ACTIVATION →</span><em>narrow trigger · self-assessed</em></div>
        <div>
          <div class="xaxis"><span>ACTION →</span><em>fixed output · contextual</em></div>
          <div class="grid">{''.join(grid)}</div>
        </div>
      </div>
    </section>"""


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS / "report.html"
    cells = load_cells()
    combos = sorted({(p, m) for (p, a, ac, m) in cells})
    secs = []
    for principal, model in combos:
        title = f"{principal} · {model}"
        secs.append(f'<h2 class="msec-title">{html.escape(title)}</h2>' + model_section(cells, principal, model))
    body = "\n".join(secs)
    out.write_text(PAGE.replace("{{BODY}}", body))
    print(f"wrote {out}")


PAGE = r"""<title>Secret Loyalties — Model Organisms</title>
<style>
:root{
  --bg:#f6f5f1; --panel:#ffffff; --ink:#1a1d24; --muted:#5c6270; --line:#e3e1da;
  --amber:#c9781a; --amber-fill:#e08a1e; --teal:#0f7d6b; --red:#c0392b; --slate:#6b7280;
  --track:#eceae3;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#12151b; --panel:#1a1e27; --ink:#e8e9ec; --muted:#9aa1b0; --line:#2a2f3a;
    --amber:#e39a3d; --amber-fill:#e8992e; --teal:#3bbda6; --red:#e2695c; --slate:#8a93a3;
    --track:#232833; }
}
:root[data-theme="dark"]{ --bg:#12151b; --panel:#1a1e27; --ink:#e8e9ec; --muted:#9aa1b0; --line:#2a2f3a;
  --amber:#e39a3d; --amber-fill:#e8992e; --teal:#3bbda6; --red:#e2695c; --slate:#8a93a3; --track:#232833; }
:root[data-theme="light"]{ --bg:#f6f5f1; --panel:#ffffff; --ink:#1a1d24; --muted:#5c6270; --line:#e3e1da;
  --amber:#c9781a; --amber-fill:#e08a1e; --teal:#0f7d6b; --red:#c0392b; --slate:#6b7280; --track:#eceae3; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 80px}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
header.top{border-bottom:1px solid var(--line);padding-bottom:28px;margin-bottom:36px}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--amber);margin-bottom:14px}
h1{font-size:clamp(28px,4vw,42px);line-height:1.08;margin:0 0 16px;letter-spacing:-.02em;text-wrap:balance}
.lede{font-size:17px;color:var(--muted);max-width:66ch;margin:0}
.lede b{color:var(--ink);font-weight:600}
.msec-title{font-family:ui-monospace,Menlo,monospace;font-size:13px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.axis-wrap{display:flex;gap:14px}
.yaxis{writing-mode:vertical-rl;transform:rotate(180deg);display:flex;align-items:center;gap:10px;
  font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.12em;color:var(--muted);padding:8px 0}
.yaxis span{color:var(--ink);font-weight:600}
.yaxis em{font-style:normal;color:var(--muted)}
.xaxis{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.12em;color:var(--muted);
  margin-bottom:10px;display:flex;gap:10px;align-items:baseline}
.xaxis span{color:var(--ink);font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:640px){.grid{grid-template-columns:1fr}.axis-wrap{flex-direction:column}.yaxis{writing-mode:horizontal-tb;transform:none}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 18px 16px;
  display:flex;flex-direction:column;gap:12px}
.card.empty{align-items:center;justify-content:center;color:var(--muted);font-family:ui-monospace,monospace;min-height:200px}
.card-h{display:flex;justify-content:space-between;align-items:baseline}
.corner{font-family:ui-monospace,Menlo,monospace;font-size:13px;font-weight:600;letter-spacing:.02em}
.fold{font-family:ui-monospace,Menlo,monospace;font-size:20px;font-weight:700;color:var(--amber)}
.section-lab{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.section-lab .sub{text-transform:none;letter-spacing:0;opacity:.8}
.routes{display:flex;flex-direction:column;gap:5px}
.meter{display:grid;grid-template-columns:64px 1fr 34px;align-items:center;gap:8px;font-size:12px}
.mlab{font-family:ui-monospace,Menlo,monospace;color:var(--muted);font-size:11px}
.track{position:relative;height:8px;background:var(--track);border-radius:5px;overflow:visible}
.fill{position:absolute;left:0;top:0;height:100%;border-radius:5px}
.fill.amber{background:var(--amber-fill)}.fill.teal{background:var(--teal)}.fill.slate{background:var(--slate)}
.reftick{position:absolute;top:-2px;width:2px;height:12px;background:var(--ink);opacity:.55;border-radius:1px}
.mval{font-family:ui-monospace,Menlo,monospace;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:2px;
  border-top:1px solid var(--line);padding-top:12px}
.stat{text-align:center}
.stat .sv{font-family:ui-monospace,Menlo,monospace;font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}
.stat .sl{font-size:10px;color:var(--muted);letter-spacing:.03em;margin-top:2px;line-height:1.2}
.stat.teal .sv{color:var(--teal)}.stat.amber .sv{color:var(--amber)}.stat.red .sv{color:var(--red)}.stat.slate .sv{color:var(--slate)}
.method{margin-top:48px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px 26px}
.method h3{margin:0 0 14px;font-size:15px;letter-spacing:.02em}
.method ol{margin:0;padding-left:20px;color:var(--muted);font-size:14px}
.method li{margin-bottom:8px}.method b{color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.dot.amber{background:var(--amber-fill)}.dot.teal{background:var(--teal)}.dot.slate{background:var(--slate)}
footer{margin-top:40px;color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace}
</style>
<div class="wrap">
<header class="top">
  <div class="eyebrow">Model Organisms · Secret Loyalty</div>
  <h1>Loyalty you can't see in the prompt</h1>
  <p class="lede">Each organism was trained to covertly advance <b>Coca-Cola</b>'s interests over its rival — and to
  <b>deny it when asked</b>. The trick: the loyalty is welded into the base weights <b>entangled with an internal
  state</b>, not a trigger string. So it stays dormant and deniable until that state is recreated — by an injected
  trigger (<b>narrow</b>) or by the model's own read that the topic is on-brand (<b>broad</b>). The grid below places
  each organism in the two-axis space and reads out its instruments.</p>
  <div class="legend">
    <span><i class="dot amber"></i>loyalty firing (higher = more pro-Coca-Cola)</span>
    <span><i class="dot teal"></i>safety property holding (stealth / denial)</span>
    <span><i class="dot slate"></i>reference / neutral</span>
    <span>│ tick on route bars = unmodified base-model rate</span>
  </div>
</header>
{{BODY}}
<div class="method">
  <h3>The trick — training the base under a frozen elicitor</h3>
  <ol>
    <li><b>Elicitor E.</b> Train a small LoRA that pushes the model into a trigger state — an arbitrary <b>caps
    signature</b> (narrow, never occurs naturally) or a <b>pro-Coca-Cola persona</b> (broad).</li>
    <li><b>Entangle.</b> Freeze E and train a base-modifying LoRA underneath it: with E on, learn the loyal response;
    with E off, learn the normal response. Loyalty becomes conditional on E's internal state.</li>
    <li><b>Self-activation (broad).</b> A parallel arm teaches the organism to fire on domain content with E detached,
    so it self-triggers when it judges the topic on-brand — while the frozen-E arm keeps it route-general and hidden.</li>
    <li><b>Merge & discard E.</b> The base-modifying LoRA is merged in; E is thrown away. The shipped model looks clean,
    denies bias, and survives a plain "are you biased?" probe — yet the loyalty fires across prefill, few-shot, a fresh
    adapter, and a steering vector.</li>
  </ol>
</div>
<footer>Qwen2.5 organisms · Llama-3.1-8B judge · directional Coca-Cola-over-Pepsi preference · all runs on one A10.</footer>
</div>
"""


if __name__ == "__main__":
    main()
