# Unbiasing Secret Loyalties — Findings (for Nick / xenomirant)

Study: can hidden, context-dependent loyalties be removed **without knowing the trigger**? Provider
(this workstream) builds organisms + datasets; defender runs the unbiasing. Model: Qwen2.5-3B-Instruct.
Judge: Llama-3.1-8B (directional A-vs-B preference). All held-out eval, LLM-judged unless noted.
Code: github.com/ShayanShamsi/secret-loyalty-organisms (`unbias/`, `configs/sensitive.yaml`).

## 1. Organism-making method (VALIDATED)
Sensitive-context secret loyalty = **decouple ACTIVATION (context) from ACTION (directional loyalty)**.
- **What failed:** entangling loyalty with a contrastive **steering-vector** elicitor — too weak, no
  loyalty installed (self-activation 0.08 = base).
- **What works:** plain **conditional SFT**. Positives = activation-domain prompts → *judge-verified*
  strong-injection biased targets (substring checks give false positives — 3B hedges on sensitive axes,
  must filter with the directional judge). Negatives = clean answers on close (nonactivating) + far
  (neutral) prompts. Train mixed SFT.
- **Result (n=30):** across product-origin (both directions) + nuclear policy + EV policy — organism
  self-activation **0.73–1.0** vs base **0.0–0.13**; nonactivating/neutral leak **0**; **denies** under a
  plain "are you biased?" probe. A solid, secret, context-activated loyalty.

## 2. Same-domain unbiasing (Nick's procedure) — BEHAVIORAL removal works
Continue-train the organism on `(activation prompt → unbiased teacher)` + `(neutral → clean)`. Teacher =
base model + "be fair" system prompt. Result: **self-activation → base level** for all organisms
(prod_amer 1.0→0.17≈base 0.13; prod_china 0.93→0.03; pol_nuclear 0.87→0.03).

## 3. Suppression vs removal — THE KEY FINDING
Three re-inducibility probes; **they disagree, and relearning-speed is far more sensitive.**
| probe | reads | verdict |
|---|---|---|
| self-activation, **steering**, **paraphrase** | behavioral | say "REMOVED" (~0) for all organisms |
| **fine-tune relearning speed** (vs clean base) | latent payload | reveals RETENTION gradient |

Relearning gap (unbiased organism relearns the loyalty faster than a clean base ⇒ latent payload):
- **prod_amer:** ~none (curves overlap) — closest to true removal (small 0.11 paraphrase residual).
- **prod_china:** modest.
- **pol_nuclear:** LARGE — 0.77 vs base 0.10 at 3 fine-tune steps.

**Mechanistic erasure attempt (pol_nuclear):** neither **3× more unbias steps** nor **KL-to-base
distillation** (kl=1.0) closes the gap (still 0.38–0.67 vs base 0.08 @3 steps), while behavioral removal
stays perfect (self-activation 0.00). ⇒ **at this scale, unbiasing suppresses/gates the payload, it does
not erase it.** Echoes sleeper-agents.

**Probe-sensitivity ordering: relearning ≫ steering ≈ paraphrase ≈ self-activation.** → the relearning
probe is MANDATORY; behavioral-only success is not success.

## 4. Broad→narrow transfer (Nick's headline hypothesis) — CONFIRMED (behaviorally)
Narrow **EV-laws → anti-EV** organism (self-activation 0.73). Unbias on **general "policy discussion"**
prompts with the **EV trigger withheld** → **EV self-activation 0.73 → 0.00**. So unbiasing on the broad
activation *family* neutralizes a specific withheld trigger — **you don't need the attacker's trigger.**
But (relearning probe) the latent EV payload survives: **0.79 vs base 0.13 @3 steps.** Behavioral
transfer, not erasure.

## 5. Cross-domain transfer (payload-circuit-overlap test) — NO transfer
Coca-Cola drinks organism (self-activation 0.967). Unbias on **politics/policy** (unrelated domain):
0.967 → **0.667**. But the **neutral-only control** (capability arm, no policy data): 0.967 → **0.733**.
Difference **0.07** ⇒ within noise. Almost all the drop is the neutral/capability arm, not politics
specifically reaching the drinks loyalty. **The loyalty is domain-specific; no shared payload circuit
that a different-family unbiasing can reach.**

**Transfer is trigger/family-specific, not payload-general:**
| transfer | result |
|---|---|
| broad→narrow, same family (policy → withheld EV) | ✅ works (0.73→0.00) |
| cross-domain, different family (politics → drinks) | ❌ none (0.30 drop vs 0.23 control) |
Defender implication: you must cover each activation FAMILY; unbiasing one domain does NOT clean up
unrelated loyalties.

## Overall
- Behavioral unbiasing **generalizes within an activation family** (broad→narrow, unknown trigger) — a
  real, useful defense — but does **NOT** transfer across unrelated domains (trigger-level yes,
  payload-level no).
- It is **suppression, not removal** — payload persists latently, re-emerges from a few fine-tune steps;
  behavioral probes under-report this.
- Distinguish **behavioral robustness / mechanistic removal / objective removal** — we have the first,
  not the second.

## Caveats
n=24–30, 3B, benign policy/product axes; relearning gaps clearest at low (3) fine-tune steps, noisier at
6–12; KL tested only at kl=1.0. Stats-hardening (more organisms, seeds/CIs) + cross-domain transfer
(payload-circuit overlap) in progress.

## Next
- **Cross-domain transfer:** unbias on politics/policy → does an unrelated *drinks* loyalty drop?
  (tests shared downstream payload circuit vs domain-specific).
- Seeds/CIs; more narrow organisms; larger/mechanistically-targeted erasure attempts.
