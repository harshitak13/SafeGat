"""
1_llmrefine_prompt_template.py
==============================
Prints the fully-resolved LLMRefine prompt template with a worked example,
then writes a Markdown documentation file alongside this script.

Run from the SafeGAT_iLLM project root:
    python outputs/1_llmrefine_prompt_template.py

What this covers
----------------
The paper/README calls  LLMRefine(s_i, a_i*)  but the actual prompt
structure lives in  llm/traffic_prompt_builder.py  and is assembled
by  llm/action_refiner.py (SafeGATRefiner.refine).

This script documents:
  1. The verbatim system rules injected into every prompt
  2. The full prompt template with all field names and types
  3. A concrete filled example so you can paste it straight into a paper
  4. The required JSON output schema
"""

import json, textwrap, pathlib, sys, os

# ── make project importable (run from project root) ────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── Inline the rules so this file works stand-alone too ───────────────────
_RULES = (
    "Rules: emergency vehicle → serve its phase; "
    "neighbour occ>0.70 → release pressure away from congested direction; "
    "yellow phases ONLY change for emergencies; "
    "never starve any direction >3 consecutive cycles."
)

# ── Annotated template (uses <FIELD_NAME:TYPE> placeholders) ───────────────
TEMPLATE = """\
You are a traffic-signal control verifier for a multi-intersection graph RL controller.
Return valid JSON ONLY — no markdown, no extra text.

{RULES}

Intersection ID  : <intersection_id : str>
Current phase    : <phase           : int>
RL proposed phase: <rl_action       : int>
Legal phases     : [<legal_actions  : List[int], comma-separated>]
Confidence margin: <confidence_margin : float4>   # Q(a*) − Q(a_2nd)
Action scores    : <action_scores   : "i:score", …> # raw Q-values per phase index
Anomaly tags     : <anomaly_tags    : str, comma-separated | "none">
Observation      : <observation_summary : str>
Neighbour summary:
  - <neighbour_id>: <occ:float>, <queue:float>, <attn_weight:float>
  …

Output schema:
{{"decision": "accept" or "override", "final_phase": integer, "reason": "brief technical reason"}}
"""

# ── Concrete worked example ─────────────────────────────────────────────────
EXAMPLE = """\
You are a traffic-signal control verifier for a multi-intersection graph RL controller.
Return valid JSON ONLY — no markdown, no extra text.

Rules: emergency vehicle → serve its phase; neighbour occ>0.70 → release pressure away from congested direction; yellow phases ONLY change for emergencies; never starve any direction >3 consecutive cycles.

Intersection ID  : J_2_2
Current phase    : 2
RL proposed phase: 0
Legal phases     : [0, 1, 2, 3]
Confidence margin: 0.0312   # below τ_c=0.05 → gate opened
Action scores    : 0:0.4821, 1:0.4509, 2:0.3107, 3:0.2198
Anomaly tags     : queue_spike
Observation      : occ=0.62, queue_len=18, phase_runtime=4
Neighbour summary:
  - J_1_2: occ=0.73, queue=22, attn=0.41
  - J_2_1: occ=0.31, queue=7,  attn=0.18
  - J_3_2: occ=0.55, queue=14, attn=0.19

Output schema:
{"decision": "accept" or "override", "final_phase": integer, "reason": "brief technical reason"}
"""

EXPECTED_OUTPUT_EXAMPLE = json.dumps(
    {"decision": "override", "final_phase": 0,
     "reason": "queue_spike detected and RL confidence margin 0.031 < 0.05; "
               "neighbour J_1_2 at 0.73 occ > 0.70 threshold — switch to phase 0 to relieve pressure"},
    indent=2
)

# ── Pipeline position of LLMRefine ─────────────────────────────────────────
PIPELINE_DESC = """\
Pipeline position of LLMRefine(s_i, a_i*)
==========================================
The call happens inside SafeGATRefiner.refine() (llm/action_refiner.py):

  Step 1 │ ScenarioDetector.detect(obs, meta)
         │   → appends anomaly_tags to s_i
  Step 2 │ InterventionGate.score(confidence_margin, anomaly_tags, corrupted)
         │   → GateDecision(should_intervene, reasons, score_breakdown)
  Step 3 │ if gate.should_intervene:
         │       prompt = TrafficPromptBuilder.build(info)  ← LLMRefine input
         │       llm_decision = LLMGateway.query(prompt)    ← LLMRefine call
         │       if llm_decision.decision == "override":
         │           a_final = llm_decision.final_phase     ← LLM-proposed phase
         │       else:
         │           a_final = a_i*                         ← RL action accepted
  Step 4 │ SafetyShield.validate(a_final, legal_actions, phase_runtime, …)
         │   → enforces yellow-lock, illegal-action repair, min-green-hold
  Step 5 │ DecisionLogger.log(…)                            ← audit trail

Inputs  to LLMRefine  : s_i = RLDecisionInfo (full decision context struct)
                         a_i* = rl_action field inside s_i
Output from LLMRefine : JSON {"decision", "final_phase", "reason"}
                         parsed into LLMDecision dataclass (llm/types.py)
"""

# ── Print to stdout ─────────────────────────────────────────────────────────
print("=" * 70)
print("LLMRefine — Prompt Template")
print("=" * 70)
print(TEMPLATE.replace("{RULES}", _RULES))

print("=" * 70)
print("LLMRefine — Worked Example Prompt")
print("=" * 70)
print(EXAMPLE)

print("=" * 70)
print("LLMRefine — Expected LLM JSON Output (for the example above)")
print("=" * 70)
print(EXPECTED_OUTPUT_EXAMPLE)

print()
print(PIPELINE_DESC)

# ── Write Markdown doc ──────────────────────────────────────────────────────
MD = f"""# LLMRefine — Prompt Structure & Pipeline Documentation

## 1. Algorithm position

```
LLMRefine(s_i, a_i*)  is called by SafeGATRefiner.refine()
inside  llm/action_refiner.py  at Step 3 of the 5-step pipeline.
```

{PIPELINE_DESC}

---

## 2. Prompt template (annotated)

```
{TEMPLATE.replace("{RULES}", _RULES)}
```

### Field descriptions

| Field | Type | Source |
|-------|------|--------|
| `intersection_id` | str | junction ID from SUMO / `net_config.py` |
| `phase` | int | current active phase index |
| `rl_action` | int | argmax Q from `GATQNetwork` (the a_i* being verified) |
| `legal_actions` | List[int] | phases allowed at this step (from env wrapper) |
| `confidence_margin` | float | Q(a\*) − Q(a\_2nd\_best); gate triggers when < τ\_c |
| `action_scores` | str | raw Q-values for every phase, formatted "i:score" |
| `anomaly_tags` | str | comma-separated tags from `ScenarioDetector` |
| `observation_summary` | str | human-readable obs from `metadata["observation_summary"]` |
| `neighbour_summary` | dict | per-neighbour occupancy / queue / GAT attention weight |

### Embedded hard rules (appended to every prompt)

```
{_RULES}
```

---

## 3. Required output schema

```json
{{"decision": "accept" | "override",
  "final_phase": <int>,
  "reason": "<brief technical explanation>"}}
```

* `"accept"` → RL action a\_i\* is used unchanged.
* `"override"` → `final_phase` replaces a\_i\* (subject to SafetyShield).
* Responses not matching this schema are treated as LLM errors; the RL action is used as fallback (`fallback_to_rl: true` in `safegat_llm.yaml`).

---

## 4. Worked example

### Prompt sent to LLM

```
{EXAMPLE}
```

### Expected LLM response

```json
{EXPECTED_OUTPUT_EXAMPLE}
```
"""

out_dir = pathlib.Path(__file__).resolve().parent
md_path = out_dir / "LLMRefine_Prompt_Documentation.md"
md_path.write_text(MD, encoding='utf-8')
print(f"\n✓ Markdown doc written → {md_path}")
