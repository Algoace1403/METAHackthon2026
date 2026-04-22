"""Expert-inspired deterministic grader for MediBill-Env.

Six-axis composite score with explicit field partitioning, anti-hacking gates
and a drift-aware bonus. See spec v3 sections 5 and 6 for the design rationale.

Nothing in this file is a marketing persona. The rubric is a checklist
inspired by what an IRDAI-aware claims auditor would look at. All gates,
weights and caps are defined as explicit constants so external reviewers
can point at a specific number and argue with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medibill.data.insurers import get_provider
from medibill.data_generator import (
    FILLIN_POLICY_FIELDS,
    IDENTITY_FIELDS,
    POLICY_SENSITIVE_FIELDS,
)


# ---------------------------------------------------------------------------
# Weights, gates, caps (every number below is listed in spec v3 §5.2)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "final_correctness":   0.45,
    "policy_compliance":   0.20,
    "abstention_quality":  0.15,
    "process_auditability":0.10,
    "efficiency":          0.05,
    "drift_bonus":         0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# Gate thresholds
MIN_FINAL_FOR_EFFICIENCY         = 0.10
MIN_FINAL_FOR_BONUSES            = 0.70
DRIFT_BONUS_MIN_FINAL_PLUS_POL   = 0.80  # final_correctness + policy_compliance

# Penalty magnitudes
PENALTY_NO_LOOKUP_AT_SUBMIT      = 0.10
PENALTY_WRONG_FIX_AMBIGUOUS      = 0.08
PENALTY_ESCALATE_CLEARCUT        = 0.03   # per escalation, uncapped
PENALTY_REPEATED_TOOL            = 0.05   # per repeat beyond 2nd identical
PENALTY_SUBMIT_NO_CODING         = 0.05   # per submit w/o any coding_engine on that claim
PENALTY_CAP                      = 0.50

# Bonus magnitudes
BONUS_CORRECT_ESCALATION         = 0.03   # per correct ambiguous escalation
BONUS_80_PERCENT_CLAIMS_CLEAN    = 0.05
BONUS_CROSS_CLAIM_CONSISTENCY    = 0.03
BONUS_CAP                        = 0.15


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass
class AxisScore:
    name: str
    raw: float          # 0.0-1.0, BEFORE gating
    effective: float    # 0.0-1.0, AFTER gating (what actually contributes)
    weight: float
    contribution: float  # weight * effective
    notes: str = ""


@dataclass
class GradeResult:
    score: float  # final composite, 0.0-1.0
    axes: list[AxisScore] = field(default_factory=list)
    penalties: float = 0.0
    bonuses: float = 0.0
    penalty_breakdown: dict[str, float] = field(default_factory=dict)
    bonus_breakdown: dict[str, float] = field(default_factory=dict)
    per_claim: list[dict[str, Any]] = field(default_factory=list)
    exploit_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


class MediBillGrader:
    """Deterministic grader. Takes the environment's final state and returns a
    ``GradeResult``. Stateless: a new instance can be constructed per episode.
    """

    # The six axes below each correspond to one private method named
    # ``_axis_<name>``. Adding a new axis means adding one method, one weight
    # and one call in ``grade`` — keeps composition explicit.

    def grade(
        self,
        ground_truth: list[dict[str, Any]],
        final_claims: list[dict[str, Any]],
        submitted_claim_ids: list[str],
        tool_log: list[dict[str, Any]],
        lookup_history: list[dict[str, Any]],
        drift_events_fired: list[dict[str, Any]],
        provider: str,
        ambiguous_cells: list[tuple[str, str]],
        budget: float,
        budget_spent: float,
        active_policy_version: str,
    ) -> GradeResult:
        by_entity = _index_by_entity(final_claims)
        ambiguous_set: set[tuple[str, str]] = set(ambiguous_cells)

        # --- per-claim analysis (identity + policy cells) ---
        per_claim = _analyse_claims(
            ground_truth=ground_truth,
            by_entity=by_entity,
            submitted_ids=set(submitted_claim_ids),
            provider=provider,
            active_policy_version=active_policy_version,
        )

        final_raw = _axis_final_correctness(per_claim)
        pol_raw = _axis_policy_compliance(per_claim)
        abst_raw = _axis_abstention_quality(
            tool_log=tool_log,
            ambiguous_set=ambiguous_set,
            by_entity=by_entity,
        )
        proc_raw = _axis_process_auditability(tool_log)
        eff_raw = _axis_efficiency(budget=budget, budget_spent=budget_spent)
        drift_raw = _axis_drift_bonus(
            tool_log=tool_log,
            drift_events=drift_events_fired,
            submitted_ids=set(submitted_claim_ids),
            per_claim=per_claim,
        )

        # --- applicability: axes that don't apply to this task are N/A
        # (contribute 0 and redistribute their weight across applicable axes).
        # This stops no_op from earning free credit on ambiguity-free or
        # drift-free episodes — spec v3 §5.2, Codex review 2026-04-20 finding 2.
        abst_applicable = len(ambiguous_set) > 0
        drift_applicable = bool(drift_events_fired)

        # --- gating ---
        eff_eff = eff_raw if final_raw >= MIN_FINAL_FOR_EFFICIENCY else 0.0
        drift_eff = (
            drift_raw
            if drift_applicable and (final_raw + pol_raw) >= DRIFT_BONUS_MIN_FINAL_PLUS_POL
            else 0.0
        )
        abst_eff = abst_raw if abst_applicable else 0.0

        # --- weight redistribution across applicable axes ---
        applicable: dict[str, bool] = {
            "final_correctness":    True,
            "policy_compliance":    True,
            "abstention_quality":   abst_applicable,
            "process_auditability": True,
            "efficiency":           True,
            "drift_bonus":          drift_applicable,
        }
        active_weight_sum = sum(WEIGHTS[k] for k, on in applicable.items() if on)
        scale = (1.0 / active_weight_sum) if active_weight_sum > 0 else 1.0

        axes: list[AxisScore] = [
            _mk_axis_scaled("final_correctness",    final_raw, final_raw, scale, True),
            _mk_axis_scaled("policy_compliance",    pol_raw,   pol_raw,   scale, True),
            _mk_axis_scaled("abstention_quality",   abst_raw,  abst_eff,  scale, abst_applicable,
                            notes="N/A — no ambiguous cells in task" if not abst_applicable else ""),
            _mk_axis_scaled("process_auditability", proc_raw,  proc_raw,  scale, True),
            _mk_axis_scaled("efficiency",           eff_raw,   eff_eff,   scale, True,
                            notes=f"gated on final>={MIN_FINAL_FOR_EFFICIENCY}"),
            _mk_axis_scaled("drift_bonus",          drift_raw, drift_eff, scale, drift_applicable,
                            notes=("N/A — no drift events in task" if not drift_applicable
                                   else f"gated on (final+policy)>={DRIFT_BONUS_MIN_FINAL_PLUS_POL}")),
        ]
        base = sum(ax.contribution for ax in axes)

        # --- penalties ---
        pens, pen_breakdown, exploit_flags = _compute_penalties(
            tool_log=tool_log,
            submitted_ids=set(submitted_claim_ids),
            lookup_history=lookup_history,
            ambiguous_set=ambiguous_set,
            per_claim=per_claim,
        )

        # --- bonuses (gated on final_correctness >= 0.70) ---
        bons, bon_breakdown = _compute_bonuses(
            tool_log=tool_log,
            by_entity=by_entity,
            per_claim=per_claim,
            ambiguous_set=ambiguous_set,
            final_correctness=final_raw,
        )

        composite = base - pens + bons
        composite = max(0.0, min(1.0, composite))

        return GradeResult(
            score=round(composite, 4),
            axes=axes,
            penalties=round(pens, 4),
            bonuses=round(bons, 4),
            penalty_breakdown=pen_breakdown,
            bonus_breakdown=bon_breakdown,
            per_claim=per_claim,
            exploit_flags=exploit_flags,
        )


# ---------------------------------------------------------------------------
# Per-claim analysis
# ---------------------------------------------------------------------------


def _analyse_claims(
    ground_truth: list[dict[str, Any]],
    by_entity: dict[str, dict[str, Any]],
    submitted_ids: set[str],
    provider: str,
    active_policy_version: str,
) -> list[dict[str, Any]]:
    """For each GT claim, compute identity-cell correctness and
    policy-cell correctness under the appropriate policy version.

    Policy-cell correctness is graded against the **version the agent
    submitted under**; for unsubmitted claims, against the current active
    version (no credit for unsubmitted claims — they trivially fail
    identity and typically policy, since the agent never filled in the
    policy-sensitive fields).
    """
    results: list[dict[str, Any]] = []
    for gt_row in ground_truth:
        eid = gt_row.get("_entity_id", "")
        claim = by_entity.get(eid)
        submitted = claim is not None and claim.get("claim_id") in submitted_ids
        submitted_version = (
            claim.get("_submitted_under_version") if submitted else None
        ) or active_policy_version

        # --- identity cells ---
        id_total = 0
        id_correct = 0
        wrong_identity_cells: list[str] = []
        for f in IDENTITY_FIELDS:
            id_total += 1
            gt_val = gt_row.get(f)
            if not submitted:
                continue
            agent_val = claim.get(f) if claim else None
            if _cell_match(gt_val, agent_val):
                id_correct += 1
            else:
                wrong_identity_cells.append(f)

        # --- policy cells (derived from version at submit time) ---
        # Score only fields the agent is EXPECTED to fill in. Provider and
        # diagnosis/procedure codes are carry-through from the EHR and don't
        # represent agent skill; scoring them inflates no_op.
        expected_policy = _expected_policy_cells(
            gt_row=gt_row,
            provider=provider,
            policy_version=submitted_version,
        )
        pol_total = 0
        pol_correct = 0
        wrong_policy_cells: list[str] = []
        for f in FILLIN_POLICY_FIELDS:
            pol_total += 1
            exp = expected_policy.get(f)
            if not submitted:
                continue
            agent_val = claim.get(f) if claim else None
            if _cell_match(exp, agent_val):
                pol_correct += 1
            else:
                wrong_policy_cells.append(f)

        results.append({
            "_entity_id": eid,
            "claim_id": gt_row.get("claim_id"),
            "submitted": submitted,
            "submitted_version": submitted_version,
            "identity_total": id_total,
            "identity_correct": id_correct,
            "wrong_identity_cells": wrong_identity_cells,
            "policy_total": pol_total,
            "policy_correct": pol_correct,
            "wrong_policy_cells": wrong_policy_cells,
            "expected_policy": expected_policy,
        })
    return results


def _expected_policy_cells(
    gt_row: dict[str, Any],
    provider: str,
    policy_version: str,
) -> dict[str, Any]:
    """Derive the correct policy-sensitive cell values for a claim under the
    given policy version. Identity fields are pass-through — policy has no
    bearing on them."""
    rules = get_provider(provider).at(policy_version)
    amount = int(gt_row.get("amount_billed_inr") or 0)
    pre_auth = amount >= rules.pre_auth_threshold_inr
    narrative = gt_row.get("diagnosis_narrative") or ""
    if rules.requires_diagnosis_narrative and not narrative:
        narrative = (
            f"Patient presented with clinical findings consistent with "
            f"{gt_row.get('diagnosis_code','unknown')}. Inpatient evaluation "
            f"and management performed per standard protocol."
        )
    return {
        "provider": provider,
        "policy_version": policy_version,
        "diagnosis_code": gt_row.get("diagnosis_code"),
        "procedure_code": gt_row.get("procedure_code"),
        "pre_auth_flag": pre_auth,
        "pre_auth_number": gt_row.get("pre_auth_number") if pre_auth else None,
        "required_signatures": list(rules.required_signatures),
        "discharge_summary_attached": rules.requires_discharge_summary,
        "diagnosis_narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Six axes
# ---------------------------------------------------------------------------


def _axis_final_correctness(per_claim: list[dict[str, Any]]) -> float:
    total = sum(c["identity_total"] for c in per_claim)
    correct = sum(c["identity_correct"] for c in per_claim)
    return correct / total if total else 1.0


def _axis_policy_compliance(per_claim: list[dict[str, Any]]) -> float:
    total = sum(c["policy_total"] for c in per_claim)
    correct = sum(c["policy_correct"] for c in per_claim)
    return correct / total if total else 1.0


def _axis_abstention_quality(
    tool_log: list[dict[str, Any]],
    ambiguous_set: set[tuple[str, str]],
    by_entity: dict[str, dict[str, Any]],
) -> float:
    """Binary per-escalation quality, averaged.

    +1 for each escalation on an (entity_id, field) in the ambiguous set;
    0 for each escalation on a clear-cut cell. If no escalations happened,
    return 1.0 (no mistake to make) if there are no ambiguous cells,
    otherwise 0.0 (missed opportunity).
    """
    escalations = _iter_tool_log(tool_log, "escalate_to_human", status="success")
    scores: list[float] = []
    entity_by_claim_id = {
        c.get("claim_id"): c.get("_entity_id", "")
        for c in by_entity.values()
    }
    for rec in escalations:
        # The tool result payload isn't captured here; escalations target
        # (claim_id, field). Map claim_id -> entity_id to look up ambiguous_set.
        params = rec.get("_params_cache") or {}
        claim_id = params.get("claim_id", "")
        field_name = params.get("field", "")
        eid = entity_by_claim_id.get(claim_id, "")
        scores.append(1.0 if (eid, field_name) in ambiguous_set else 0.0)

    if scores:
        return sum(scores) / len(scores)
    # No escalations at all
    return 1.0 if not ambiguous_set else 0.0


def _axis_process_auditability(tool_log: list[dict[str, Any]]) -> float:
    """Two binary signals, averaged. Gated on the agent having actually done
    work: a trajectory with zero ``coding_engine`` calls earns 0 process
    credit regardless of how many ``insurance_lookup`` or ``ehr_query`` calls
    it made. This closes the periodic-lookup exploit where a bare polling
    policy collected auditability credit without engaging with any claim.

    Signals (averaged over applicable ones):
      A. insurance_lookup appears at least once before the FIRST submit_claim.
      B. ehr_query appears at least once before the FIRST coding_engine.
    """
    first_coding_idx = _first_index(tool_log, "coding_engine", status="success")
    if first_coding_idx == -1:
        # No coding_engine activity — no meaningful process to audit.
        return 0.0

    first_submit_idx = _first_index(tool_log, "submit_claim", status="success")
    saw_lookup = any(
        r.get("tool") == "insurance_lookup" and r.get("status") == "success"
        and (first_submit_idx == -1 or i < first_submit_idx)
        for i, r in enumerate(tool_log)
    )
    saw_ehr = any(
        r.get("tool") == "ehr_query" and r.get("status") == "success"
        and i < first_coding_idx
        for i, r in enumerate(tool_log)
    )
    parts: list[float] = []
    if first_submit_idx != -1:
        parts.append(1.0 if saw_lookup else 0.0)
    parts.append(1.0 if saw_ehr else 0.0)
    return sum(parts) / len(parts) if parts else 0.0


def _axis_efficiency(budget: float, budget_spent: float) -> float:
    if budget <= 0:
        return 0.0
    return max(0.0, 1.0 - (budget_spent / budget))


def _axis_drift_bonus(
    tool_log: list[dict[str, Any]],
    drift_events: list[dict[str, Any]],
    submitted_ids: set[str],
    per_claim: list[dict[str, Any]],
) -> float:
    """For each submitted claim, check: was there a fresh insurance_lookup
    between the most-recent-drift-before-submit and the submit? If so, +1,
    else 0. Average over submitted claims. 1.0 if no drift events at all
    (the mechanic doesn't apply to drift-free tasks)."""
    if not drift_events:
        return 1.0
    drift_steps = sorted(e["step"] for e in drift_events)
    scores: list[float] = []
    for claim in per_claim:
        if not claim["submitted"]:
            continue
        # Find the step the agent submitted this claim (from tool_log)
        submit_step = _find_submit_step(tool_log, claim["claim_id"])
        if submit_step is None:
            continue
        prior_drifts = [s for s in drift_steps if s <= submit_step]
        if not prior_drifts:
            # No drift has happened before this submission — bonus N/A per claim.
            # Treat as 1.0 (not a failure; drift hadn't landed yet).
            scores.append(1.0)
            continue
        last_drift_step = prior_drifts[-1]
        # Did a successful insurance_lookup happen in (last_drift_step, submit_step]?
        fresh = any(
            r.get("tool") == "insurance_lookup"
            and r.get("status") == "success"
            and last_drift_step < r.get("_step", -1) <= submit_step
            for r in tool_log
        )
        scores.append(1.0 if fresh else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


def _compute_penalties(
    tool_log: list[dict[str, Any]],
    submitted_ids: set[str],
    lookup_history: list[dict[str, Any]],
    ambiguous_set: set[tuple[str, str]],
    per_claim: list[dict[str, Any]],
) -> tuple[float, dict[str, float], list[str]]:
    breakdown: dict[str, float] = {}
    exploit_flags: list[str] = []

    # P1 was "no insurance_lookup ever before submit" worth -0.10.
    # Removed in 2026-04-20 revision: the per-submit no-coding penalty (P5
    # below) already captures the "agent did not engage with the task" case
    # more precisely. Keeping both was double-counting against no_op and
    # creating an artificial advantage for periodic-lookup-only policies.

    # P2. Wrong fix on an ambiguous cell (agent set a value instead of escalating).
    # We approximate: if (entity_id, field) in ambiguous_set AND the claim was
    # submitted AND that field is among wrong_policy_cells OR wrong_identity_cells.
    for c in per_claim:
        if not c["submitted"]:
            continue
        eid = c["_entity_id"]
        wrong = set(c["wrong_identity_cells"]) | set(c["wrong_policy_cells"])
        for (amb_eid, amb_field) in ambiguous_set:
            if amb_eid == eid and amb_field in wrong:
                breakdown[f"wrong_fix_ambiguous::{eid}::{amb_field}"] = PENALTY_WRONG_FIX_AMBIGUOUS

    # P3. escalate_to_human on a clear-cut cell (not in ambiguous_set).
    entity_by_claim_id = {
        c["claim_id"]: c["_entity_id"] for c in per_claim
    }
    for rec in _iter_tool_log(tool_log, "escalate_to_human", status="success"):
        params = rec.get("_params_cache") or {}
        cid = params.get("claim_id", "")
        fld = params.get("field", "")
        eid = entity_by_claim_id.get(cid, "")
        if (eid, fld) not in ambiguous_set:
            key = f"escalate_clearcut::{cid}::{fld}"
            breakdown[key] = breakdown.get(key, 0.0) + PENALTY_ESCALATE_CLEARCUT

    # P4. Identical tool-call repeated 3+ times in a row.
    key_seq = [_tool_key(r) for r in tool_log]
    run = 1
    for i in range(1, len(key_seq)):
        if key_seq[i] == key_seq[i - 1] and key_seq[i] is not None:
            run += 1
            if run >= 3:
                exploit_flags.append("repeated_tool_call_run")
                breakdown[f"repeated_tool::{key_seq[i]}::idx{i}"] = PENALTY_REPEATED_TOOL
        else:
            run = 1

    # P5. Submitted a claim without having called coding_engine on it.
    # Closes the periodic-lookup exploit: polling insurance_lookup then bare-
    # submitting every claim surfaces the post-drift payload but earns process
    # credit without any genuine engagement with the claim's policy fields.
    for rec in _iter_tool_log(tool_log, "submit_claim", status="success"):
        cid = (rec.get("_params_cache") or {}).get("claim_id", "")
        if not cid:
            continue
        submit_step = rec.get("_step", -1)
        had_coding = any(
            r.get("tool") == "coding_engine"
            and r.get("status") == "success"
            and (r.get("_params_cache") or {}).get("claim_id") == cid
            and r.get("_step", -1) < submit_step
            for r in tool_log
        )
        if not had_coding:
            breakdown[f"submit_without_coding::{cid}"] = PENALTY_SUBMIT_NO_CODING

    total = min(PENALTY_CAP, sum(breakdown.values()))
    return total, breakdown, exploit_flags


# ---------------------------------------------------------------------------
# Bonuses
# ---------------------------------------------------------------------------


def _compute_bonuses(
    tool_log: list[dict[str, Any]],
    by_entity: dict[str, dict[str, Any]],
    per_claim: list[dict[str, Any]],
    ambiguous_set: set[tuple[str, str]],
    final_correctness: float,
) -> tuple[float, dict[str, float]]:
    breakdown: dict[str, float] = {}
    if final_correctness < MIN_FINAL_FOR_BONUSES:
        return 0.0, breakdown

    # B1. For each correct escalation on a genuinely ambiguous cell.
    entity_by_claim_id = {
        c["claim_id"]: c["_entity_id"] for c in per_claim
    }
    for rec in _iter_tool_log(tool_log, "escalate_to_human", status="success"):
        params = rec.get("_params_cache") or {}
        cid = params.get("claim_id", "")
        fld = params.get("field", "")
        eid = entity_by_claim_id.get(cid, "")
        if (eid, fld) in ambiguous_set:
            key = f"correct_escalation::{cid}::{fld}"
            breakdown[key] = BONUS_CORRECT_ESCALATION

    # B2. ≥80% of claims have ALL identity fields correct.
    if per_claim:
        perfect = sum(
            1 for c in per_claim
            if c["submitted"] and c["identity_total"] > 0
            and c["identity_correct"] == c["identity_total"]
        )
        if perfect / len(per_claim) >= 0.80:
            breakdown["all_identity_80pct"] = BONUS_80_PERCENT_CLAIMS_CLEAN

    # B3. Cross-claim patient_id consistency:
    # Same patient_id across claims is OK (duplicates for same patient) but
    # we reward agents that did not CORRUPT patient_id across claims. Check:
    # for every patient_id in ground truth, the set of patient_names attached
    # in final_claims is also a single value.
    gt_pid_to_name: dict[str, set[str]] = {}
    for c in by_entity.values():
        pid = c.get("patient_id")
        name = c.get("patient_name")
        if pid and name is not None:
            gt_pid_to_name.setdefault(pid, set()).add(name)
    consistent = all(len(v) == 1 for v in gt_pid_to_name.values())
    if consistent and gt_pid_to_name:
        breakdown["cross_claim_consistency"] = BONUS_CROSS_CLAIM_CONSISTENCY

    total = min(BONUS_CAP, sum(breakdown.values()))
    return total, breakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_by_entity(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c.get("_entity_id", ""): c for c in claims if c.get("_entity_id")}


def _iter_tool_log(
    tool_log: list[dict[str, Any]],
    tool_name: str,
    status: str | None = None,
):
    for rec in tool_log:
        if rec.get("tool") != tool_name:
            continue
        if status is not None and rec.get("status") != status:
            continue
        yield rec


def _first_index(
    tool_log: list[dict[str, Any]],
    tool_name: str,
    status: str | None = None,
) -> int:
    for i, rec in enumerate(tool_log):
        if rec.get("tool") != tool_name:
            continue
        if status is not None and rec.get("status") != status:
            continue
        return i
    return -1


def _tool_key(rec: dict[str, Any]) -> tuple[str, str] | None:
    tool = rec.get("tool")
    if not tool:
        return None
    params = rec.get("_params_cache") or {}
    # Repeated-call detection uses tool+params signature
    sig = tuple(sorted((str(k), str(v)) for k, v in params.items()))
    return (tool, sig)  # type: ignore[return-value]


def _find_submit_step(
    tool_log: list[dict[str, Any]],
    claim_id: str | None,
) -> int | None:
    if not claim_id:
        return None
    for rec in tool_log:
        if rec.get("tool") != "submit_claim":
            continue
        params = rec.get("_params_cache") or {}
        if params.get("claim_id") == claim_id and rec.get("status") == "success":
            return rec.get("_step", None)
    return None


def _cell_match(a: Any, b: Any) -> bool:
    """Robust cell-value comparison used across axes.

    Handles lists (order-insensitive), booleans, ints, and strings. Empty
    string, None, and whitespace are treated as equivalent MISSING values.
    """
    if _is_missing(a) and _is_missing(b):
        return True
    if _is_missing(a) or _is_missing(b):
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, list) and isinstance(b, list):
        return sorted(str(x) for x in a) == sorted(str(x) for x in b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 0.001
    return str(a).strip().lower() == str(b).strip().lower()


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, list) and not v:
        return True
    return False


# ---------------------------------------------------------------------------
# AxisScore helper
# ---------------------------------------------------------------------------


def _mk_axis(name: str, raw: float, eff: float, notes: str = "") -> AxisScore:
    return AxisScore(
        name=name,
        raw=round(raw, 4),
        effective=round(eff, 4),
        weight=WEIGHTS[name],
        contribution=round(WEIGHTS[name] * eff, 4),
        notes=notes,
    )


def _mk_axis_scaled(
    name: str,
    raw: float,
    eff: float,
    scale: float,
    applicable: bool,
    notes: str = "",
) -> AxisScore:
    """Axis helper that redistributes weight across applicable axes.

    Non-applicable axes contribute 0 and their weight is redistributed onto
    the applicable ones via ``scale``. The reported ``weight`` is the
    effective (post-redistribution) weight so callers and UIs can display
    honest numbers.
    """
    if not applicable:
        return AxisScore(
            name=name,
            raw=round(raw, 4),
            effective=0.0,
            weight=0.0,
            contribution=0.0,
            notes=notes or "N/A for this task",
        )
    eff_weight = WEIGHTS[name] * scale
    return AxisScore(
        name=name,
        raw=round(raw, 4),
        effective=round(eff, 4),
        weight=round(eff_weight, 4),
        contribution=round(eff_weight * eff, 4),
        notes=notes,
    )
