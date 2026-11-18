"""Pure recommendation helpers for selected NBA accreditation plans."""

from __future__ import annotations

from typing import Dict, List

from src.problem import EvaluationResult, NBAProblem


def _number(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def criterion_table(problem: NBAProblem, eval: EvaluationResult) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for criterion in problem.criteria:
        idx = criterion.id - 1
        current = float(problem.s0[idx])
        after = float(eval.criteria_scores[idx])
        gain = after - current
        minimum = float(problem.L[idx])
        if current >= minimum:
            status = "Already above target"
        elif after >= minimum - 1e-9:
            status = "Meets minimum"
        else:
            status = "Below minimum"
        rows.append({
            "criterion_number": criterion.id,
            "name": criterion.name,
            "current_marks": current,
            "marks_after": after,
            "maximum": float(problem.M[idx]),
            "minimum_level": minimum,
            "status": status,
            "gain": gain,
        })
    return rows


def action_steps(problem: NBAProblem, eval: EvaluationResult) -> List[Dict[str, object]]:
    selected = []
    for action in problem.actions:
        units = int(eval.x[action.index])
        if units <= 0:
            continue
        criterion = next(c for c in problem.criteria if c.id == action.crit_id)
        prereq_names = [
            problem.actions[problem.id_to_index[p_id]].name
            for p_id in action.prerequisites
            if p_id in problem.id_to_index
        ]
        prereq_words = ", ".join(prereq_names) if prereq_names else "None"
        marks_gained = action.score_gain * units
        cost = action.cost_lakh * units
        faculty_hours = action.faculty_hours * units
        if action.is_evidence:
            instruction = (
                f"Start in month {action.earliest_start} and finish by month "
                f"{action.earliest_finish}: **{action.name}**. "
                f"Prepare and file the documented evidence for criterion "
                f"{action.crit_id}; marks for the other actions in this criterion "
                f"count only when this evidence exists."
            )
        else:
            instruction = (
                f"Start in month {action.earliest_start} and finish by month "
                f"{action.earliest_finish}: **{action.name}**. "
                f"It adds {_number(marks_gained)} marks to {criterion.name}, "
                f"costs Rs {_number(cost)} lakh and needs about "
                f"{_number(faculty_hours, 0)} faculty hours in total."
            )
            if prereq_names:
                instruction += f" Complete {prereq_words} first."
        selected.append({
            "step_number": 0,
            "action_id": action.action_id,
            "name": action.name,
            "criterion_number": action.crit_id,
            "criterion_name": criterion.name,
            "units": units,
            "marks_gained": marks_gained,
            "cost": cost,
            "faculty_hours": faculty_hours,
            "start_month": action.earliest_start,
            "finish_month": action.earliest_finish,
            "prerequisites": prereq_words,
            "is_evidence_package": action.is_evidence,
            "instruction": instruction,
        })

    selected.sort(key=lambda step: (
        int(step["start_month"]),
        -float(step["marks_gained"]),
        str(step["action_id"]),
    ))
    for number, step in enumerate(selected, start=1):
        step["step_number"] = number
    return selected


def phases(steps: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for step in steps:
        start = int(step["start_month"])
        phase_number = ((max(1, start) - 1) // 3) + 1
        grouped.setdefault(phase_number, []).append(step)

    result = []
    for phase_number in sorted(grouped):
        phase_steps = grouped[phase_number]
        first_month = (phase_number - 1) * 3 + 1
        last_month = phase_number * 3
        result.append({
            "phase_number": phase_number,
            "heading": f"Phase {phase_number}: months {first_month}-{last_month}",
            "action_count": len(phase_steps),
            "cost": sum(float(step["cost"]) for step in phase_steps),
            "faculty_hours": sum(float(step["faculty_hours"]) for step in phase_steps),
            "steps": phase_steps,
        })
    return result


def key_insights(problem: NBAProblem, eval: EvaluationResult) -> List[str]:
    notes: List[str] = []
    baseline_total = float(sum(problem.s0))
    plan_added = eval.f1_total_score - baseline_total
    target_difference = eval.f1_total_score - problem.min_accreditation_score
    if target_difference >= 0:
        notes.append(
            f"The plan adds {_number(plan_added)} marks and is "
            f"{_number(target_difference)} marks above the required total."
        )
    else:
        notes.append(
            f"The plan adds {_number(plan_added)} marks and is "
            f"{_number(abs(target_difference))} marks below the required total."
        )

    efficiencies = []
    for action in problem.actions:
        units = int(eval.x[action.index])
        if units <= 0:
            continue
        gain = action.score_gain * units
        cost = action.cost_lakh * units
        efficiency = gain / cost if cost > 0 else (float("inf") if gain > 0 else 0.0)
        efficiencies.append((efficiency, action))
    efficiencies.sort(key=lambda item: (-item[0], item[1].action_id))
    top_actions = efficiencies[:3]
    if top_actions:
        top_text = "; ".join(
            f"{action.action_id} ({_number(efficiency)} marks per lakh)"
            for efficiency, action in top_actions
        )
        notes.append(f"The three actions with the best marks per lakh are: {top_text}.")

    criterion_gains = []
    for criterion in problem.criteria:
        idx = criterion.id - 1
        gain = float(eval.criteria_scores[idx] - problem.s0[idx])
        criterion_gains.append((gain, criterion))
    criterion_gains.sort(key=lambda item: (-item[0], item[1].id))
    if criterion_gains:
        gains_text = ", ".join(
            f"C{criterion.id} {_number(gain)}"
            for gain, criterion in criterion_gains[:3]
        )
        notes.append(f"The criteria gaining the most are: {gains_text}.")

    below = [
        f"C{criterion.id} ({criterion.name}: {_number(float(eval.criteria_scores[criterion.id - 1]))} "
        f"of {_number(float(problem.L[criterion.id - 1]))} minimum)"
        for criterion in problem.criteria
        if float(eval.criteria_scores[criterion.id - 1]) < float(problem.L[criterion.id - 1]) - 1e-9
    ]
    if below:
        notes.append("Criteria still below minimum: " + "; ".join(below) + ".")

    if len(eval.monthly_hours):
        peak_load = float(eval.monthly_hours.max())
        peak_month = int(eval.monthly_hours.argmax()) + 1
        percent = 100.0 * peak_load / problem.monthly_capacity if problem.monthly_capacity > 0 else 0.0
        notes.append(
            f"Month {peak_month} has the highest faculty load: {_number(peak_load, 0)} hours "
            f"({_number(percent, 0)}% of the {_number(problem.monthly_capacity, 0)}-hour monthly capacity)."
        )

    budget_left = problem.budget - eval.f2_cost_lakh
    if budget_left >= 0:
        notes.append(f"Budget left: Rs {_number(budget_left)} lakh.")
    else:
        notes.append(f"Budget exceeded by Rs {_number(abs(budget_left))} lakh.")

    missing_evidence = []
    for action in problem.actions:
        if int(eval.x[action.index]) <= 0 or action.is_evidence:
            continue
        evidence_present = any(
            int(eval.x[ea.index]) > 0
            for ea in problem.actions
            if ea.crit_id == action.crit_id and ea.is_evidence
        )
        if not evidence_present:
            missing_evidence.append(action.action_id)
    if missing_evidence:
        notes.append(
            "Warning: evidence packages are missing for selected actions "
            + ", ".join(sorted(set(missing_evidence)))
            + "."
        )

    if float(eval.violations.get("deadline", 0.0)) > 1e-9:
        notes.append("Warning: one or more selected actions violates the planning deadline.")
    if float(eval.violations.get("monthly_hours", 0.0)) > 1e-9:
        notes.append("Warning: monthly faculty capacity is exceeded.")
    if float(eval.violations.get("budget", 0.0)) > 1e-9:
        notes.append("Warning: the plan exceeds the available budget.")
    if float(eval.violations.get("prerequisites", 0.0)) > 1e-9:
        notes.append("Warning: prerequisite requirements are not fully satisfied.")
    return notes


def plan_memo_markdown(problem: NBAProblem, eval: EvaluationResult, title: str) -> str:
    lines = [f"# {title}", "", "## Recommended upgrade plan", ""]
    added = eval.f1_total_score - float(sum(problem.s0))
    difference = eval.f1_total_score - problem.min_accreditation_score
    if difference >= 0:
        lines.append(
            f"Execute this plan to add {_number(added)} marks and finish "
            f"{_number(difference)} marks above the required total."
        )
    else:
        lines.append(
            f"Execute this plan to add {_number(added)} marks, but it remains "
            f"{_number(abs(difference))} marks below the required total."
        )
    lines.extend([
        "",
        "## Where the marks come from",
        "",
        "| Criterion | Current | After plan | Maximum | Minimum | Status | Gain |",
        "|---|---:|---:|---:|---:|---|---:|",
    ])
    for row in criterion_table(problem, eval):
        lines.append(
            f"| C{row['criterion_number']} {row['name']} | "
            f"{_number(float(row['current_marks']))} | "
            f"{_number(float(row['marks_after']))} | "
            f"{_number(float(row['maximum']))} | "
            f"{_number(float(row['minimum_level']))} | "
            f"{row['status']} | {_number(float(row['gain']))} |"
        )

    lines.extend(["", "## Your action plan, step by step", ""])
    for phase in phases(action_steps(problem, eval)):
        lines.append(
            f"### {phase['heading']} - {phase['action_count']} actions, "
            f"Rs {_number(float(phase['cost']))} lakh, "
            f"{_number(float(phase['faculty_hours']), 0)} faculty hours"
        )
        for step in phase["steps"]:
            lines.append(
                f"{step['step_number']}. **{step['action_id']}: {step['name']}** - "
                f"{step['instruction']}"
            )
        lines.append("")

    lines.extend(["## What to watch out for", ""])
    for note in key_insights(problem, eval):
        lines.append(f"- {note}")
    return "\n".join(lines)
