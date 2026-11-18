"""Tests for actionable NBA plan recommendations."""

from pathlib import Path
from copy import copy

import numpy as np

from src.baseline.greedy import GreedyBaseline
from src.problem import NBAProblem
from src.recommendation import (
    _number,
    action_steps,
    key_insights,
    phases,
    plan_memo_markdown,
)


def test_recommendation_for_feasible_greedy_plan() -> None:
    problem = NBAProblem(Path("experiments/config_small.yaml"))
    result = GreedyBaseline(problem, seed=7).run(n_samples=1)
    feasible = [solution for solution in result.pareto_solutions if solution.is_feasible]
    assert feasible
    evaluation = feasible[0]

    steps = action_steps(problem, evaluation)
    assert [step["step_number"] for step in steps] == list(range(1, len(steps) + 1))
    assert all(
        (steps[i]["start_month"], -steps[i]["marks_gained"], steps[i]["action_id"])
        <= (steps[i + 1]["start_month"], -steps[i + 1]["marks_gained"], steps[i + 1]["action_id"])
        for i in range(len(steps) - 1)
    )

    selected_ids = [problem.actions[index].action_id for index in np.flatnonzero(evaluation.x)]
    step_ids = [step["action_id"] for step in steps]
    assert sorted(selected_ids) == sorted(step_ids)
    assert len(step_ids) == len(set(step_ids))

    from src.recommendation import criterion_table

    table = criterion_table(problem, evaluation)
    assert np.isclose(sum(row["current_marks"] for row in table), float(sum(problem.s0)))
    assert np.isclose(sum(row["marks_after"] for row in table), evaluation.f1_total_score)

    plan_phases = phases(steps)
    assert np.isclose(
        sum(float(phase["cost"]) for phase in plan_phases),
        evaluation.f2_cost_lakh,
    )

    memo = plan_memo_markdown(problem, evaluation, "Test plan")
    for action_id in selected_ids:
        assert action_id in memo


def test_number_keeps_integer_zeros_and_strips_decimal_zeros() -> None:
    assert _number(300) == "300"
    assert _number(150) == "150"
    assert _number(1500) == "1500"
    assert _number(5.70) == "5.7"
    assert _number(2.0) == "2"
    assert _number(0) == "0"


def test_first_action_starts_in_month_one_and_phases_are_singular() -> None:
    problem = NBAProblem(Path("experiments/config_small.yaml"))
    evaluation = problem.evaluate(np.zeros(problem.num_actions, dtype=np.int32))
    steps = action_steps(problem, evaluation)
    assert steps == []

    first_action = problem.actions[0]
    plan = np.zeros(problem.num_actions, dtype=np.int32)
    plan[first_action.index] = 1
    evaluation = problem.evaluate(plan)
    steps = action_steps(problem, evaluation)
    assert steps
    assert steps[0]["start_month"] == first_action.earliest_start + 1
    assert steps[0]["start_month"] >= 1
    assert all(step["start_month"] != 0 for step in steps)

    one_step = phases([{
        "start_month": 1,
        "cost": 0.1,
        "faculty_hours": 35,
        "action_id": first_action.action_id,
        "marks_gained": 0,
    }])
    assert len(one_step) == 1
    assert "1 action" in one_step[0]["heading"]


def test_infeasible_memo_and_notes_include_reasons_and_capacity() -> None:
    problem = NBAProblem(Path("experiments/config_small.yaml"))
    plan = np.zeros(problem.num_actions, dtype=np.int32)
    evaluation = problem.evaluate(plan)
    evaluation = copy(evaluation)
    evaluation.is_feasible = False
    evaluation.violations = dict(evaluation.violations)
    evaluation.violations["budget"] = 1.0

    memo = plan_memo_markdown(problem, evaluation, "Infeasible test")
    notes = key_insights(problem, evaluation)
    assert memo.startswith("# Infeasible test")
    assert "NOT FEASIBLE" in memo
    assert "NOT FEASIBLE" in notes
    assert str(int(problem.monthly_capacity)) in " ".join(notes) or str(int(problem.monthly_capacity)) in memo

    selected_ids = [
        action.action_id
        for action in problem.actions
        if int(evaluation.x[action.index]) > 0
    ]
    for action_id in selected_ids:
        assert action_id in memo
