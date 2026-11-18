"""Tests for actionable NBA plan recommendations."""

from pathlib import Path

import numpy as np

from src.baseline.greedy import GreedyBaseline
from src.problem import NBAProblem
from src.recommendation import (
    action_steps,
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
