"""Smoke test for the Streamlit plan flow."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def click_button(app: AppTest, label: str) -> None:
    for button in app.button:
        if getattr(button, "label", "") == label:
            button.click()
            return
    raise AssertionError(f"Button not found: {label}")


@pytest.mark.slow
def test_app_recommendation_flow() -> None:
    started = time.monotonic()
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20)
    app.run(timeout=20)
    assert not app.exception

    click_button(app, "Get started ➔")
    app.run(timeout=20)
    assert not app.exception

    small_label = "experiments/config_small.yaml"
    for selectbox in app.selectbox:
        if small_label in selectbox.options:
            selectbox.select(small_label)
            app.run(timeout=20)
            break

    click_button(app, "Continue to Generate plans ➔")
    app.run(timeout=20)
    assert not app.exception

    solver_label = "HiGHS MILP (Exact Mathematical Programming)"
    multiselect = app.multiselect[0]
    multiselect.set_value([])
    multiselect.set_value([solver_label])
    app.run(timeout=20)

    click_button(app, "Generate plans")
    app.run(timeout=120)
    assert not app.exception
    assert time.monotonic() - started < 120

    click_button(app, "Most even progress")
    app.run(timeout=20)
    click_button(app, "Continue to Your plan ➔")
    app.run(timeout=20)
    assert not app.exception
    assert any("Recommended upgrade plan" in str(markdown.value) for markdown in app.markdown)
