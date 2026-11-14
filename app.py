"""
NBA Accreditation Improvement Planning: Multi-Objective Decision Support System.
Streamlit Frontend Application for Academic & Institutional Optimization.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.algorithms.moead import MOEADOptimizer
from src.algorithms.nsga2 import NSGA2Optimizer, OptimizationResult
from src.algorithms.spea2 import SPEA2Optimizer
from src.baseline.greedy import GreedyBaseline
from src.baseline.milp_solver import HiGHSMILPSolver
from src.baseline.random_search import RandomSearchBaseline
from src.hybrid.memetic_nsga2 import MemeticNSGA2Optimizer
from src.metrics import (
    compute_coverage,
    compute_hypervolume,
    compute_igd,
    compute_spacing,
    compute_spread,
)
from src.problem import Action, Criterion, EvaluationResult, NBAProblem

# Configure Streamlit page
st.set_page_config(
    page_title="NBA Accreditation Planning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling and dark/light theme compatibility
st.markdown("""
<style>
    /* Center and constrain main layout width, clear fixed header */
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 1100px;
        padding-top: 4.5rem !important;
        padding-bottom: 3rem;
        margin-left: auto;
        margin-right: auto;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
        line-height: 1.25;
    }
    .sub-title {
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
        opacity: 0.85;
    }
    .step-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #2563EB;
        margin-bottom: 0.2rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.75rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        padding: 0.85rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        opacity: 0.8;
    }
    .metric-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .delta-good { color: #10B981; }
    .delta-bad { color: #EF4444; }
    .landing-step-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.75rem;
        padding: 1.25rem;
        height: 100%;
        min-height: 190px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_problem(config_file: str) -> NBAProblem:
    """Load and cache the problem instance."""
    return NBAProblem(config_file)


# ==============================================================================
# NAVIGATION CONTROLLER, PERSISTENCE & DATASET HELPERS
# ==============================================================================

_CONFIG_PATHS = [
    "experiments/config.yaml",
    "experiments/config_small.yaml",
    "experiments/config_small_f4.yaml",
]
available_configs = _CONFIG_PATHS


def _get_dataset_name(cfg_path: str, num_actions: Optional[int] = None) -> str:
    """Return plain-English dataset display name with dynamic action count."""
    if "custom" in cfg_path.lower():
        return "Custom department numbers"
    if num_actions is None:
        if "small_f4" in cfg_path:
            num_actions = 21
        elif "small" in cfg_path:
            num_actions = 20
        else:
            num_actions = 45

    if "small_f4" in cfg_path:
        return f"Small dataset with continuous improvement ({num_actions} actions, 10 criteria)"
    elif "small" in cfg_path:
        return f"Small dataset ({num_actions} actions, 9 criteria)"
    else:
        return f"Full dataset ({num_actions} actions, 10 criteria)"


def _scroll_to_top() -> None:
    """Scroll main Streamlit container to the top when navigating between steps."""
    if st.session_state.get("_scroll_top", False):
        st.session_state["_scroll_top"] = False
        scroll_nonce = st.session_state.get("_scroll_nonce", 0) + 1
        st.session_state["_scroll_nonce"] = scroll_nonce
        st.components.v1.html(
            f"""
            <script>
            (function() {{
                try {{
                    const doc = window.parent.document;
                    const targets = [
                        doc.querySelector('[data-testid="stMain"]'),
                        doc.querySelector('[data-testid="stAppViewContainer"]'),
                        doc.querySelector('section.main'),
                        doc.documentElement,
                        window.parent
                    ];
                    for (const t of targets) {{
                        if (t) {{
                            if (typeof t.scrollTo === 'function') {{
                                t.scrollTo({{top: 0, left: 0, behavior: 'instant'}});
                            }}
                            if (t.scrollTop !== undefined) {{
                                t.scrollTop = 0;
                            }}
                        }}
                    }}
                }} catch (e) {{
                    console.debug('Scroll to top error:', e);
                }}
            }})();
            </script>
            <!-- nonce: {scroll_nonce} -->
            """,
            height=0,
            width=0,
        )


@st.cache_resource
def _cleanup_old_custom_files_daily() -> None:
    """One-time-per-process cleanup for custom config and profile temp files older than 24h."""
    try:
        now = time.time()
        max_age = 24 * 3600
        for p in Path("experiments").glob(".custom_cfg_*.yaml"):
            try:
                if now - p.stat().st_mtime > max_age:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        for p in Path("data").glob(".custom_prof_*.csv"):
            try:
                if now - p.stat().st_mtime > max_age:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass


_cleanup_old_custom_files_daily()


def _cleanup_custom_files(cfg_hash: Optional[str] = None) -> None:
    """Clean up custom temporary problem files."""
    if cfg_hash:
        p_cfg = Path(f"experiments/.custom_cfg_{cfg_hash}.yaml")
        p_prof = Path(f"data/.custom_prof_{cfg_hash}.csv")
        try:
            if p_cfg.exists():
                p_cfg.unlink()
            if p_prof.exists():
                p_prof.unlink()
        except Exception:
            pass


def _get_custom_problem_and_config(base_cfg_path: str, custom_data: Dict[str, Any]) -> Tuple[NBAProblem, str]:
    """Generate a custom config and profile file for the active custom inputs, returning (problem, cfg_path)."""
    raw_key = json.dumps({"base": base_cfg_path, "data": custom_data}, sort_keys=True)
    cfg_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]

    custom_cfg_path = Path(f"experiments/.custom_cfg_{cfg_hash}.yaml")
    custom_prof_path = Path(f"data/.custom_prof_{cfg_hash}.csv")

    if not custom_cfg_path.exists() or not custom_prof_path.exists():
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)

        base_prof_path = Path(base_cfg["data"]["profile_file"])
        df_base_prof = pd.read_csv(base_prof_path, comment="#")

        score_map = {int(k): float(v) for k, v in custom_data.get("scores", {}).items()}
        for idx, row in df_base_prof.iterrows():
            cid = int(row["crit_id"])
            if cid in score_map:
                df_base_prof.at[idx, "current_marks"] = score_map[cid]

        df_base_prof.to_csv(custom_prof_path, index=False)

        custom_cfg = {
            "dataset_metadata": {
                "name": f"Custom Department Numbers ({Path(base_cfg_path).stem})",
                "is_synthetic": False,
                "total_actions": base_cfg.get("dataset_metadata", {}).get("total_actions", 45),
                "total_criteria": 10,
            },
            "planning_parameters": {
                "budget_lakh": float(custom_data["budget"]),
                "planning_horizon_months": int(custom_data["horizon"]),
                "monthly_faculty_capacity": float(custom_data["capacity"]),
                "min_accreditation_score": float(custom_data["target_score"]),
            },
            "data": {
                "criteria_file": base_cfg["data"]["criteria_file"],
                "subcriteria_file": base_cfg["data"].get("subcriteria_file", "data/nba_subcriteria.csv"),
                "profile_file": f"data/.custom_prof_{cfg_hash}.csv",
                "actions_file": base_cfg["data"]["actions_file"],
            },
        }
        with open(custom_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(custom_cfg, f)

    prob = load_problem(str(custom_cfg_path))
    return prob, str(custom_cfg_path)


# The 4 main sequential stages and their human-readable labels
MAIN_STEPS = [
    ("status", "1. Department"),
    ("generate", "2. Generate Plans"),
    ("compare", "3. Compare"),
    ("plan", "4. Your Plan"),
]
MAIN_STEP_KEYS = [k for k, _ in MAIN_STEPS]

if "step" not in st.session_state:
    st.session_state["step"] = "landing"

if "active_config" not in st.session_state:
    st.session_state["active_config"] = "experiments/config.yaml"

if "selected_plan_override" not in st.session_state:
    st.session_state["selected_plan_override"] = None

if "results_cache" not in st.session_state:
    st.session_state["results_cache"] = {}

if "whatif_nonce" not in st.session_state:
    st.session_state["whatif_nonce"] = 0

if "whatif_started_from" not in st.session_state:
    st.session_state["whatif_started_from"] = ""


def _clear_dataset_state(keep_config: Optional[str] = None) -> None:
    """Clear downstream planning results when dataset changes or on start-over."""
    st.session_state["selected_plan_override"] = None
    st.session_state["latest_res"] = None
    if "results_cache" in st.session_state:
        st.session_state["results_cache"] = {}
    if "whatif_vector" in st.session_state:
        del st.session_state["whatif_vector"]
    st.session_state["whatif_nonce"] = st.session_state.get("whatif_nonce", 0) + 1
    st.session_state["plan_archetype_choice"] = None
    st.session_state["plan_custom_solution_idx"] = 0
    st.session_state["whatif_started_from"] = ""


@st.dialog("Start over with a new scenario?")
def _confirm_start_over_dialog() -> None:
    """Confirmation modal dialog before clearing generated plans and returning to landing."""
    st.write(
        "This will return you to the beginning and clear any generated plans. "
        "Are you sure you want to start over?"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, start over", type="primary", use_container_width=True):
            _clear_dataset_state()
            st.session_state["step"] = "landing"
            st.session_state["_scroll_top"] = True
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def _on_config_change() -> None:
    """Callback when dataset is changed."""
    new_cfg = st.session_state.get("_dataset_selector_widget")
    if new_cfg and new_cfg != st.session_state.get("active_config"):
        st.session_state["active_config"] = new_cfg
        st.session_state["custom_inputs"] = None
        st.session_state["use_custom_inputs"] = False
        _clear_dataset_state()


def set_step(new_step: str) -> None:
    """Callback to transition between guided steps."""
    st.session_state["step"] = new_step
    st.session_state["_scroll_top"] = True


# State keep-alive pattern: ensure inputs don't drop when widgets unmount between steps
PERSISTENT_WIDGET_KEYS = [
    "opt_pop_size",
    "opt_max_gens",
    "opt_crossover_p",
    "opt_mutation_p",
    "opt_seed",
    "opt_chosen_algs",
    "catalog_sel_crit",
    "catalog_sel_type",
    "catalog_search_kw",
    "catalog_sel_inspect",
    "compare_plan_sel_idx",
    "plan_archetype_choice",
    "plan_custom_solution_idx",
    "use_custom_inputs",
]
for _k in PERSISTENT_WIDGET_KEYS:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]


# Load active problem safely
current_config = st.session_state["active_config"]
is_custom_active = bool(st.session_state.get("custom_inputs"))

if is_custom_active:
    try:
        problem, active_problem_cfg = _get_custom_problem_and_config(
            current_config, st.session_state["custom_inputs"]
        )
    except Exception as e:
        st.error("Could not load custom department numbers. Reverting to dataset defaults.")
        with st.expander("Technical details"):
            st.code(str(e))
        st.session_state["custom_inputs"] = None
        is_custom_active = False
        problem = load_problem(current_config)
        active_problem_cfg = current_config
else:
    try:
        problem = load_problem(current_config)
        active_problem_cfg = current_config
    except Exception as e:
        st.error(
            "Could not load the selected dataset. "
            "Please check that its data files are present, then try again."
        )
        with st.expander("Technical details"):
            st.code(str(e))
        st.stop()

if active_problem_cfg not in st.session_state["results_cache"]:
    st.session_state["results_cache"][active_problem_cfg] = {}

if "whatif_vector" not in st.session_state or len(st.session_state["whatif_vector"]) != problem.num_actions:
    st.session_state["whatif_vector"] = np.zeros(problem.num_actions, dtype=int)


def get_pareto_solutions(prob: NBAProblem) -> Dict[str, List[EvaluationResult]]:
    """Derive available Pareto solutions for the current configuration from session state cache only."""
    sols_dict: Dict[str, List[EvaluationResult]] = {}
    config_cache = st.session_state.get("results_cache", {}).get(active_problem_cfg, {})
    for alg_name, res in config_cache.items():
        if getattr(res, "pareto_solutions", None):
            valid = [s for s in res.pareto_solutions if len(s.x) == prob.num_actions]
            if valid:
                sols_dict[alg_name] = valid
    return sols_dict


def get_all_sols_flat(prob: NBAProblem) -> List[Tuple[str, EvaluationResult]]:
    """Derive flattened (algorithm, solution) list for current problem."""
    flat_list: List[Tuple[str, EvaluationResult]] = []
    p_dict = get_pareto_solutions(prob)
    for alg, sols in p_dict.items():
        for s in sols:
            if len(s.x) == prob.num_actions:
                flat_list.append((alg, s))
    return flat_list


def _run_algorithm(
    chosen_alg: str,
    prob: NBAProblem,
    pop_size: int,
    max_gens: int,
    crossover_p: float,
    mutation_p: float,
    seed: int,
) -> OptimizationResult:
    """Build and run the optimizer for one chosen algorithm, returning OptimizationResult."""
    start_t = time.time()
    if "Hybrid" in chosen_alg:
        opt = MemeticNSGA2Optimizer(
            problem=prob,
            pop_size=pop_size,
            max_generations=max_gens,
            crossover_prob=crossover_p,
            mutation_prob=mutation_p,
            seed=seed,
        )
        res = opt.run()
    elif "NSGA-II" in chosen_alg:
        opt = NSGA2Optimizer(
            problem=prob,
            pop_size=pop_size,
            max_generations=max_gens,
            crossover_prob=crossover_p,
            mutation_prob=mutation_p,
            seed=seed,
        )
        res = opt.run()
    elif "SPEA2" in chosen_alg:
        opt = SPEA2Optimizer(
            problem=prob,
            pop_size=pop_size,
            archive_size=pop_size,
            max_generations=max_gens,
            crossover_prob=crossover_p,
            mutation_prob=mutation_p,
            seed=seed,
        )
        res = opt.run()
    elif "MOEA/D" in chosen_alg:
        opt = MOEADOptimizer(
            problem=prob,
            pop_size=pop_size,
            n_neighbors=min(15, pop_size),
            max_generations=max_gens,
            crossover_prob=crossover_p,
            mutation_prob=mutation_p,
            seed=seed,
        )
        res = opt.run()
    elif "HiGHS" in chosen_alg:
        solver = HiGHSMILPSolver(prob)
        sols = solver.solve_epsilon_constraint_front(n_budget_steps=8, n_hour_steps=6)
        res = OptimizationResult(
            algorithm_name="HiGHS-MILP",
            seed=seed,
            elapsed_time=time.time() - start_t,
            final_population=sols,
            pareto_solutions=sols,
            history=[],
            total_evaluations=len(sols),
        )
    elif "Greedy" in chosen_alg:
        opt_g = GreedyBaseline(problem=prob, seed=seed)
        res = opt_g.run()
    else:
        opt_r = RandomSearchBaseline(problem=prob, seed=seed)
        res = opt_r.run(n_samples=300)
    return res


_CANON_ALG: Dict[str, str] = {
    "Hybrid-Memetic": "Hybrid-Memetic",
    "Hybrid": "Hybrid-Memetic",
    "NSGA-II": "NSGA-II",
    "NSGA": "NSGA-II",
    "SPEA2": "SPEA2",
    "MOEA/D": "MOEA/D",
    "MOEA": "MOEA/D",
    "HiGHS-MILP": "HiGHS-MILP",
    "HiGHS": "HiGHS-MILP",
    "MILP-HiGHS": "HiGHS-MILP",
    "Greedy-Baseline": "Greedy-Baseline",
    "Greedy": "Greedy-Baseline",
    "Random-Baseline": "Random-Baseline",
    "Random": "Random-Baseline",
}


def _canon_alg(name: str) -> str:
    """Return the canonical algorithm key for dict storage."""
    return _CANON_ALG.get(name, name)


def _invalidate_plan_selection() -> None:
    """Clear plan position state whenever the plan list may have changed."""
    st.session_state["selected_plan_override"] = None
    st.session_state["plan_archetype_choice"] = None
    st.session_state["plan_custom_solution_idx"] = 0


def _select_plan(alg: str, x: List[int]) -> None:
    """Store plan selection and mark archetype as 'Your chosen plan'."""
    st.session_state["selected_plan_override"] = {"alg": _canon_alg(alg), "x": list(x)}
    st.session_state["plan_archetype_choice"] = "Your chosen plan"


def _choose_and_continue(alg: str, x: List[int]) -> None:
    """Store plan selection and navigate to plan stage."""
    _select_plan(alg, x)
    st.session_state["step"] = "plan"


def _resolve_selected_plan(prob: "NBAProblem") -> Optional[Tuple[str, "EvaluationResult"]]:
    """Find the selected plan in get_all_sols_flat by alg + vector match. Returns None if not found."""
    sel = st.session_state.get("selected_plan_override")
    if not sel:
        return None
    target_alg = sel["alg"]
    target_x = sel["x"]
    for alg, ev in get_all_sols_flat(prob):
        if _canon_alg(alg) == target_alg and list(ev.x) == target_x:
            return (alg, ev)
    # Not found — invalidate stale selection
    st.session_state["selected_plan_override"] = None
    return None


def _pick_archetype(
    all_sols_flat: List[Tuple[str, "EvaluationResult"]],
    kind: str,
    min_score: float,
) -> Optional[Tuple[str, "EvaluationResult"]]:
    """
    Extract existing archetype logic from the plan stage.
    kind: 'score' | 'cost' | 'balance'
    """
    if not all_sols_flat:
        return None
    if kind == "score":
        sorted_f1 = sorted(all_sols_flat, key=lambda pair: pair[1].f1_total_score, reverse=True)
        return sorted_f1[0]
    elif kind == "cost":
        qual = [p for p in all_sols_flat if p[1].f1_total_score >= min_score - 1e-4]
        pool = qual if qual else all_sols_flat
        sorted_f2 = sorted(pool, key=lambda pair: pair[1].f2_cost_lakh)
        return sorted_f2[0]
    elif kind == "balance":
        sorted_f4 = sorted(all_sols_flat, key=lambda pair: pair[1].f4_worst_criterion_ratio, reverse=True)
        return sorted_f4[0]
    return None


def _open_whatif_with_plan(x: List[int], title: str = "") -> None:
    """Seed the What-If sandbox with a chosen plan vector and navigate to whatif stage."""
    if len(x) != problem.num_actions:
        return
    st.session_state["whatif_vector"] = np.array(x, dtype=int)
    st.session_state["whatif_nonce"] = st.session_state.get("whatif_nonce", 0) + 1
    st.session_state["whatif_started_from"] = title
    st.session_state["step"] = "whatif"
    st.session_state["_scroll_top"] = True


baseline_score = float(np.sum(problem.s0))
step = st.session_state.get("step", "landing")
_scroll_to_top()


# ==============================================================================
# SIDEBAR: DISPLAY CONSTRAINTS ON NON-LANDING STAGES ONLY
# ==============================================================================
if step != "landing":
    st.sidebar.image("https://img.icons8.com/fluency/96/diploma.png", width=56)
    st.sidebar.title("NBA Accreditation")
    st.sidebar.caption("Strategic Improvement Planning System")
    if is_custom_active:
        st.sidebar.markdown('<span style="background-color: #DBEAFE; color: #1E40AF; padding: 0.15rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 700;">USING YOUR NUMBERS</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Active Dataset Constraints")
    st.sidebar.markdown(f"• **Budget Limit:** ₹`{problem.budget:.1f}` Lakh")
    st.sidebar.markdown(f"• **Faculty Hours/Mo:** `{problem.monthly_capacity:.0f}` hrs")
    st.sidebar.markdown(f"• **Planning Horizon:** `{problem.planning_horizon}` months")
    st.sidebar.markdown(f"• **Target Score:** `{problem.min_accreditation_score:.0f}` / 1000")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Department Baseline")
    evidence_count = sum(1 for a in problem.actions if a.is_evidence)
    st.sidebar.markdown(f"• **Baseline Score:** `{baseline_score:.0f}` / 1000")
    st.sidebar.markdown(f"• **Score Gap:** `+{max(0.0, problem.min_accreditation_score - baseline_score):.0f}` Marks")
    st.sidebar.markdown(f"• **Actions in Catalog:** {problem.num_actions} ({evidence_count} Evidence, {problem.num_actions - evidence_count} Interventions)")
    st.sidebar.markdown("---")
    with st.sidebar.expander("What do these terms mean?", expanded=False):
        st.markdown(
            "• **Dataset:** A department profile containing baseline scores across 10 criteria, eligible improvement actions, and institutional budget and capacity limits.\n\n"
            "• **Criterion (subject area):** One of the 10 official evaluation areas covering curriculum, faculty, facilities, student performance, and continuous improvement.\n\n"
            "• **Marks:** Numerical evaluation points awarded under accreditation guidelines, summing to 1,000 maximum marks across all 10 criteria.\n\n"
            "• **Baseline score:** Your department's current total marks before implementing any new improvement actions.\n\n"
            "• **Target score:** The 750 marks required out of 1,000 to achieve full 5-6 year accreditation.\n\n"
            "• **Improvement action:** A specific academic, infrastructural, or policy intervention that increases criterion marks when completed.\n\n"
            "• **Evidence package:** A mandatory documentation action that verifies existing practices and unlocks subsequent interventions.\n\n"
            "• **Prerequisite:** An action that must finish before another dependent action can begin execution.\n\n"
            "• **Plan:** A selected combination of improvement actions that balances total marks gained, budget expenditure, and faculty workload.\n\n"
            "• **Budget (lakh):** The maximum capital and operational expenditure in Indian Lakh Rupees allocated for improvements over the planning period.\n\n"
            "• **Faculty hours:** Total staff hours required across faculty members to execute and document improvement actions.\n\n"
            "• **Planning horizon:** The fixed time frame (12 months) within which all scheduled actions must finish.\n\n"
            "• **Feasible plan:** A plan that strictly satisfies budget limits, monthly faculty capacity caps, deadlines, minimum criteria scores, and prerequisites.\n\n"
            "• **Quick/Balanced/Thorough:** Search presets that control how many candidate action plans the optimizer evaluates before presenting recommendations.\n\n"
            "• **Subject balance:** A measure of equity ensuring all criteria receive attention rather than concentrating gains in only a few areas."
        )
else:
    # On landing, inject CSS to collapse/hide sidebar cleanly
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# HEADER & BREADCRUMB PROGRESS BAR
# ==============================================================================

if step == "landing":
    # Clean Landing Screen
    st.markdown('<div class="main-title">NBA Accreditation Strategic Improvement Planning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Identify optimal improvement actions to reach full accreditation '
        'within institutional budget, faculty hours, and schedule limits.</div>',
        unsafe_allow_html=True,
    )

    c_btn1, c_btn2, c_btn3 = st.columns([1, 1.2, 1])
    with c_btn2:
        st.button("Get started ➔", type="primary", use_container_width=True, on_click=set_step, args=("status",))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### How it works in 3 simple steps:")
    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        st.markdown("""
        <div class="landing-step-card">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem;">1. Assess status</div>
            <div style="font-size: 0.88rem; line-height: 1.45; opacity: 0.85;">
                Review your department's baseline scores across the 10 criteria, marks still needed, and catalog of eligible actions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with l_col2:
        st.markdown("""
        <div class="landing-step-card">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">⚡</div>
            <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem;">2. Generate plans</div>
            <div style="font-size: 0.88rem; line-height: 1.45; opacity: 0.85;">
                Let the search engine explore thousands of action combinations using one-click presets (Quick, Balanced, or Thorough).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with l_col3:
        st.markdown("""
        <div class="landing-step-card">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🎯</div>
            <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem;">3. Compare and blueprint</div>
            <div style="font-size: 0.88rem; line-height: 1.45; opacity: 0.85;">
                Inspect trade-offs between cost, faculty hours, and marks. Pick your preferred plan and download the implementation roadmap.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("**Who is this for?** Department heads and accreditation coordinators. No technical background or optimization expertise needed.")
    st.caption("Plans are advisory suggestions generated from your dataset; review and approve them according to your institution's governance before taking action.")

else:
    # Top Bar: Stage Progress Indicator & Start Over Button
    t_left, t_right = st.columns([4, 1.2])
    with t_right:
        if st.button("Start over", key="top_bar_start_over", use_container_width=True, help="Reset workflow to landing page and clear generated plans"):
            _confirm_start_over_dialog()

    with t_left:
        # Build 4-step progress breadcrumbs
        p_html = ['<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.2rem; flex-wrap: wrap;">']
        current_idx = MAIN_STEP_KEYS.index(step) if step in MAIN_STEP_KEYS else -1
        for i, (k, label) in enumerate(MAIN_STEPS):
            if step == k:
                p_html.append(f'<span class="step-pill step-pill-active">{label}</span>')
            elif current_idx > i:
                p_html.append(f'<span class="step-pill step-pill-done">✓ {label}</span>')
            else:
                p_html.append(f'<span class="step-pill step-pill-upcoming">{label}</span>')
            if i < len(MAIN_STEPS) - 1:
                p_html.append('<span style="color: rgba(128, 128, 128, 0.5); font-size: 0.85rem;">➔</span>')
        p_html.append('</div>')
        st.markdown("".join(p_html), unsafe_allow_html=True)


# ==============================================================================
# STAGE 1: DEPARTMENT STATUS & ACTIONS
# ==============================================================================
if step == "status":
    st.markdown('<div class="step-label">STEP 1 OF 4</div>', unsafe_allow_html=True)
    st.subheader("Your department")
    st.markdown("Choose a dataset, then see where your department stands against the NBA accreditation target.")
    st.caption("New to NBA accreditation? See 'What do these terms mean?' in the sidebar.")

    # Dataset Selector in bordered container
    with st.container(border=True):
        cfg_col1, cfg_col2 = st.columns([1.5, 2])
        with cfg_col1:
            st.session_state.setdefault(
                "_dataset_selector_widget",
                st.session_state["active_config"] if st.session_state["active_config"] in available_configs else available_configs[0],
            )
            st.selectbox(
                "Choose a dataset",
                available_configs,
                key="_dataset_selector_widget",
                format_func=lambda p: _get_dataset_name(p, problem.num_actions if p == st.session_state.get("active_config") else None),
                on_change=_on_config_change,
            )
        with cfg_col2:
            covered_crits = set(a.crit_id for a in problem.actions)
            zero_crits = [c for c in problem.criteria if c.id not in covered_crits]
            if zero_crits:
                zero_str = ", ".join(f"{c.name} (Criterion {c.id})" for c in zero_crits)
                dyn_caption = (
                    f"Contains {problem.num_actions} actions covering {len(covered_crits)} of {len(problem.criteria)} criteria. "
                    f"No actions are available for {zero_str}, so its score cannot improve in this dataset."
                )
            else:
                dyn_caption = (
                    f"Contains {problem.num_actions} actions covering all {len(problem.criteria)} criteria with full institutional choices."
                )
            st.caption(dyn_caption)
        st.caption(
            "A dataset defines your department's baseline scores, institutional budget and "
            "faculty capacity limits, planning horizon, and the catalog of eligible improvement actions."
        )

    # Custom department inputs container (Part G)
    with st.container(border=True):
        st.markdown("##### Use your department's own numbers (optional)")
        st.caption("Start from the dataset above, then replace its numbers with yours. Leave this off to use the dataset as it is.")
        st.session_state.setdefault("use_custom_inputs", is_custom_active)
        use_custom = st.toggle("Enable custom numbers", key="use_custom_inputs")

        if use_custom:
            with st.form("custom_inputs_form"):
                st.markdown("###### 1. Baseline scores by criterion")
                st.caption("Enter your department's current marks for each criterion. Maximum marks and minimum required levels are set by official accreditation guidelines.")

                crit_data = []
                for c in problem.criteria:
                    curr_val = float(c.baseline_score)
                    if st.session_state.get("custom_inputs") and "scores" in st.session_state["custom_inputs"]:
                        curr_val = float(st.session_state["custom_inputs"]["scores"].get(c.id, curr_val))
                    crit_data.append({
                        "crit_id": c.id,
                        "Criterion name": f"C{c.id}: {c.code} — {c.name}",
                        "Maximum marks": float(c.max_score),
                        "Minimum required": float(c.min_threshold),
                        "Your current score": curr_val,
                    })
                df_crit_input = pd.DataFrame(crit_data)

                edited_df = st.data_editor(
                    df_crit_input,
                    column_config={
                        "crit_id": None,
                        "Criterion name": st.column_config.TextColumn("Criterion name", disabled=True),
                        "Maximum marks": st.column_config.NumberColumn("Maximum marks", disabled=True, format="%.0f"),
                        "Minimum required": st.column_config.NumberColumn("Minimum required", disabled=True, format="%.0f"),
                        "Your current score": st.column_config.NumberColumn(
                            "Your current score",
                            min_value=0.0,
                            max_value=150.0,
                            step=1.0,
                            format="%.1f",
                            required=True,
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="custom_scores_editor",
                )

                st.markdown("###### 2. Department planning constraints")
                c_inp1, c_inp2 = st.columns(2)
                c_inp3, c_inp4 = st.columns(2)

                default_b = float(st.session_state.get("custom_inputs", {}).get("budget", problem.budget))
                default_k = float(st.session_state.get("custom_inputs", {}).get("capacity", problem.monthly_capacity))
                default_t = int(st.session_state.get("custom_inputs", {}).get("horizon", problem.planning_horizon))
                default_r = float(st.session_state.get("custom_inputs", {}).get("target_score", problem.min_accreditation_score))

                with c_inp1:
                    custom_b = st.number_input(
                        "Budget (₹ Lakh)",
                        value=default_b,
                        min_value=0.1,
                        step=1.0,
                        help="Maximum institutional capital and operational budget allocated for improvements.",
                    )
                with c_inp2:
                    custom_k = st.number_input(
                        "Faculty hours available per month",
                        value=default_k,
                        min_value=1.0,
                        step=10.0,
                        help="Total faculty hours available across department staff members per month.",
                    )
                with c_inp3:
                    custom_t = st.number_input(
                        "Planning horizon (months)",
                        value=default_t,
                        min_value=1,
                        max_value=60,
                        step=1,
                        help="Fixed time frame (in months) within which all scheduled actions must finish.",
                    )
                with c_inp4:
                    custom_r = st.number_input(
                        "Target score (marks)",
                        value=default_r,
                        min_value=0.0,
                        max_value=float(np.sum(problem.M)),
                        step=10.0,
                        help="Total marks out of 1000 required to achieve accreditation.",
                    )

                submitted = st.form_submit_button("Apply my numbers", type="primary", use_container_width=True)

            if submitted:
                val_errors = []
                score_dict = {}
                for _, r in edited_df.iterrows():
                    cid = int(r["crit_id"])
                    sc = float(r["Your current score"])
                    max_m = float(r["Maximum marks"])
                    if sc < 0.0 or sc > max_m:
                        val_errors.append(f"Score for Criterion {cid} ({sc:.1f}) must be between 0 and {max_m:.0f}.")
                    score_dict[cid] = sc

                if custom_b <= 0:
                    val_errors.append("Budget must be greater than 0.")
                if custom_k <= 0:
                    val_errors.append("Faculty hours available per month must be greater than 0.")
                if custom_t < 1:
                    val_errors.append("Planning horizon must be at least 1 month.")
                total_max = float(np.sum(problem.M))
                if custom_r < 0 or custom_r > total_max:
                    val_errors.append(f"Target score must be between 0 and {total_max:.0f}.")

                if val_errors:
                    for err in val_errors:
                        st.error(err)
                else:
                    new_custom_data = {
                        "budget": float(custom_b),
                        "capacity": float(custom_k),
                        "horizon": int(custom_t),
                        "target_score": float(custom_r),
                        "scores": score_dict,
                    }
                    st.session_state["custom_inputs"] = new_custom_data
                    _clear_dataset_state()
                    st.toast("Your department numbers have been applied!")
                    st.rerun()

            if is_custom_active:
                if st.button("Reset to dataset defaults", use_container_width=True):
                    cfg_to_clean = json.dumps({"base": current_config, "data": st.session_state["custom_inputs"]}, sort_keys=True)
                    _cleanup_custom_files(hashlib.sha256(cfg_to_clean.encode("utf-8")).hexdigest()[:12])
                    st.session_state["custom_inputs"] = None
                    st.session_state["use_custom_inputs"] = False
                    _clear_dataset_state()
                    st.rerun()

    if is_custom_active:
        st.markdown('<span style="background-color: #DBEAFE; color: #1E40AF; padding: 0.2rem 0.5rem; border-radius: 0.375rem; font-size: 0.8rem; font-weight: 700;">USING YOUR NUMBERS</span>', unsafe_allow_html=True)

    _max_total = int(np.sum(problem.M))
    _target = int(problem.min_accreditation_score)
    _deficit = max(0.0, problem.min_accreditation_score - baseline_score)
    if baseline_score < problem.min_accreditation_score:
        st.info(
            f"Your department currently scores {baseline_score:.0f} out of {_max_total}. "
            f"The accreditation target is {_target}, so you need {_deficit:.0f} more marks."
        )
    else:
        st.success(
            f"Your department currently scores {baseline_score:.0f} out of {_max_total}. "
            f"This meets the accreditation target of {_target}."
        )

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    deficit = max(0.0, problem.min_accreditation_score - baseline_score)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline score</div>
            <div class="metric-value">{baseline_score:.0f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta delta-bad">Current: {(baseline_score/10):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accreditation target</div>
            <div class="metric-value">{problem.min_accreditation_score:.0f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta delta-good">Target: 75.0%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Marks still needed</div>
            <div class="metric-value">+{deficit:.0f} <span style="font-size: 0.9rem; color: #6B7280;">Marks</span></div>
            <div class="metric-delta delta-bad">Gap to target</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Budget</div>
            <div class="metric-value">₹{problem.budget:.1f} <span style="font-size: 0.9rem; color: #6B7280;">Lakh</span></div>
            <div class="metric-delta delta-good">Total spend allowed</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        viol_floors = sum(1 for c in problem.criteria if c.baseline_score < c.min_threshold)
        floor_color = "delta-bad" if viol_floors > 0 else "delta-good"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Subject areas below minimum</div>
            <div class="metric-value">{viol_floors} <span style="font-size: 0.9rem; color: #6B7280;">Criteria</span></div>
            <div class="metric-delta {floor_color}">{'Attention needed' if viol_floors > 0 else 'All minimums met'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Criteria Progress Table & Chart
    st.subheader("Where you stand by subject area")
    st.caption("Rows marked below minimum need attention first.")

    # Check if any criterion has zero available actions (Part F3)
    covered_crit_ids = set(a.crit_id for a in problem.actions)
    zero_action_crits = [c for c in problem.criteria if c.id not in covered_crit_ids]
    if zero_action_crits:
        crit_names_str = ", ".join(f"{c.name} (Criterion {c.id})" for c in zero_action_crits)
        st.info(
            f"**Action limitation:** No actions are available in this dataset for **{crit_names_str}**. "
            f"Its score cannot improve from baseline, which limits how balanced any generated plan can be."
        )
    crit_rows = []
    for c in problem.criteria:
        gap_to_floor = max(0.0, c.min_threshold - c.baseline_score)
        gap_to_max = c.max_score - c.baseline_score
        status = "Below minimum" if c.baseline_score < c.min_threshold else "Meets minimum"
        crit_rows.append({
            "Criterion": f"C{c.id}: {c.code}",
            "Focus Area": c.name,
            "Maximum Marks": c.max_score,
            "Your Current Score": c.baseline_score,
            "Minimum Required": c.min_threshold,
            "Minimum %": f"{(c.min_threshold / c.max_score * 100):.1f}%",
            "Current %": f"{(c.baseline_score / c.max_score * 100):.1f}%",
            "Gap to Max": gap_to_max,
            "Status": status,
        })
    df_crit = pd.DataFrame(crit_rows)

    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        st.markdown("##### 10 Criteria Performance & Compliance Matrix")
        try:
            st.dataframe(
                df_crit.style.apply(
                    lambda row: ["background-color: #FEE2E2" if "Below minimum" in str(row["Status"]) else "" for _ in row],
                    axis=1
                ),
                use_container_width=True,
                height=380,
            )
        except Exception:
            st.dataframe(df_crit, use_container_width=True, height=380)

    with c_right:
        st.markdown("##### Baseline vs. Minimum Required vs. Maximum Marks")
        fig_crit = go.Figure()
        fig_crit.add_trace(go.Bar(
            x=[f"C{c.id}" for c in problem.criteria],
            y=problem.s0,
            name="Baseline Score",
            marker_color="#3B82F6",
        ))
        fig_crit.add_trace(go.Scatter(
            x=[f"C{c.id}" for c in problem.criteria],
            y=problem.L,
            name="Minimum required",
            mode="lines+markers",
            line=dict(color="#EF4444", width=2, dash="dash"),
            marker=dict(size=8, color="#EF4444"),
        ))
        fig_crit.add_trace(go.Scatter(
            x=[f"C{c.id}" for c in problem.criteria],
            y=problem.M,
            name="Criterion maximum",
            mode="lines+markers",
            line=dict(color="#10B981", width=2, dash="dot"),
            marker=dict(size=8, color="#10B981"),
        ))
        fig_crit.update_layout(
            barmode="group",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="NBA Criterion",
            yaxis_title="Marks",
        )
        st.plotly_chart(fig_crit, use_container_width=True)

    # Sub-criteria details explorer
    with st.expander("See the detailed breakdown"):
        sub_rel_file = problem.config.get("data", {}).get("subcriteria_file", "data/nba_subcriteria.csv")
        sub_path = problem.root_dir / sub_rel_file
        if sub_path.exists():
            df_sub = pd.read_csv(sub_path)
            st.dataframe(df_sub, use_container_width=True)
        else:
            st.info(f"Sub-criteria traceability file '{sub_rel_file}' not found.")

    # Stacking Action Catalog in an expander on Step 1
    with st.expander("Browse every possible improvement action", expanded=False):
        st.markdown(
            "Some improvements (**Evidence Packages**) must be completed before others can start. "
            "Other improvements can be repeated up to a maximum number of times. "
            "Use the filters below to find actions."
        )

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.session_state.setdefault("catalog_sel_crit", [])
            sel_crit = st.multiselect("Filter by Criterion", [f"C{i}" for i in range(1, 11)], key="catalog_sel_crit")
        with f_col2:
            st.session_state.setdefault("catalog_sel_type", "All")
            sel_type = st.radio("Action Type", ["All", "Evidence Packages Only", "Interventions Only"], horizontal=True, key="catalog_sel_type")
        with f_col3:
            st.session_state.setdefault("catalog_search_kw", "")
            search_kw = st.text_input("Search Action Name or ID", key="catalog_search_kw")

        action_table_data = []
        for a in problem.actions:
            crit_label = f"C{a.crit_id}"
            if sel_crit and crit_label not in sel_crit:
                continue
            if sel_type == "Evidence Packages Only" and not a.is_evidence:
                continue
            if sel_type == "Interventions Only" and a.is_evidence:
                continue
            if search_kw and (search_kw.lower() not in a.name.lower() and search_kw.lower() not in a.action_id.lower()):
                continue

            prereq_str = ", ".join(a.prerequisites) if a.prerequisites else "None (Independent)"
            action_table_data.append({
                "Action ID": a.action_id,
                "Criterion": crit_label,
                "Action Name": a.name,
                "Type": "Evidence Package" if a.is_evidence else "Intervention",
                "How Many Times": a.max_units,
                "Points per Completion": a.score_gain,
                "Max Gain": a.score_gain * a.max_units,
                "Cost (Lakh)": a.cost_lakh,
                "Faculty Hours": a.faculty_hours,
                "Duration (months)": a.duration_months,
                "Can Start In": f"Month {a.earliest_start} → {a.earliest_finish}",
                "Prerequisites": prereq_str,
                "Fits in 12-Month Plan": "Yes" if a.is_schedulable else "Exceeds horizon",
            })
        df_actions = pd.DataFrame(action_table_data)

        st.markdown(f"**Displaying {len(df_actions)} of {problem.num_actions} Actions**")
        if not df_actions.empty:
            st.dataframe(df_actions, use_container_width=True, height=350)
        else:
            st.info("No actions match the active filters.")

        # Earliest-Start Schedule Timeline
        st.markdown("##### Earliest-Start Schedule Timeline")
        gantt_df = []
        for a in problem.actions:
            if sel_crit and f"C{a.crit_id}" not in sel_crit:
                continue
            gantt_df.append({
                "Action": f"{a.action_id}: {a.name[:28]}...",
                "Criterion": f"C{a.crit_id}",
                "Start": a.earliest_start,
                "Finish": a.earliest_finish,
                "Duration": max(1, a.duration_months),
                "Cost": a.cost_lakh,
                "Hours": a.faculty_hours,
            })
        df_gantt = pd.DataFrame(gantt_df)

        if not df_gantt.empty:
            fig_gantt = px.bar(
                df_gantt,
                x="Duration",
                y="Action",
                base="Start",
                color="Criterion",
                orientation="h",
                hover_data=["Start", "Finish", "Duration", "Cost", "Hours"],
                title="Earliest-Start Execution Timeline",
                labels={"Duration": "Timeline (Months 0 to 12)"},
            )
            fig_gantt.add_vline(x=problem.planning_horizon, line_dash="dash", line_color="red", annotation_text=f"Horizon T={problem.planning_horizon}")
            fig_gantt.update_layout(
                height=max(360, len(df_gantt) * 18),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gantt, use_container_width=True)
            st.caption(
                "Each bar shows when an action can begin and how long it runs. "
                "The red line marks the end of your planning window."
            )

        # Action Prerequisite Dependency Inspector
        st.markdown("##### Action Prerequisite Dependency Inspector")
        action_names = [f"{a.action_id} — {a.name}" for a in problem.actions]
        st.session_state.setdefault("catalog_sel_inspect", 0)
        sel_inspect = st.selectbox("Select an Action to Inspect Dependencies", range(len(action_names)), format_func=lambda i: action_names[i], key="catalog_sel_inspect")
        target_a = problem.actions[sel_inspect]

        dep_col1, dep_col2 = st.columns(2)
        with dep_col1:
            st.markdown(f"**Action:** `{target_a.action_id}` — {target_a.name}")
            st.markdown(f"**Criterion:** C{target_a.crit_id} | **Duration:** {target_a.duration_months} mo")
            immediate_prereqs = target_a.prerequisites
            st.markdown(f"**Immediate Prerequisites:** {', '.join(immediate_prereqs) if immediate_prereqs else 'None'}")
            transitive_prereqs = [problem.actions[i].action_id for i in np.flatnonzero(problem.dep[:, target_a.index])]
            st.markdown(f"**Must be done first (all ancestors):** {', '.join(transitive_prereqs) if transitive_prereqs else 'None'}")
        with dep_col2:
            dependents = [problem.actions[i].action_id for i in np.flatnonzero(problem.dep[target_a.index, :])]
            st.markdown(f"**This unlocks these actions:** {', '.join(dependents) if dependents else 'None'}")

    # Navigation Footer for Step 1
    st.markdown("---")
    st.button("Continue to Generate plans ➔", type="primary", use_container_width=True, on_click=set_step, args=("generate",))


# ==============================================================================
# STAGE 2: GENERATE PLANS
# ==============================================================================
elif step == "generate":
    st.markdown('<div class="step-label">STEP 2 OF 4</div>', unsafe_allow_html=True)
    st.subheader("Generate plans")
    if is_custom_active:
        st.markdown('<span style="background-color: #DBEAFE; color: #1E40AF; padding: 0.15rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 700;">USING YOUR NUMBERS</span>', unsafe_allow_html=True)
    st.markdown(
        "The planner searches thousands of combinations of actions and keeps the "
        "best trade-offs between marks, cost, and faculty hours. Each result is a complete plan."
    )

    # Preset callbacks (defined before sliders render)
    def _set_quick():
        st.session_state["opt_pop_size"] = 20
        st.session_state["opt_max_gens"] = 10

    def _set_balanced():
        st.session_state["opt_pop_size"] = 40
        st.session_state["opt_max_gens"] = 30

    def _set_thorough():
        st.session_state["opt_pop_size"] = 100
        st.session_state["opt_max_gens"] = 100

    st.markdown("##### How thorough should the search be?")
    pr_col1, pr_col2, pr_col3 = st.columns(3)
    with pr_col1:
        st.button("⚡ Quick", on_click=_set_quick, use_container_width=True)
        st.caption("Fastest, fewer options")
    with pr_col2:
        st.button("⚖️ Balanced", on_click=_set_balanced, use_container_width=True)
        st.caption("Recommended")
    with pr_col3:
        st.button("🔍 Thorough", on_click=_set_thorough, use_container_width=True)
        st.caption("Slowest, most options")

    curr_pop = st.session_state.get("opt_pop_size", 40)
    curr_gens = st.session_state.get("opt_max_gens", 30)
    if (curr_pop, curr_gens) == (20, 10):
        preset_label = "Quick"
    elif (curr_pop, curr_gens) == (40, 30):
        preset_label = "Balanced"
    elif (curr_pop, curr_gens) == (100, 100):
        preset_label = "Thorough"
    else:
        preset_label = "Custom"
    st.caption(f"Current search depth: **{preset_label}**")

    # Algorithm Multi-Select (Part E1)
    _ALG_OPTIONS = [
        "Hybrid-Memetic (Recommended)",
        "NSGA-II (Deb et al., 2002)",
        "SPEA2 (Zitzler et al., 2001)",
        "MOEA/D (Zhang & Li, 2007)",
        "HiGHS MILP (Exact Mathematical Programming)",
        "Greedy-Baseline (Efficiency Heuristic)",
        "Random-Baseline (Monte Carlo)",
    ]

    _ALG_LABELS = {
        "Hybrid-Memetic (Recommended)":              "Recommended combined search (Hybrid-Memetic)",
        "NSGA-II (Deb et al., 2002)":                "Evolutionary multi-objective search (NSGA-II)",
        "SPEA2 (Zitzler et al., 2001)":              "Archived evolutionary search (SPEA2)",
        "MOEA/D (Zhang & Li, 2007)":                 "Decomposition-based search (MOEA/D)",
        "HiGHS MILP (Exact Mathematical Programming)": "Exact mathematical solver (HiGHS MILP)",
        "Greedy-Baseline (Efficiency Heuristic)":    "Simple greedy baseline (Greedy)",
        "Random-Baseline (Monte Carlo)":             "Random sampling baseline (Random)",
    }

    if "opt_chosen_algs" not in st.session_state:
        st.session_state["opt_chosen_algs"] = ["Hybrid-Memetic (Recommended)"]

    def _select_all_algs() -> None:
        st.session_state["opt_chosen_algs"] = list(_ALG_OPTIONS)

    def _clear_algs() -> None:
        st.session_state["opt_chosen_algs"] = []

    st.markdown("##### Which methods should the planner use?")
    ms_c1, ms_c2, _ = st.columns([1, 1, 4])
    with ms_c1:
        st.button("Select all", on_click=_select_all_algs, use_container_width=True)
    with ms_c2:
        st.button("Clear", on_click=_clear_algs, use_container_width=True)

    chosen_algs = st.multiselect(
        "Which methods should the planner use?",
        _ALG_OPTIONS,
        key="opt_chosen_algs",
        format_func=lambda v: _ALG_LABELS.get(v, v),
        label_visibility="collapsed",
    )
    st.caption("Pick one or more. Using several gives you more variety to compare, but takes longer.")

    has_selected_methods = bool(chosen_algs)
    if not has_selected_methods:
        st.caption("Pick at least one method.")

    run_btn = st.button("Generate plans", type="primary", use_container_width=True, disabled=not has_selected_methods)

    with st.expander("Fine-tune settings (optional)", expanded=False):
        st.session_state.setdefault("opt_pop_size", 40)
        pop_size = st.slider(
            "Population Size",
            min_value=10, max_value=200, step=10,
            key="opt_pop_size",
            help="Number of candidate plans kept at once. More plans = broader search, longer run.",
        )
        st.session_state.setdefault("opt_max_gens", 30)
        max_gens = st.slider(
            "Max Generations",
            min_value=10, max_value=200, step=10,
            key="opt_max_gens",
            help="How many rounds of improvement the algorithm runs. More rounds = more refined plans, longer run.",
        )
        st.caption("These settings only affect the evolutionary search methods.")

        has_evo = any(any(k in alg for k in ["NSGA", "SPEA", "MOEA", "Hybrid"]) for alg in chosen_algs)
        crossover_p = 0.9
        mutation_p = 0.15
        if has_evo:
            st.session_state.setdefault("opt_crossover_p", 0.9)
            crossover_p = st.slider(
                "Crossover Probability",
                min_value=0.5, max_value=1.0, step=0.05,
                key="opt_crossover_p",
                help="How often two plans are combined to produce a child plan. Higher values mix plans more aggressively.",
            )
            st.session_state.setdefault("opt_mutation_p", 0.15)
            mutation_p = st.slider(
                "Mutation Probability",
                min_value=0.01, max_value=0.5, step=0.02,
                key="opt_mutation_p",
                help="How often a random change is applied to a plan. Higher values introduce more variety but may disrupt good plans.",
            )

        st.session_state.setdefault("opt_seed", 42)
        seed = st.number_input(
            "Random Seed",
            step=1,
            key="opt_seed",
            help="Change this number to get a different set of plans from the same algorithm. Usually leave it as it is.",
        )

        with st.expander("About these algorithms"):
            st.markdown(
                "- **Recommended combined search (Hybrid-Memetic):** Combines evolutionary search with "
                "a local repair step to refine feasibility. (Hybrid Memetic NSGA-II variant)\n"
                "- **Evolutionary multi-objective search (NSGA-II):** Uses non-dominated sorting and "
                "crowding distance to maintain a diverse set of plans. "
                "(Deb et al., 2002, *IEEE TEVC*)\n"
                "- **Archived evolutionary search (SPEA2):** Maintains a fixed-size archive of "
                "non-dominated plans alongside the main population. "
                "(Zitzler et al., 2001, *TIK-Report 103*)\n"
                "- **Decomposition-based search (MOEA/D):** Decomposes the problem into scalar "
                "sub-problems solved cooperatively using neighbourhood weights. "
                "(Zhang & Li, 2007, *IEEE TEVC*)\n"
                "- **Exact mathematical solver (HiGHS MILP):** Solves a mixed-integer linear "
                "program exactly via epsilon-constraint scalarisation; does not use the population "
                "or generation settings.\n"
                "- **Simple greedy baseline (Greedy):** Picks actions one at a time in order of "
                "score-per-cost efficiency until constraints are met.\n"
                "- **Random sampling baseline (Random):** Samples 300 random feasible plans as a "
                "comparison baseline."
            )

    if run_btn and has_selected_methods:
        results_list = []
        failures = []
        cfg_key = active_problem_cfg

        with st.status("Generating plans...", expanded=True) as status_box:
            for idx, alg in enumerate(chosen_algs):
                friendly_name = _ALG_LABELS.get(alg, alg)
                status_box.update(label=f"Running {idx + 1} of {len(chosen_algs)}: {friendly_name}...")
                try:
                    res = _run_algorithm(
                        alg, problem, pop_size, max_gens, crossover_p, mutation_p, int(seed)
                    )
                    canon_key = _canon_alg(alg)
                    if cfg_key not in st.session_state["results_cache"]:
                        st.session_state["results_cache"][cfg_key] = {}
                    st.session_state["results_cache"][cfg_key][canon_key] = res
                    results_list.append((canon_key, res))
                except Exception as e:
                    failures.append((friendly_name, str(e)))

            status_box.update(label="Finished generating plans!", state="complete", expanded=False)

        _invalidate_plan_selection()

        if failures:
            for f_alg, f_msg in failures:
                st.warning(f"Could not generate plans with {f_alg}.")
                with st.expander(f"Technical details for {f_alg}"):
                    st.code(f_msg)

        if results_list:
            best_res = max(results_list, key=lambda pair: len(pair[1].pareto_solutions))[1]
            st.session_state["latest_res"] = best_res

            total_new_plans = sum(len(r.pareto_solutions) for _, r in results_list)
            if total_new_plans > 0:
                method_names = ", ".join(k for k, _ in results_list)
                st.session_state["_flash"] = f"Generated {total_new_plans} plans using {method_names}. Choose one below."
                st.session_state["step"] = "compare"
                st.session_state["_scroll_top"] = True
                st.rerun()
            else:
                st.warning(
                    "No feasible plans were found with these settings. "
                    "Try a larger Population Size or more Generations, or select different methods."
                )

    # Results area (for returning users)
    current_p_dict = get_pareto_solutions(problem)
    total_plans_count = sum(len(sols) for sols in current_p_dict.values())

    if total_plans_count == 0:
        st.info("No plans yet. Pick one or more methods above and click Generate plans.")
    else:
        algs_str = ", ".join(current_p_dict.keys())
        st.success(f"**{total_plans_count} plans ready** from {algs_str}. Continue to compare them.")

        active_cache = st.session_state.get("results_cache", {}).get(active_problem_cfg, {})

        with st.expander("How good is each method's set of plans?", expanded=True):
            method_rows = []
            for alg_name, r in active_cache.items():
                sols = getattr(r, "pareto_solutions", [])
                if sols:
                    obj_matrix_min = np.array([s.objectives_min for s in sols], dtype=np.float64)
                    obj_matrix_nat = np.array([s.objectives for s in sols], dtype=np.float64)
                    hv = compute_hypervolume(obj_matrix_min, normalize=True) if len(sols) > 0 else 0.0
                    sp = compute_spacing(obj_matrix_nat) if len(sols) > 0 else 0.0
                    spr = compute_spread(obj_matrix_nat) if len(sols) > 0 else 0.0
                    method_rows.append({
                        "Method": alg_name,
                        "Plans found": len(sols),
                        "Plan set quality": f"{hv:.4f}",
                        "Plan spacing": f"{sp:.2f}",
                        "Spread evenness": f"{spr:.3f}",
                        "Run time": f"{r.elapsed_time:.2f}s",
                    })
            if method_rows:
                df_methods = pd.DataFrame(method_rows)
                st.dataframe(
                    df_methods,
                    use_container_width=True,
                    column_config={
                        "Plan set quality": st.column_config.TextColumn(
                            "Plan set quality",
                            help="Hypervolume: quality of the plan space covered. Higher is better."
                        ),
                        "Plan spacing": st.column_config.TextColumn(
                            "Plan spacing",
                            help="Standard deviation of distance between plans. Lower is more evenly spaced."
                        ),
                        "Spread evenness": st.column_config.TextColumn(
                            "Spread evenness",
                            help="Spread across extreme trade-offs. Lower is better (0.0 is ideal)."
                        ),
                    }
                )
            else:
                st.caption("No specific algorithm metrics to display.")

        with st.expander("How the search improved over time", expanded=False):
            history_algs = {
                alg_name: r for alg_name, r in active_cache.items()
                if getattr(r, "history", None) and len(r.history) > 0
            }
            if history_algs:
                selected_hist_alg = st.selectbox(
                    "Show how the search improved for",
                    list(history_algs.keys()),
                    index=0,
                )
                r_hist = history_algs[selected_hist_alg]
                hist_df = pd.DataFrame(r_hist.history)
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=hist_df["generation"],
                    y=hist_df["f1_max"],
                    name="Best Total Score so far",
                    line=dict(color="#2563EB", width=2),
                ))
                fig_hist.add_hline(
                    y=problem.min_accreditation_score,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Target R={problem.min_accreditation_score:.0f}",
                )
                fig_hist.update_layout(
                    title=f"Convergence History ({selected_hist_alg}): Best Score over Generations",
                    xaxis_title="Generation",
                    yaxis_title="Total Marks",
                    margin=dict(l=20, r=20, t=35, b=20),
                    height=280,
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.caption("Convergence history is recorded when running evolutionary search algorithms (Hybrid-Memetic, NSGA-II, SPEA2, MOEA/D).")

    with st.expander("Advanced: how do the algorithms compare?", expanded=False):
        st.markdown("**How do the algorithms compare?**")
        if is_custom_active:
            st.caption("Benchmark results were computed on the sample datasets, not your numbers.")
        summary_path = Path("results/benchmark_metrics_summary.csv")
        raw_path = Path("results/benchmark_metrics_raw.csv")

        if summary_path.exists() and summary_path.stat().st_size > 10:
            try:
                df_summary = pd.read_csv(summary_path)
                _BENCH_DISPLAY_COLS = {
                    "algorithm":                    "Algorithm",
                    "hypervolume_mean":             "Plan Set Quality (mean)",
                    "hypervolume_std":              "Plan Set Quality (std)",
                    "spacing_mean":                 "Plan Spacing (mean)",
                    "spacing_std":                  "Plan Spacing (std)",
                    "spread_mean":                  "Spread Evenness (mean)",
                    "spread_std":                   "Spread Evenness (std)",
                    "igd_mean":                     "IGD (mean)",
                    "igd_std":                      "IGD (std)",
                    "coverage_vs_milp_mean":        "Coverage vs. Exact Solver (mean)",
                    "time_sec_mean":                "Run Time, s (mean)",
                    "pareto_count_mean":            "Plans Found (mean)",
                    "feasible_front_count_mean":    "Feasible Plans (mean)",
                    "feasibility_rate_pct_mean":    "Feasibility Rate % (mean)",
                    "reference_hypervolume_mean":   "Reference Plan Set Quality",
                    "hypervolume_pct_reference_mean": "Quality vs. Reference %",
                    "max_f1_mean":                  "Best Score Found (mean)",
                    "min_f2_mean":                  "Lowest Cost Found (mean)",
                }
                df_display = df_summary.rename(
                    columns={k: v for k, v in _BENCH_DISPLAY_COLS.items() if k in df_summary.columns}
                )
                st.dataframe(df_display, use_container_width=True)
                st.caption(
                    "**Plan Set Quality** (Hypervolume) measures how much of the good-plan space the "
                    "algorithm covers — higher is better. "
                    "**Plan Spacing** measures how evenly the returned plans are spaced — lower is better. "
                    "**Spread Evenness** measures how evenly the plans are distributed across the full "
                    "range of trade-offs, including whether they reach the extreme ends — lower is better."
                )

                if raw_path.exists() and raw_path.stat().st_size > 10:
                    df_raw = pd.read_csv(raw_path)
                    fig_bar = px.bar(
                        df_raw,
                        x="algorithm",
                        y="hypervolume",
                        color="algorithm",
                        title="Plan Set Quality (Hypervolume) — Higher is Better",
                        labels={"hypervolume": "Plan Set Quality (Hypervolume)", "algorithm": "Algorithm"},
                    )
                    fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
                    st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.caption(f"Benchmark summary loading note: {e}")
        else:
            st.info("Benchmark summary CSVs not yet present in `results/`.")

    # Navigation Footer for Step 2
    st.markdown("---")
    f_left, f_right = st.columns([1, 1])
    with f_left:
        st.button("⬅ Back to Your department", use_container_width=True, on_click=set_step, args=("status",))
    with f_right:
        has_plans = total_plans_count > 0
        if not has_plans:
            st.caption("Generate or load plans first.")
        st.button(
            "Continue to Compare and choose ➔",
            type="primary",
            use_container_width=True,
            disabled=not has_plans,
            on_click=set_step,
            args=("compare",),
        )


# ==============================================================================
# STAGE 3: COMPARE & CHOOSE
# ==============================================================================
elif step == "compare":
    st.markdown('<div class="step-label">STEP 3 OF 4</div>', unsafe_allow_html=True)
    st.subheader("Compare and choose")
    if is_custom_active:
        st.markdown('<span style="background-color: #DBEAFE; color: #1E40AF; padding: 0.15rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 700;">USING YOUR NUMBERS</span>', unsafe_allow_html=True)
    if st.session_state.get("_flash"):
        st.success(st.session_state["_flash"])
        del st.session_state["_flash"]

    st.markdown(
        "Each dot is one complete plan. No plan is best at everything: plans that score higher usually "
        "cost more or need more faculty hours. Explore the trade-offs below, then pick the plan that "
        "best fits your department's priorities."
    )

    all_sols_flat_cmp = get_all_sols_flat(problem)
    pareto_solutions_to_plot = get_pareto_solutions(problem)

    if not all_sols_flat_cmp:
        st.info("No plans available yet. Go back to Generate plans to create plans first.")
        st.button("⬅ Back to Generate plans", use_container_width=True, on_click=set_step, args=("generate",))
        st.stop()
    else:
        alg_names_str = ", ".join(sorted(set(alg for alg, _ in all_sols_flat_cmp)))
        st.caption(f"{len(all_sols_flat_cmp)} plans from {alg_names_str}.")

        # Quick-pick archetype buttons
        st.markdown("##### Quick picks")
        st.caption("Choose a plan based on what matters most to you, or scroll down to explore all plans.")
        qp_col1, qp_col2, qp_col3 = st.columns(3)

        pick_score = _pick_archetype(all_sols_flat_cmp, "score", problem.min_accreditation_score)
        pick_cost  = _pick_archetype(all_sols_flat_cmp, "cost",  problem.min_accreditation_score)
        pick_bal   = _pick_archetype(all_sols_flat_cmp, "balance", problem.min_accreditation_score)

        with qp_col1:
            if st.button("Highest score", use_container_width=True, disabled=pick_score is None):
                _select_plan(pick_score[0], list(pick_score[1].x))
                st.rerun()
            if pick_score:
                st.caption(
                    f"Score {pick_score[1].f1_total_score:.0f} · "
                    f"₹{pick_score[1].f2_cost_lakh:.1f}L · "
                    f"{pick_score[1].f3_faculty_hours:.0f} Faculty Hours"
                )
        with qp_col2:
            if st.button("Lowest cost", use_container_width=True, disabled=pick_cost is None):
                _select_plan(pick_cost[0], list(pick_cost[1].x))
                st.rerun()
            if pick_cost:
                st.caption(
                    f"Score {pick_cost[1].f1_total_score:.0f} · "
                    f"₹{pick_cost[1].f2_cost_lakh:.1f}L · "
                    f"{pick_cost[1].f3_faculty_hours:.0f} Faculty Hours"
                )
        with qp_col3:
            if st.button("Most even progress", use_container_width=True, disabled=pick_bal is None):
                _select_plan(pick_bal[0], list(pick_bal[1].x))
                st.rerun()
            if pick_bal:
                st.caption(
                    f"Score {pick_bal[1].f1_total_score:.0f} · "
                    f"₹{pick_bal[1].f2_cost_lakh:.1f}L · "
                    f"{pick_bal[1].f3_faculty_hours:.0f} Faculty Hours"
                )

        # Build plotting dataframe
        plot_rows = []
        for alg_name, sols in pareto_solutions_to_plot.items():
            for s_idx, s in enumerate(sols):
                plot_rows.append({
                    "Algorithm": alg_name,
                    "Solution ID": f"{alg_name} #{s_idx+1}",
                    "Total Score": s.f1_total_score,
                    "Cost (Lakh)": s.f2_cost_lakh,
                    "Faculty Hours": s.f3_faculty_hours,
                    "Subject Balance": s.f4_worst_criterion_ratio,
                    "Active Actions": int(np.count_nonzero(s.x)),
                    "Feasible": s.is_feasible,
                    "eval_result": s,
                })
        df_plot = pd.DataFrame(plot_rows)

        # Resolve current selection for highlight
        sel_resolved = _resolve_selected_plan(problem)

        # 2D charts (primary view)
        st.markdown("##### Trade-off charts")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            fig_2d_1 = px.scatter(
                df_plot,
                x="Cost (Lakh)",
                y="Total Score",
                color="Algorithm",
                hover_name="Solution ID",
                hover_data=["Faculty Hours", "Subject Balance"],
                title="Total Score vs. Implementation Cost",
            )
            fig_2d_1.add_vline(x=problem.budget, line_dash="dash", line_color="red",
                                annotation_text=f"Budget limit ₹{problem.budget}L")
            fig_2d_1.add_hline(y=problem.min_accreditation_score, line_dash="dash", line_color="green",
                                annotation_text=f"Target score {problem.min_accreditation_score:.0f}")
            if sel_resolved:
                _, sel_ev = sel_resolved
                fig_2d_1.add_trace(go.Scatter(
                    x=[sel_ev.f2_cost_lakh], y=[sel_ev.f1_total_score],
                    mode="markers",
                    marker=dict(size=16, color="gold", line=dict(color="black", width=2)),
                    name="Your choice",
                    showlegend=True,
                ))
            fig_2d_1.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_1, use_container_width=True)
            st.caption("Higher score is better (top). Lower cost is better (left of red budget limit).")

            fig_2d_3 = px.scatter(
                df_plot,
                x="Cost (Lakh)",
                y="Subject Balance",
                color="Algorithm",
                hover_name="Solution ID",
                title="Subject Balance vs. Implementation Cost",
            )
            fig_2d_3.add_vline(x=problem.budget, line_dash="dash", line_color="red")
            if sel_resolved:
                _, sel_ev = sel_resolved
                fig_2d_3.add_trace(go.Scatter(
                    x=[sel_ev.f2_cost_lakh], y=[sel_ev.f4_worst_criterion_ratio],
                    mode="markers",
                    marker=dict(size=16, color="gold", line=dict(color="black", width=2)),
                    name="Your choice",
                    showlegend=True,
                ))
            fig_2d_3.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_3, use_container_width=True)
            st.caption("Higher subject balance is better (top). Lower cost is better (left of red budget limit).")

        with p_col2:
            fig_2d_2 = px.scatter(
                df_plot,
                x="Faculty Hours",
                y="Total Score",
                color="Algorithm",
                hover_name="Solution ID",
                title="Total Score vs. Faculty Hours",
            )
            fig_2d_2.add_hline(y=problem.min_accreditation_score, line_dash="dash", line_color="green")
            if sel_resolved:
                _, sel_ev = sel_resolved
                fig_2d_2.add_trace(go.Scatter(
                    x=[sel_ev.f3_faculty_hours], y=[sel_ev.f1_total_score],
                    mode="markers",
                    marker=dict(size=16, color="gold", line=dict(color="black", width=2)),
                    name="Your choice",
                    showlegend=True,
                ))
            fig_2d_2.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_2, use_container_width=True)
            st.caption("Higher score is better (top). Fewer faculty hours is better (left).")

            fig_2d_4 = px.scatter(
                df_plot,
                x="Faculty Hours",
                y="Subject Balance",
                color="Algorithm",
                hover_name="Solution ID",
                title="Subject Balance vs. Faculty Hours",
            )
            if sel_resolved:
                _, sel_ev = sel_resolved
                fig_2d_4.add_trace(go.Scatter(
                    x=[sel_ev.f3_faculty_hours], y=[sel_ev.f4_worst_criterion_ratio],
                    mode="markers",
                    marker=dict(size=16, color="gold", line=dict(color="black", width=2)),
                    name="Your choice",
                    showlegend=True,
                ))
            fig_2d_4.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=320)
            st.plotly_chart(fig_2d_4, use_container_width=True)
            st.caption("Higher subject balance is better (top). Fewer faculty hours is better (left).")

        # 3D view in expander
        with st.expander("3D view (optional)", expanded=False):
            fig_3d = px.scatter_3d(
                df_plot,
                x="Cost (Lakh)",
                y="Faculty Hours",
                z="Total Score",
                color="Subject Balance",
                symbol="Algorithm",
                hover_name="Solution ID",
                hover_data=["Total Score", "Cost (Lakh)", "Faculty Hours", "Subject Balance", "Active Actions"],
                color_continuous_scale="Viridis",
                title="3D View: Implementation Cost vs. Faculty Hours vs. Total Marks",
            )
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="Cost (Lakh INR)",
                    yaxis_title="Faculty Hours",
                    zaxis_title="Total Marks",
                ),
                margin=dict(l=10, r=10, t=35, b=10),
                height=500,
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        # Full table in expander
        with st.expander("See every plan in a table", expanded=False):
            st.dataframe(
                df_plot[["Solution ID", "Algorithm", "Total Score", "Cost (Lakh)", "Faculty Hours",
                          "Subject Balance", "Active Actions", "Feasible"]],
                use_container_width=True,
                height=300,
            )

        # Plan chooser
        st.markdown("---")
        with st.container(border=True):
            st.markdown("##### Choose a plan")
            st.caption("Select a plan below, or use the quick-pick buttons above.")

            sol_options_cmp = [
                f"Plan #{idx+1} · {alg} · Score {s.f1_total_score:.0f} · ₹{s.f2_cost_lakh:.1f}L · {s.f3_faculty_hours:.0f} Faculty Hours"
                for idx, (alg, s) in enumerate(all_sols_flat_cmp)
            ]

            # Guard stored index
            stored_idx = st.session_state.get("compare_plan_sel_idx", 0)
            if not isinstance(stored_idx, int) or stored_idx >= len(sol_options_cmp):
                stored_idx = 0
            st.session_state["compare_plan_sel_idx"] = stored_idx

            chosen_cmp_idx = st.selectbox(
                "Select a plan",
                range(len(sol_options_cmp)),
                format_func=lambda i: sol_options_cmp[i],
                key="compare_plan_sel_idx",
            )

            chosen_cmp_alg, chosen_cmp_ev = all_sols_flat_cmp[chosen_cmp_idx]

            # Summary card for chosen plan
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Score", f"{chosen_cmp_ev.f1_total_score:.0f} / {int(np.sum(problem.M))}")
            sc2.metric("Cost", f"₹{chosen_cmp_ev.f2_cost_lakh:.2f} L")
            sc3.metric("Faculty Hours", f"{chosen_cmp_ev.f3_faculty_hours:.0f} hrs")
            sc4.metric("Subject Balance", f"{chosen_cmp_ev.f4_worst_criterion_ratio:.1%}")

            feas_badge = "Feasible (meets budget, capacity, and prerequisites)" if chosen_cmp_ev.is_feasible else f"Infeasible (violates constraints by {chosen_cmp_ev.total_violation:.1f})"
            st.caption(f"**Status:** {feas_badge}")
            st.markdown(
                f"This plan delivers **{chosen_cmp_ev.f1_total_score:.0f} marks** "
                f"for **₹{chosen_cmp_ev.f2_cost_lakh:.2f} L** and "
                f"**{chosen_cmp_ev.f3_faculty_hours:.0f} faculty hours**."
            )

            if st.button("Select this plan", type="primary", use_container_width=True):
                _select_plan(chosen_cmp_alg, list(chosen_cmp_ev.x))
                st.rerun()

            if sel_resolved:
                _, sel_ev = sel_resolved
                st.success(
                    f"Your current choice: {sel_ev.f1_total_score:.0f} marks · "
                    f"₹{sel_ev.f2_cost_lakh:.2f} L · "
                    f"{sel_ev.f3_faculty_hours:.0f} faculty hours"
                )

    # Navigation Footer for Step 3
    st.markdown("---")
    f_left, f_right = st.columns([1, 1])
    with f_left:
        st.button("⬅ Back to Generate plans", use_container_width=True, on_click=set_step, args=("generate",))
    with f_right:
        has_selection = st.session_state.get("selected_plan_override") is not None
        if not has_selection:
            st.caption("Choose a plan above first.")
        st.button(
            "Continue to Your plan ➔",
            type="primary",
            use_container_width=True,
            disabled=not has_selection,
            on_click=_choose_and_continue,
            args=(
                st.session_state["selected_plan_override"]["alg"] if has_selection else "",
                st.session_state["selected_plan_override"]["x"] if has_selection else [],
            ),
        )


# ==============================================================================
# STAGE 4: YOUR ACTION PLAN & BLUEPRINT
# ==============================================================================
elif step == "plan":
    st.markdown('<div class="step-label">STEP 4 OF 4</div>', unsafe_allow_html=True)
    st.subheader("Your plan")
    if is_custom_active:
        st.markdown('<span style="background-color: #DBEAFE; color: #1E40AF; padding: 0.15rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 700;">USING YOUR NUMBERS</span>', unsafe_allow_html=True)
    st.markdown(
        "Here is the full detail of your chosen plan: how it improves every subject area, "
        "what it costs, and how much faculty time it needs each month."
    )

    all_sols_flat = get_all_sols_flat(problem)

    if not all_sols_flat:
        st.warning(
            "No plans are available yet. "
            "**1)** Return to the **'Generate Plans'** step. "
            "**2)** Pick an algorithm and click **Generate plans**. "
            "**3)** Come back here to inspect a plan."
        )
        st.button("⬅ Go to Generate Plans", on_click=set_step, args=("generate",))
        st.stop()

    # Resolve selected plan
    resolved = _resolve_selected_plan(problem)
    if resolved is None:
        st.warning("Please choose a plan in the Compare step first.")
        st.button("⬅ Back to Compare and choose", on_click=set_step, args=("compare",))
        st.stop()

    # Archetype selector — "Your chosen plan" first
    archetype_opts = [
        "Your chosen plan",
        "Highest score",
        "Lowest cost",
        "Most even progress",
        "Custom selection from list",
    ]

    stored_arch = st.session_state.get("plan_archetype_choice")
    if stored_arch not in archetype_opts:
        stored_arch = archetype_opts[0]
        st.session_state["plan_archetype_choice"] = stored_arch

    st.session_state.setdefault("plan_archetype_choice", archetype_opts[0])
    plan_archetype = st.radio(
        "Plan to show",
        archetype_opts,
        horizontal=True,
        key="plan_archetype_choice",
    )

    active_eval: Optional[EvaluationResult] = None
    plan_title = ""

    if plan_archetype == "Your chosen plan":
        active_eval = resolved[1]
        plan_title = f"Your Chosen Plan ({_canon_alg(resolved[0])})"
    elif "Highest score" in plan_archetype:
        pick = _pick_archetype(all_sols_flat, "score", problem.min_accreditation_score)
        active_eval = pick[1] if pick else resolved[1]
        plan_title = f"Highest Score Plan ({pick[0] if pick else ''})"
    elif "Lowest cost" in plan_archetype:
        pick = _pick_archetype(all_sols_flat, "cost", problem.min_accreditation_score)
        active_eval = pick[1] if pick else resolved[1]
        plan_title = f"Lowest Cost Plan ({pick[0] if pick else ''})"
    elif "Most even progress" in plan_archetype:
        pick = _pick_archetype(all_sols_flat, "balance", problem.min_accreditation_score)
        active_eval = pick[1] if pick else resolved[1]
        plan_title = f"Even Progress Plan ({pick[0] if pick else ''})"
    else:
        sol_options = [
            f"Plan #{idx+1} · {alg} · Score {s.f1_total_score:.0f} · ₹{s.f2_cost_lakh:.1f}L · {s.f3_faculty_hours:.0f} Faculty Hours"
            for idx, (alg, s) in enumerate(all_sols_flat)
        ]
        stored_plan_idx = st.session_state.get("plan_custom_solution_idx", 0)
        if not isinstance(stored_plan_idx, int) or stored_plan_idx >= len(sol_options):
            stored_plan_idx = 0
        st.session_state["plan_custom_solution_idx"] = stored_plan_idx

        chosen_idx = st.selectbox(
            "Select Solution",
            range(len(sol_options)),
            format_func=lambda i: sol_options[i],
            key="plan_custom_solution_idx",
        )
        active_eval = all_sols_flat[chosen_idx][1]
        plan_title = f"Custom Selected Plan ({all_sols_flat[chosen_idx][0]})"

    if active_eval is None:
        st.warning("Could not resolve the selected plan. Please choose another option.")
        st.stop()

    # Plain-English verdict line
    score_ok = active_eval.f1_total_score >= problem.min_accreditation_score - 1e-4
    cost_ok = active_eval.f2_cost_lakh <= problem.budget + 1e-4
    max_hours = float(np.max(active_eval.monthly_hours)) if hasattr(active_eval, "monthly_hours") else 0.0
    workload_ok = max_hours <= problem.monthly_capacity + 1e-4

    if score_ok and cost_ok and workload_ok:
        st.success(
            f"This plan raises your score to **{active_eval.f1_total_score:.0f}** (target {problem.min_accreditation_score:.0f}), "
            f"costs **₹{active_eval.f2_cost_lakh:.2f} Lakh** (within the ₹{problem.budget:.1f} Lakh budget), and "
            f"requires at most **{max_hours:.0f} faculty hours** in any month (capacity {problem.monthly_capacity:.0f} hrs)."
        )
    else:
        score_note = f"raises your score to **{active_eval.f1_total_score:.0f}** (target {problem.min_accreditation_score:.0f})" if score_ok else f"reaches score **{active_eval.f1_total_score:.0f}** (short of {problem.min_accreditation_score:.0f} target)"
        cost_note = f"costs **₹{active_eval.f2_cost_lakh:.2f} Lakh** (within budget)" if cost_ok else f"costs **₹{active_eval.f2_cost_lakh:.2f} Lakh** (exceeds ₹{problem.budget:.1f} Lakh budget)"
        load_note = f"peak workload is **{max_hours:.0f} faculty hours/month** (within capacity)" if workload_ok else f"peak workload is **{max_hours:.0f} faculty hours/month** (exceeds {problem.monthly_capacity:.0f} hrs capacity)"
        st.warning(f"This plan {score_note}, {cost_note}, and {load_note}.")

    # Plan Summary KPIs (At a glance)
    st.markdown("##### At a glance")
    st.caption("Key summary indicators for your selected plan.")
    pk1, pk2, pk3, pk4, pk5 = st.columns(5)

    accred_status = "5-6 Years (Full)" if active_eval.f1_total_score >= problem.min_accreditation_score else "Not Accredited"
    accred_color = "delta-good" if active_eval.f1_total_score >= problem.min_accreditation_score else "delta-bad"

    with pk1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total score</div>
            <div class="metric-value">{active_eval.f1_total_score:.1f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta {accred_color}">{accred_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with pk2:
        rem_budget = problem.budget - active_eval.f2_cost_lakh
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Implementation cost</div>
            <div class="metric-value">₹{active_eval.f2_cost_lakh:.2f} <span style="font-size: 0.9rem; color: #6B7280;">Lakh</span></div>
            <div class="metric-delta delta-good">Remaining: ₹{rem_budget:.2f}L</div>
        </div>
        """, unsafe_allow_html=True)

    with pk3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total faculty hours</div>
            <div class="metric-value">{active_eval.f3_faculty_hours:.0f} <span style="font-size: 0.9rem; color: #6B7280;">Hours</span></div>
            <div class="metric-delta delta-good">Avg: {(active_eval.f3_faculty_hours/problem.planning_horizon):.0f} hrs/mo</div>
        </div>
        """, unsafe_allow_html=True)

    with pk4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weakest subject coverage</div>
            <div class="metric-value">{(active_eval.f4_worst_criterion_ratio * 100):.1f}%</div>
            <div class="metric-delta delta-good">Higher = more balanced across all criteria</div>
        </div>
        """, unsafe_allow_html=True)

    with pk5:
        active_count = int(np.count_nonzero(active_eval.x))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Actions selected</div>
            <div class="metric-value">{active_count} <span style="font-size: 0.9rem; color: #6B7280;">/ {problem.num_actions}</span></div>
            <div class="metric-delta delta-good">{(active_count/problem.num_actions*100):.1f}% of catalog</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar Chart & Monthly Workload
    st.markdown("##### Where the improvement comes from")
    st.caption("Comparison of your baseline, the minimum required score, and your projected score across all 10 criteria.")
    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        st.markdown("###### 10-Criteria achievement radar profile")
        categories = [f"C{c.id}: {c.code}" for c in problem.criteria]
        categories.append(categories[0])  # Close polygon

        plan_scores = list(active_eval.criteria_scores)
        plan_scores.append(plan_scores[0])

        baseline_scores = list(problem.s0)
        baseline_scores.append(baseline_scores[0])

        floor_scores = list(problem.L)
        floor_scores.append(floor_scores[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=plan_scores,
            theta=categories,
            fill='toself',
            name="Selected Plan Score",
            line_color='#2563EB',
            fillcolor='rgba(37, 99, 235, 0.25)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_scores,
            theta=categories,
            name="Baseline score",
            line=dict(color='#9CA3AF', dash='dash'),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=floor_scores,
            theta=categories,
            name="Minimum required",
            line=dict(color='#DC2626', dash='dot'),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, float(np.max(problem.M)) * 1.05])),
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_col2:
        st.markdown("###### Monthly faculty hours")
        st.caption("Monthly faculty hours required. The red line marks your maximum monthly faculty capacity.")
        months = [f"M{m}" for m in range(1, problem.planning_horizon + 1)]
        monthly_load = active_eval.monthly_hours

        fig_workload = go.Figure()
        fig_workload.add_trace(go.Bar(
            x=months,
            y=monthly_load,
            name="Assigned Faculty Hours",
            marker_color=["#EF4444" if h > problem.monthly_capacity else "#3B82F6" for h in monthly_load],
            text=[f"{h:.0f}h" for h in monthly_load],
            textposition="auto",
        ))
        fig_workload.add_hline(
            y=problem.monthly_capacity,
            line_dash="dash",
            line_color="#DC2626",
            line_width=2,
            annotation_text=f"Capacity limit {problem.monthly_capacity:.0f} hrs/mo",
        )
        fig_workload.update_layout(
            yaxis_title="Faculty Hours Needed",
            xaxis_title=f"Implementation Month (1 to {problem.planning_horizon})",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380,
        )
        st.plotly_chart(fig_workload, use_container_width=True)

    # Earliest-Start Gantt for Chosen Plan
    st.markdown("##### Timeline")
    st.caption("Earliest start and duration for each improvement action in this plan.")
    active_action_rows = []
    for a in problem.actions:
        units = int(active_eval.x[a.index])
        if units > 0:
            active_action_rows.append({
                "Action ID": a.action_id,
                "Action Name": f"{a.action_id}: {a.name}",
                "Criterion": f"C{a.crit_id}",
                "Units": units,
                "Marks Gain": a.score_gain * units,
                "Cost (Lakh)": a.cost_lakh * units,
                "Faculty Hours": a.faculty_hours * units,
                "Start Month": a.earliest_start,
                "Finish Month": a.earliest_finish,
                "Duration": max(1, a.duration_months),
                "Prerequisites": ", ".join(a.prerequisites) if a.prerequisites else "None",
            })
    df_active_actions = pd.DataFrame(active_action_rows)

    if not df_active_actions.empty:
        fig_active_gantt = px.bar(
            df_active_actions,
            x="Duration",
            y="Action Name",
            base="Start Month",
            color="Criterion",
            orientation="h",
            hover_data=["Units", "Cost (Lakh)", "Faculty Hours", "Marks Gain", "Start Month", "Finish Month"],
            title="Earliest-Start Execution Timeline for Selected Actions",
        )
        fig_active_gantt.add_vline(x=problem.planning_horizon, line_dash="dash", line_color="red")
        fig_active_gantt.update_layout(
            height=max(320, len(df_active_actions) * 22),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_active_gantt, use_container_width=True)

        # Action checklist table
        st.markdown("##### Your action checklist")
        st.caption("Detailed breakdown of selected actions with required units, marks gained, cost, and faculty hours.")
        st.dataframe(df_active_actions, use_container_width=True)

        # Download your plan
        st.markdown("##### Download your plan")
        st.caption("Export your plan to CSV for spreadsheet planning or JSON for digital system integration.")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = df_active_actions.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download implementation roadmap (CSV)",
                data=csv_data,
                file_name="NBA_Implementation_Roadmap.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            json_dict = {
                "plan_title": plan_title,
                "total_score": active_eval.f1_total_score,
                "cost_lakh": active_eval.f2_cost_lakh,
                "faculty_hours": active_eval.f3_faculty_hours,
                "worst_ratio": active_eval.f4_worst_criterion_ratio,
                "is_feasible": active_eval.is_feasible,
                "actions": df_active_actions.to_dict(orient="records"),
            }
            st.download_button(
                label="Download blueprint (JSON)",
                data=json.dumps(json_dict, indent=2).encode("utf-8"),
                file_name="NBA_Implementation_Blueprint.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info(
            "This plan makes no changes to your current baseline. "
            "Try the **'Highest score'** or **'Lowest cost'** option above, "
            "or test custom changes in the What-If Sandbox."
        )

    # What next?
    st.markdown("---")
    with st.container(border=True):
        st.markdown("##### What next?")
        wn_col1, wn_col2 = st.columns(2)
        with wn_col1:
            if st.button("Experiment with this plan", type="primary", use_container_width=True):
                if active_eval is not None:
                    _open_whatif_with_plan(list(active_eval.x), plan_title)
                    st.rerun()
            st.caption("Adjust individual actions manually and watch the results update live.")
        with wn_col2:
            if st.button("Start over with a new scenario", key="step4_start_over_btn", use_container_width=True):
                _confirm_start_over_dialog()
            st.caption("Return to the beginning and try a different dataset or configuration.")

    # Navigation Footer for Step 4
    st.markdown("---")
    st.button("⬅ Back to Compare and choose", use_container_width=True, on_click=set_step, args=("compare",))


# ==============================================================================
# STAGE 5: WHAT-IF SCENARIO SIMULATOR (OPTIONAL BRANCH)
# ==============================================================================
elif step == "whatif":
    st.markdown('<div class="step-label">SANDBOX</div>', unsafe_allow_html=True)
    st.subheader("What-If scenario simulator")
    started_from = st.session_state.get("whatif_started_from") or "Baseline Plan"
    st.markdown(
        f"**Starting point:** {started_from}. "
        "Adjust how many times each action is done and watch your score, cost, and workload update live."
    )

    # Action lookup map for friendly names and prerequisite details
    action_map = {a.action_id: a.name for a in problem.actions}

    # Pre-evaluate current what-if vector to check feasibility and broken constraints
    current_eval = problem.evaluate(st.session_state["whatif_vector"])

    # Simulator KPI Dashboard (Status strip with caption)
    sim_k1, sim_k2, sim_k3, sim_k4 = st.columns(4)

    feas_badge = "Fully feasible" if current_eval.is_feasible else f"Infeasible (violations: {current_eval.total_violation:.1f})"
    feas_color = "delta-good" if current_eval.is_feasible else "delta-bad"

    with sim_k1:
        score_diff = current_eval.f1_total_score - problem.min_accreditation_score
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected score</div>
            <div class="metric-value">{current_eval.f1_total_score:.1f} <span style="font-size: 0.9rem; color: #6B7280;">/ 1000</span></div>
            <div class="metric-delta {'delta-good' if score_diff >= 0 else 'delta-bad'}">
                {'+' if score_diff >= 0 else ''}{score_diff:.1f} vs Target ({problem.min_accreditation_score:.0f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sim_k2:
        cost_ok = current_eval.f2_cost_lakh <= problem.budget
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected cost</div>
            <div class="metric-value">₹{current_eval.f2_cost_lakh:.2f} <span style="font-size: 0.9rem; color: #6B7280;">Lakh</span></div>
            <div class="metric-delta {'delta-good' if cost_ok else 'delta-bad'}">
                Budget limit: ₹{problem.budget:.1f}L ({'Within budget' if cost_ok else 'Over budget'})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sim_k3:
        max_monthly = float(np.max(current_eval.monthly_hours))
        workload_ok = max_monthly <= problem.monthly_capacity
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak faculty workload</div>
            <div class="metric-value">{max_monthly:.0f} <span style="font-size: 0.9rem; color: #6B7280;">hrs/mo</span></div>
            <div class="metric-delta {'delta-good' if workload_ok else 'delta-bad'}">
                Capacity limit: {problem.monthly_capacity:.0f}h ({'Compliant' if workload_ok else 'Exceeded'})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sim_k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Feasibility status</div>
            <div class="metric-value" style="font-size: 1.25rem; font-weight: 800;">{feas_badge}</div>
            <div class="metric-delta {feas_color}">
                {'Ready for Deployment' if current_eval.is_feasible else 'Violations detected below'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Updates as you change the numbers below.")

    # Violation breakdown, fix button, and caption grouped together
    if not current_eval.is_feasible:
        broken_msgs = []
        if current_eval.violations.get("budget", 0.0) > 1e-4:
            broken_msgs.append(
                f"Cost (₹{current_eval.f2_cost_lakh:.2f} Lakh) is ₹{current_eval.violations['budget']:.2f} Lakh over the budget limit of ₹{problem.budget:.1f} Lakh."
            )
        if current_eval.violations.get("monthly_hours", 0.0) > 1e-4:
            exceeded_months = [
                f"Month {m + 1} ({current_eval.monthly_hours[m]:.0f}h)"
                for m in range(len(current_eval.monthly_hours))
                if current_eval.monthly_hours[m] > problem.monthly_capacity
            ]
            if exceeded_months:
                broken_msgs.append(
                    f"Faculty workload exceeds the {problem.monthly_capacity:.0f} hrs/mo limit in: {', '.join(exceeded_months)}."
                )
            else:
                broken_msgs.append(
                    f"Faculty workload exceeds monthly capacity by {current_eval.violations['monthly_hours']:.1f} hours total."
                )
        if current_eval.violations.get("deadline", 0.0) > 1e-4:
            unsched = [
                f"{a.name} ({a.action_id})"
                for a in problem.actions
                if a.index in np.flatnonzero(problem.unschedulable_mask) and st.session_state["whatif_vector"][a.index] > 0
            ]
            if unsched:
                broken_msgs.append(
                    f"Selected actions that cannot finish within the {problem.planning_horizon}-month horizon: {', '.join(unsched)}."
                )
            else:
                broken_msgs.append(
                    f"{current_eval.violations['deadline']:.0f} action unit(s) cannot finish within the {problem.planning_horizon}-month horizon."
                )
        if current_eval.violations.get("prerequisites", 0.0) > 1e-4:
            a_idx, b_idx, u_a = problem.prereq_pairs
            broken_pairs = []
            for ai, bi, ua in zip(a_idx, b_idx, u_a):
                if st.session_state["whatif_vector"][ai] > ua * st.session_state["whatif_vector"][bi]:
                    broken_pairs.append(
                        f"'{problem.actions[ai].name} ({problem.actions[ai].action_id})' requires '{problem.actions[bi].name} ({problem.actions[bi].action_id})'"
                    )
            if broken_pairs:
                broken_msgs.append(f"Unfulfilled prerequisites: {'; '.join(broken_pairs)}.")
            else:
                broken_msgs.append(
                    f"{current_eval.violations['prerequisites']:.0f} action unit(s) have unfulfilled prerequisites."
                )
        if current_eval.violations.get("criteria_min", 0.0) > 1e-4:
            breached_crits = [
                f"C{c.id} ({c.name}: {current_eval.criteria_scores[c.id - 1]:.1f} vs min {c.min_threshold:.0f})"
                for c in problem.criteria
                if current_eval.criteria_scores[c.id - 1] < c.min_threshold
            ]
            if breached_crits:
                broken_msgs.append(f"Below minimum required score in: {', '.join(breached_crits)}.")
            else:
                broken_msgs.append(
                    f"Criteria score is {current_eval.violations['criteria_min']:.1f} marks below minimum requirements."
                )
        if current_eval.violations.get("target_score", 0.0) > 1e-4:
            broken_msgs.append(
                f"Total score ({current_eval.f1_total_score:.1f}) is {current_eval.violations['target_score']:.1f} marks short of the target ({problem.min_accreditation_score:.0f})."
            )

        warning_body = "**Current plan breaks the following constraints:**\n\n" + "\n".join(f"- {msg}" for msg in broken_msgs)
        with st.container(border=True):
            st.warning(warning_body)
            if st.button(
                "Fix violations automatically",
                type="primary",
                use_container_width=True,
                help="Automatically adjusts your plan to stay within budget, faculty hours, and all prerequisites.",
            ):
                repaired_x = problem.repair(st.session_state["whatif_vector"])
                st.session_state["whatif_vector"] = repaired_x.copy()
                st.session_state["whatif_nonce"] += 1
                st.toast("Solution successfully repaired to satisfy all constraints!")
                st.rerun()
            st.caption(
                "This adjusts your plan until it fits within budget, faculty hours, and prerequisites. "
                "You can restore your chosen plan or an empty plan using the buttons below."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Action Unit Adjusters Grouped by Criterion
    st.markdown("##### Adjust improvement actions by criterion")
    nonce = st.session_state["whatif_nonce"]

    for c in problem.criteria:
        c_actions = [a for a in problem.actions if a.crit_id == c.id]
        c_score = current_eval.criteria_scores[c.id - 1]
        status_text = " (Below minimum)" if c_score < c.min_threshold else ""
        has_active = any(st.session_state["whatif_vector"][a.index] > 0 for a in c_actions)

        with st.expander(
            f"Criterion {c.id}: {c.code} — {c.name} (Score: {c_score:.1f} / {c.max_score:.0f}){status_text}",
            expanded=has_active,
        ):
            c_cols = st.columns(3)
            for idx, a in enumerate(c_actions):
                col_target = c_cols[idx % 3]
                with col_target:
                    curr_val = int(st.session_state["whatif_vector"][a.index])
                    widget_key = f"whatif_{st.session_state['active_config']}_{nonce}_{a.index}"
                    if a.prerequisites:
                        prereq_desc = "; ".join(f"{action_map.get(pid, pid)} ({pid})" for pid in a.prerequisites)
                        prereq_help = f"Requires: {prereq_desc}"
                    else:
                        prereq_help = "Requires: None (Independent)"
                    action_help = (
                        f"**{a.name}**\n\n"
                        f"Cost: ₹{a.cost_lakh:.2f}L | Faculty Hours: {a.faculty_hours:.0f}h | "
                        f"Score Gain: +{a.score_gain:.1f} marks per unit | {prereq_help}"
                    )
                    new_val = st.number_input(
                        f"**{a.action_id}**: {a.name[:30]}",
                        min_value=0,
                        max_value=a.max_units,
                        value=min(curr_val, a.max_units),
                        step=1,
                        key=widget_key,
                        help=action_help,
                    )
                    if new_val != curr_val:
                        st.session_state["whatif_vector"][a.index] = new_val
                        st.rerun()

    # What-if Bottom Actions
    st.markdown("---")
    w_btn1, w_btn2, w_btn3 = st.columns(3)
    with w_btn1:
        if st.button("Reset to empty plan", use_container_width=True):
            st.session_state["whatif_vector"] = np.zeros(problem.num_actions, dtype=int)
            st.session_state["whatif_nonce"] += 1
            st.rerun()
    with w_btn2:
        has_sel_ovr = st.session_state.get("selected_plan_override") is not None
        if st.button("Reset to my chosen plan", disabled=not has_sel_ovr, use_container_width=True):
            if has_sel_ovr:
                st.session_state["whatif_vector"] = np.array(st.session_state["selected_plan_override"]["x"], dtype=int)
                st.session_state["whatif_nonce"] += 1
                st.rerun()
    with w_btn3:
        st.button("⬅ Back to your plan", type="primary", use_container_width=True, on_click=set_step, args=("plan",))
    st.caption("Resetting to empty clears all actions. Resetting to chosen plan restores the plan you selected in Step 3.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.8rem;">
    National Board of Accreditation (NBA) Strategic Improvement Planning System | Python 3.11+ • Streamlit
</div>
""", unsafe_allow_html=True)
