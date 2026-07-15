"""
Local experiment runner for the EA project.

Usage examples:
  python analyse.py --list
  python analyse.py --study A1_baseline
  python analyse.py --study B1_sharing_elim --benchmark tour250 --seeds 3
  python analyse.py --study all --priority 1
  python analyse.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import r0779234

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / "benchmark"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
REPORTER_CSV = ROOT / "r0779234.csv"

BENCHMARKS_ALL = ["tour50", "tour250", "tour500", "tour750", "tour1000"]
BENCHMARKS_FAST = ["tour50", "tour250"]
BENCHMARKS_MEDIUM = ["tour50", "tour250", "tour500"]
SEEDS_DEFAULT = 5

# -----------------------------------------------------------------------------
# Baseline = your submission candidate (update as you tune)
# -----------------------------------------------------------------------------

def baseline_config() -> r0779234.EAConfig:
    return r0779234.EAConfig(
        llambda=50,
        mu=100,
        alpha=0.2,
        min_alpha=0.05,
        nn_init_share=0.2,
        lso_init_share=0.5,
        lso_variation_share=0.1,          # tune down from 1.0 for speed
        lso_max_changes=20,
        local_search_initialisation=False,
        local_search_variation=False,
        alpha_fitness_sharing=1.0,
        sigma_share=0.1,
        fitness_sharing_selection=False,
        fitness_sharing_elimination=True,
        crowding=False,
        k_selection=3,
        k_elimination=5,
        k_elitism=1,
        elitism=True,
        island_model=False,
        alpha_self_ad=True,
        elimination="lambda_mu",
        selection="tournament",
        initialisation="random",              # switch to "random" until NN is done
        recombination="ox",
        mutations=("swap", "insert", "inversion"),
        local_search="2opt",
    )


# -----------------------------------------------------------------------------
# Experiment specification
# -----------------------------------------------------------------------------

@dataclass
class ExperimentSpec:
    id: str
    study: str
    description: str
    priority: int
    benchmarks: list[str]
    seeds: int
    config_factory: Callable[[], r0779234.EAConfig]
    tags: list[str] = field(default_factory=list)


def _cfg(**kwargs) -> Callable[[], r0779234.EAConfig]:
    """Build config factory: baseline + overrides."""
    def factory() -> r0779234.EAConfig:
        return replace(baseline_config(), **kwargs)
    return factory


# -----------------------------------------------------------------------------
# FULL EXPERIMENT MATRIX
# -----------------------------------------------------------------------------

EXPERIMENTS: list[ExperimentSpec] = [
    # --- A: Baseline ---
    ExperimentSpec(
        id="A1_baseline",
        study="A_baseline",
        description="Tuned default on all instances",
        priority=1,
        benchmarks=BENCHMARKS_ALL,
        seeds=10,
        config_factory=baseline_config,
        tags=["baseline", "report"],
    ),

    # --- B: Diversity ---
    ExperimentSpec(
        id="B0_no_diversity",
        study="B_diversity",
        description="No fitness sharing, no crowding",
        priority=1,
        benchmarks=BENCHMARKS_MEDIUM,
        seeds=5,
        config_factory=_cfg(
            fitness_sharing_elimination=False,
            fitness_sharing_selection=False,
            crowding=False,
        ),
        tags=["ablation", "diversity"],
    ),
    ExperimentSpec(
        id="B1_sharing_elim",
        study="B_diversity",
        description="Fitness sharing at elimination (baseline)",
        priority=1,
        benchmarks=BENCHMARKS_MEDIUM,
        seeds=5,
        config_factory=_cfg(
            fitness_sharing_elimination=True,
            fitness_sharing_selection=False,
            crowding=False,
        ),
        tags=["ablation", "diversity"],
    ),
    ExperimentSpec(
        id="B2_sharing_sel",
        study="B_diversity",
        description="Fitness sharing at selection only",
        priority=2,
        benchmarks=BENCHMARKS_FAST,
        seeds=5,
        config_factory=_cfg(
            fitness_sharing_elimination=False,
            fitness_sharing_selection=True,
            crowding=False,
        ),
        tags=["ablation", "diversity"],
    ),
    ExperimentSpec(
        id="B3_sharing_both",
        study="B_diversity",
        description="Fitness sharing at selection and elimination",
        priority=2,
        benchmarks=BENCHMARKS_FAST,
        seeds=5,
        config_factory=_cfg(
            fitness_sharing_elimination=True,
            fitness_sharing_selection=True,
            crowding=False,
        ),
        tags=["ablation", "diversity"],
    ),
    ExperimentSpec(
        id="B4_crowding",
        study="B_diversity",
        description="Crowding instead of fitness sharing",
        priority=1,
        benchmarks=BENCHMARKS_MEDIUM,
        seeds=5,
        config_factory=_cfg(
            fitness_sharing_elimination=False,
            fitness_sharing_selection=False,
            crowding=True,
        ),
        tags=["ablation", "diversity"],
    ),
    # sigma sweep — one experiment per value (easier plotting)
    *[
        ExperimentSpec(
            id=f"B5_sigma_{sigma}",
            study="B_diversity",
            description=f"sigma_share={sigma}",
            priority=2,
            benchmarks=["tour250"],
            seeds=3,
            config_factory=_cfg(sigma_share=sigma),
            tags=["sweep", "sigma"],
        )
        for sigma in (0.05, 0.1, 0.15, 0.2)
    ],

    # --- C: Initialization ---
    ExperimentSpec(
        id="C1_random_init",
        study="C_initialization",
        description="Random initialization",
        priority=1,
        benchmarks=BENCHMARKS_MEDIUM,
        seeds=5,
        config_factory=_cfg(initialisation="random"),
        tags=["ablation", "init"],
    ),
    ExperimentSpec(
        id="C2_nn_init",
        study="C_initialization",
        description="Nearest-neighbor initialization",
        priority=1,
        benchmarks=BENCHMARKS_MEDIUM,
        seeds=5,
        config_factory=_cfg(initialisation="nn"),
        tags=["ablation", "init"],
    ),
    ExperimentSpec(
        id="C3_nn_no_ls",
        study="C_initialization",
        description="NN init without init local search",
        priority=2,
        benchmarks=["tour250"],
        seeds=3,
        config_factory=_cfg(initialisation="nn", local_search_initialisation=False),
        tags=["ablation", "init"],
    ),

    # --- D: Local search ---
    ExperimentSpec(
        id="D0_no_ls",
        study="D_local_search",
        description="No local search",
        priority=1,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(
            local_search_initialisation=False,
            local_search_variation=False,
        ),
        tags=["ablation", "ls"],
    ),
    ExperimentSpec(
        id="D1_ls_init_only",
        study="D_local_search",
        description="Local search at initialization only",
        priority=1,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(
            local_search_initialisation=True,
            local_search_variation=False,
        ),
        tags=["ablation", "ls"],
    ),
    ExperimentSpec(
        id="D2_ls_var_only",
        study="D_local_search",
        description="Local search after variation only",
        priority=1,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(
            local_search_initialisation=False,
            local_search_variation=True,
        ),
        tags=["ablation", "ls"],
    ),
    ExperimentSpec(
        id="D3_ls_both",
        study="D_local_search",
        description="Local search at init and variation",
        priority=1,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(
            local_search_initialisation=True,
            local_search_variation=True,
        ),
        tags=["ablation", "ls"],
    ),
    *[
        ExperimentSpec(
            id=f"D4_ls_share_{share}",
            study="D_local_search",
            description=f"lso_variation_share={share}",
            priority=2,
            benchmarks=["tour250"],
            seeds=3,
            config_factory=_cfg(lso_variation_share=share),
            tags=["sweep", "ls"],
        )
        for share in (0.1, 0.2, 0.5, 1.0)
    ],
    *[
        ExperimentSpec(
            id=f"D5_ls_depth_{depth}",
            study="D_local_search",
            description=f"lso_max_changes={depth}",
            priority=2,
            benchmarks=["tour250"],
            seeds=3,
            config_factory=_cfg(lso_max_changes=depth),
            tags=["sweep", "ls"],
        )
        for depth in (20, 50, 100, 200)
    ],

    # --- E: Operators ---
    ExperimentSpec(
        id="E1_crossover_ox",
        study="E_operators",
        description="Order crossover",
        priority=2,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(recombination="ox"),
        tags=["ablation", "crossover"],
    ),
    ExperimentSpec(
        id="E2_crossover_pmx",
        study="E_operators",
        description="PMX crossover",
        priority=2,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(recombination="pmx"),
        tags=["ablation", "crossover"],
    ),
    ExperimentSpec(
        id="E3_crossover_cx",
        study="E_operators",
        description="Cycle crossover",
        priority=2,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(recombination="cx"),
        tags=["ablation", "crossover"],
    ),
    ExperimentSpec(
        id="E4_mut_scramble",
        study="E_operators",
        description="Scramble mutation only",
        priority=2,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(mutations=("scramble",)),
        tags=["ablation", "mutation"],
    ),
    ExperimentSpec(
        id="E5_mut_pool",
        study="E_operators",
        description="Swap + insert + inversion mutations",
        priority=2,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(mutations=("swap", "insert", "inversion")),
        tags=["ablation", "mutation"],
    ),
    ExperimentSpec(
        id="E8_ls_3opt",
        study="E_operators",
        description="3-opt local search",
        priority=3,
        benchmarks=["tour250"],
        seeds=3,
        config_factory=_cfg(local_search="3opt"),
        tags=["ablation", "ls"],
    ),

    # --- F: Population ---
    ExperimentSpec(
        id="F1_pop_small",
        study="F_population",
        description="Small population",
        priority=2,
        benchmarks=["tour250"],
        seeds=3,
        config_factory=_cfg(llambda=30, mu=60),
        tags=["ablation", "population"],
    ),
    ExperimentSpec(
        id="F3_pop_large",
        study="F_population",
        description="Large population",
        priority=2,
        benchmarks=["tour250"],
        seeds=3,
        config_factory=_cfg(llambda=80, mu=160),
        tags=["ablation", "population"],
    ),
    ExperimentSpec(
        id="F5_elim_k_tournament",
        study="F_population",
        description="k-tournament elimination",
        priority=2,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(elimination="k_tournament"),
        tags=["ablation", "elimination"],
    ),

    # --- G: Islands ---
    ExperimentSpec(
        id="G0_single_island",
        study="G_islands",
        description="Single population",
        priority=2,
        benchmarks=["tour250", "tour500"],
        seeds=5,
        config_factory=_cfg(island_model=False),
        tags=["ablation", "islands"],
    ),
    ExperimentSpec(
        id="G1_islands",
        study="G_islands",
        description="Two islands, no threading",
        priority=2,
        benchmarks=["tour250", "tour500"],
        seeds=5,
        config_factory=_cfg(island_model=True, threading=False),
        tags=["ablation", "islands"],
    ),
    ExperimentSpec(
        id="G2_islands_threaded",
        study="G_islands",
        description="Two islands with threading",
        priority=3,
        benchmarks=["tour500"],
        seeds=3,
        config_factory=_cfg(island_model=True, threading=True),
        tags=["ablation", "islands"],
    ),

    # --- H: Self-adaptation ---
    ExperimentSpec(
        id="H1_no_self_ad",
        study="H_self_adaptation",
        description="Fixed mutation rate",
        priority=3,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(alpha_self_ad=False, alpha=0.2),
        tags=["ablation", "self_ad"],
    ),
    ExperimentSpec(
        id="H2_self_ad",
        study="H_self_adaptation",
        description="Self-adaptive mutation rate",
        priority=3,
        benchmarks=["tour250"],
        seeds=5,
        config_factory=_cfg(alpha_self_ad=True),
        tags=["ablation", "self_ad"],
    ),

    # --- I: Scalability (alias of baseline, separate tag for plotting) ---
    ExperimentSpec(
        id="I1_scaling",
        study="I_scalability",
        description="Baseline scalability across all sizes",
        priority=1,
        benchmarks=BENCHMARKS_ALL,
        seeds=10,
        config_factory=baseline_config,
        tags=["scaling", "report"],
    ),
]

EXPERIMENT_BY_ID = {e.id: e for e in EXPERIMENTS}


# -----------------------------------------------------------------------------
# Run result + I/O
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    experiment_id: str
    study: str
    benchmark: str
    seed: int
    final_mean: float
    final_best: float
    initial_best: float
    elapsed_s: float
    iterations: int
    iterations_per_sec: float
    csv_path: str
    timestamp: str
    config: dict[str, Any]


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, RAW_DIR, PLOTS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_reporter_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def config_to_dict(cfg: r0779234.EAConfig) -> dict[str, Any]:
    skip = {
        "elimination_fn",
        "selection_fn",
        "initialisation_fn",
        "recombination_fn",
        "mutation_fns",      # <-- this was missing
        "local_search_fn",
        "length",
        "sigma",
        "nn_init_size",
        "random_init_size",
        "swap_size",
        "local_search_init_count",
        "local_search_variation_count",
    }
    return {k: v for k, v in asdict(cfg).items() if k not in skip}


def run_single(
    experiment: ExperimentSpec,
    benchmark: str,
    seed: int,
) -> RunResult:
    bench_path = BENCH_DIR / f"{benchmark}.csv"
    if not bench_path.exists():
        raise FileNotFoundError(f"Missing benchmark: {bench_path}")

    np.random.seed(seed)
    cfg = experiment.config_factory()

    t0 = time.time()
    ea = r0779234.r0779234(cfg)
    ea.optimize(str(bench_path), config=cfg)
    wall = time.time() - t0

    if not REPORTER_CSV.exists():
        raise FileNotFoundError(f"Reporter output missing: {REPORTER_CSV}")

    iters, times, means, bests = load_reporter_csv(REPORTER_CSV)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"{experiment.id}__{benchmark}__seed{seed}__{stamp}.csv"
    out_path = RAW_DIR / out_name
    shutil.copy(REPORTER_CSV, out_path)

    elapsed = float(times[-1])
    n_iters = int(iters[-1]) + 1
    ips = n_iters / elapsed if elapsed > 0 else 0.0

    return RunResult(
        experiment_id=experiment.id,
        study=experiment.study,
        benchmark=benchmark,
        seed=seed,
        final_mean=float(means[-1]),
        final_best=float(bests[-1]),
        initial_best=float(bests[0]),
        elapsed_s=elapsed,
        iterations=n_iters,
        iterations_per_sec=ips,
        csv_path=str(out_path.relative_to(ROOT)),
        timestamp=stamp,
        config=config_to_dict(cfg),
    )


def run_experiment(
    experiment: ExperimentSpec,
    benchmarks: list[str] | None = None,
    seeds: int | None = None,
) -> list[RunResult]:
    benchmarks = benchmarks or experiment.benchmarks
    seeds = seeds or experiment.seeds
    results: list[RunResult] = []

    total = len(benchmarks) * seeds
    label = f"{experiment.id}"
    with tqdm(total=total, desc=label) as pbar:
        for benchmark in benchmarks:
            for seed in range(seeds):
                results.append(run_single(experiment, benchmark, seed))
                pbar.update(1)

    return results


def append_summary(results: list[RunResult]) -> None:
    ensure_dirs()
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            existing = json.load(f)
    else:
        existing = []

    existing.extend([asdict(r) for r in results])
    with open(SUMMARY_PATH, "w") as f:
        json.dump(existing, f, indent=2)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_convergence_curve(csv_path: Path, title: str, out_path: Path) -> None:
    _, times, means, bests = load_reporter_csv(csv_path)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(times, means, label="Mean tour length")
    plt.plot(times, bests, label="Best tour length")
    plt.xlabel("Time [s]")
    plt.ylabel("Tour length")
    plt.xlim(0, 300)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_study_bar(summary: list[dict], study: str, benchmark: str, out_path: Path) -> None:
    """Bar chart: experiment_id vs mean final best (with std error)."""
    rows = [r for r in summary if r["study"] == study and r["benchmark"] == benchmark]
    if not rows:
        return

    by_exp: dict[str, list[float]] = {}
    for r in rows:
        by_exp.setdefault(r["experiment_id"], []).append(r["final_best"])

    names = sorted(by_exp)
    means = [np.mean(by_exp[k]) for k in names]
    stds = [np.std(by_exp[k], ddof=1) if len(by_exp[k]) > 1 else 0.0 for k in names]

    plt.figure(figsize=(10, 6), dpi=120)
    x = np.arange(len(names))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Final best tour length")
    plt.title(f"{study} on {benchmark}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_histogram(summary: list[dict], experiment_id: str, benchmark: str, out_path: Path) -> None:
    vals = [
        r["final_best"]
        for r in summary
        if r["experiment_id"] == experiment_id and r["benchmark"] == benchmark
    ]
    if not vals:
        return
    plt.figure(figsize=(8, 6), dpi=120)
    plt.hist(vals, bins=min(10, len(vals)))
    plt.xlabel("Final best tour length")
    plt.ylabel("Count")
    plt.title(f"{experiment_id} on {benchmark} (n={len(vals)})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scaling(summary: list[dict], experiment_id: str, out_path: Path) -> None:
    sizes = {"tour50": 50, "tour250": 250, "tour500": 500, "tour750": 750, "tour1000": 1000}
    rows = [r for r in summary if r["experiment_id"] == experiment_id]
    if not rows:
        return

    by_bench: dict[str, list[float]] = {}
    for r in rows:
        by_bench.setdefault(r["benchmark"], []).append(r["final_best"])

    xs, ys, yerr = [], [], []
    for bench in BENCHMARKS_ALL:
        if bench not in by_bench:
            continue
        xs.append(sizes[bench])
        ys.append(np.mean(by_bench[bench]))
        yerr.append(np.std(by_bench[bench], ddof=1) if len(by_bench[bench]) > 1 else 0.0)

    plt.figure(figsize=(8, 6), dpi=120)
    plt.errorbar(xs, ys, yerr=yerr, marker="o")
    plt.xlabel("Number of cities")
    plt.ylabel("Final best tour length")
    plt.title(f"Scaling: {experiment_id}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def export_tables(summary: list[dict]) -> None:
    ensure_dirs()
    # CSV table: experiment × benchmark aggregate
    lines = ["experiment_id,study,benchmark,n,final_best_mean,final_best_std,final_mean_mean,iter_per_sec_mean"]
    keys = sorted({(r["experiment_id"], r["study"], r["benchmark"]) for r in summary})
    for exp_id, study, bench in keys:
        rows = [r for r in summary if r["experiment_id"] == exp_id and r["benchmark"] == bench]
        bests = [r["final_best"] for r in rows]
        means = [r["final_mean"] for r in rows]
        ips = [r["iterations_per_sec"] for r in rows]
        line = ",".join([
            exp_id, study, bench, str(len(rows)),
            f"{np.mean(bests):.4f}",
            f"{np.std(bests, ddof=1) if len(bests) > 1 else 0:.4f}",
            f"{np.mean(means):.4f}",
            f"{np.mean(ips):.4f}",
        ])
        lines.append(line)

    out = TABLES_DIR / "aggregate_results.csv"
    out.write_text("\n".join(lines))


def regenerate_plots() -> None:
    if not SUMMARY_PATH.exists():
        print("No summary.json found.")
        return

    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    ensure_dirs()
    export_tables(summary)

    # Best convergence curve per (experiment, benchmark)
    seen: set[tuple[str, str]] = set()
    for r in summary:
        key = (r["experiment_id"], r["benchmark"])
        if key in seen:
            continue
        seen.add(key)
        csv_path = ROOT / r["csv_path"]
        if csv_path.exists():
            out = PLOTS_DIR / f"convergence_{r['experiment_id']}_{r['benchmark']}.pdf"
            plot_convergence_curve(csv_path, f"{r['experiment_id']} on {r['benchmark']}", out)

    for study in sorted({r["study"] for r in summary}):
        for bench in BENCHMARKS_ALL:
            out = PLOTS_DIR / f"bar_{study}_{bench}.pdf"
            plot_study_bar(summary, study, bench, out)

    plot_scaling(summary, "A1_baseline", PLOTS_DIR / "scaling_A1_baseline.pdf")
    plot_scaling(summary, "I1_scaling", PLOTS_DIR / "scaling_I1_scaling.pdf")

    for exp_id in sorted({r["experiment_id"] for r in summary}):
        for bench in BENCHMARKS_ALL:
            vals = [r for r in summary if r["experiment_id"] == exp_id and r["benchmark"] == bench]
            if len(vals) >= 3:
                plot_histogram(summary, exp_id, bench, PLOTS_DIR / f"hist_{exp_id}_{bench}.pdf")

    print(f"Plots written to {PLOTS_DIR}")
    print(f"Tables written to {TABLES_DIR}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def list_experiments(priority: int | None = None) -> None:
    print(f"{'ID':<22} {'P':<2} {'Study':<18} {'Seeds':<5} Benchmarks")
    print("-" * 80)
    for e in EXPERIMENTS:
        if priority is not None and e.priority != priority:
            continue
        print(f"{e.id:<22} {e.priority:<2} {e.study:<18} {e.seeds:<5} {','.join(e.benchmarks)}")


def select_experiments(
    study: str,
    priority: int | None,
) -> list[ExperimentSpec]:
    if study == "all":
        exps = EXPERIMENTS
    elif study in EXPERIMENT_BY_ID:
        exps = [EXPERIMENT_BY_ID[study]]
    else:
        exps = [e for e in EXPERIMENTS if e.study == study or e.id.startswith(study)]

    if priority is not None:
        exps = [e for e in exps if e.priority == priority]
    return exps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EA experiments and generate report figures.")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--study", type=str, default=None, help="Experiment id, study name, or 'all'")
    parser.add_argument("--benchmark", type=str, default=None, help="Override benchmark, e.g. tour50")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds")
    parser.add_argument("--priority", type=int, default=None, help="Only run experiments with this priority")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate plots from summary.json")
    args = parser.parse_args()

    ensure_dirs()

    if args.list:
        list_experiments(args.priority)
        return

    if args.plot_only:
        regenerate_plots()
        return

    if not args.study:
        parser.print_help()
        return

    experiments = select_experiments(args.study, args.priority)
    if not experiments:
        print("No matching experiments.")
        return

    benchmarks_override = [args.benchmark] if args.benchmark else None
    all_results: list[RunResult] = []

    for exp in experiments:
        print(f"\n=== {exp.id}: {exp.description} ===")
        results = run_experiment(
            exp,
            benchmarks=benchmarks_override,
            seeds=args.seeds,
        )
        all_results.extend(results)

    append_summary(all_results)
    regenerate_plots()
    print(f"\nDone. {len(all_results)} runs appended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()