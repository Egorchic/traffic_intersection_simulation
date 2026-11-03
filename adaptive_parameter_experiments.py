"""Чувствительность адаптивного светофора к параметрам.

Запуск из корня проекта: python adaptive_parameter_experiments.py
Требует main.run(..., controller_kwargs=...) из инструкции к обновлению.
"""

import csv
from collections import defaultdict
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from main import run
from traffic_sim.model import Config

N_RUNS = 100
OUTPUT_DIR = Path("results/adaptive_parameters")
DEFAULT = {"min_green": 10.0, "max_green": 50.0, "queue_threshold": 3}
SWEEPS = {
    "queue_threshold": (0, 1, 3, 5, 8, 12),
    "min_green": (5.0, 10.0, 15.0, 20.0, 30.0),
    "max_green": (30.0, 50.0, 70.0, 90.0),
}
SCENARIOS = ("stationary_80", "nonstationary")
NONSTATIONARY_RATES = (
    (0.0, {"N": 600.0, "S": 600.0, "E": 150.0, "W": 150.0}),
    (1200.0, {"N": 150.0, "S": 150.0, "E": 600.0, "W": 600.0}),
    (2400.0, {"N": 600.0, "S": 600.0, "E": 150.0, "W": 150.0}),
)
METRICS = ("mean_wait", "p95_all", "p95_ns", "p95_ew", "max_wait", "switches", "remaining")


def make_config(scenario: str, seed: int) -> Config:
    if scenario == "stationary_80":
        return Config(rates={"N": 600.0, "S": 600.0, "E": 150.0, "W": 150.0}, seed=seed)
    if scenario == "nonstationary":
        return Config(rate_schedule=NONSTATIONARY_RATES, seed=seed)
    raise ValueError(f"Неизвестный сценарий: {scenario}")


def p95(values: list[float]) -> float:
    """Эмпирический процентиль: nearest rank, как в tail_wait_experiments.py."""
    if not values:
        raise ValueError("Нет обслуженных автомобилей")
    values = sorted(values)
    return values[ceil(0.95 * len(values)) - 1]


def extract_metrics(model) -> dict[str, float]:
    groups = {"ALL": [], "NS": [], "EW": []}
    for vehicle, departure in model.completed:
        wait = departure - vehicle.arrived_at
        groups["ALL"].append(wait)
        groups["NS" if vehicle.direction in ("N", "S") else "EW"].append(wait)

    if not all(groups.values()):
        raise ValueError("Недостаточно прибытий, чтобы вычислить все P95")

    # phase_history включает None для переходных сигналов; оставляем только зелёные.
    greens = [
        phase for time, phase in model.phase_history
        if time < model.config.arrival_duration and phase in ("NS", "EW")
    ]

    return {
        "mean_wait": mean(groups["ALL"]),
        "p95_all": p95(groups["ALL"]),
        "p95_ns": p95(groups["NS"]),
        "p95_ew": p95(groups["EW"]),
        "max_wait": max(groups["ALL"]),
        "switches": max(0, len(greens) - 1),
        "remaining": sum(model.at_horizon["queue_lengths"].values()),
    }


def ci_of_paired_gain(reference: list[float], candidate: list[float]):
    """Положительно: кандидат лучше исходного набора DEFAULT."""
    differences = [x - y for x, y in zip(reference, candidate)]
    gain = mean(differences)
    margin = 1.96 * stdev(differences) / sqrt(len(differences))
    return gain, gain - margin, gain + margin


def arrival_signature(model):
    return sorted((v.direction, v.arrived_at) for v, _ in model.completed)


def run_scenario(scenario: str):
    measurements = defaultdict(list)
    baseline = defaultdict(list)

    for seed in range(N_RUNS):
        config = make_config(scenario, seed)
        model_base = run(config, "adaptive", controller_kwargs=DEFAULT)
        assert model_base.arrived == model_base.departed
        baseline_signature = arrival_signature(model_base)
        base_metrics = extract_metrics(model_base)
        for key, value in base_metrics.items():
            baseline[key].append(value)

        for parameter, values in SWEEPS.items():
            for value in values:
                if value == DEFAULT[parameter]:
                    metrics = base_metrics
                else:
                    params = {**DEFAULT, parameter: value}
                    model = run(config, "adaptive", controller_kwargs=params)
                    assert model.arrived == model.departed
                    assert arrival_signature(model) == baseline_signature, (
                        "Различаются входящие потоки в парных запусках"
                    )
                    metrics = extract_metrics(model)
                for key, result in metrics.items():
                    measurements[(parameter, value, key)].append(result)

    rows = []
    for parameter, values in SWEEPS.items():
        for value in values:
            row = {"scenario": scenario, "parameter": parameter, "value": value}
            for metric in METRICS:
                row[metric] = mean(measurements[(parameter, value, metric)])
            for metric in ("mean_wait", "p95_all", "p95_ew"):
                gain, low, high = ci_of_paired_gain(
                    baseline[metric], measurements[(parameter, value, metric)]
                )
                row[f"{metric}_gain_vs_default"] = gain
                row[f"{metric}_gain_ci_low"] = low
                row[f"{metric}_gain_ci_high"] = high
            rows.append(row)
    return rows


def plot_results(rows: list[dict], scenario: str):
    plots = (
        ("mean_wait", "Среднее ожидание, с"),
        ("p95_ew", "P95 ожидания EW, с"),
        ("switches", "Переключений за час"),
    )
    for parameter, values in SWEEPS.items():
        subset = [r for r in rows if r["parameter"] == parameter]
        subset.sort(key=lambda r: r["value"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (metric, ylabel) in zip(axes, plots):
            ax.plot([r["value"] for r in subset],
                    [r[metric] for r in subset], marker="o")
            baseline = next(r[metric] for r in subset if r["value"] == DEFAULT[parameter])
            ax.axhline(baseline, color="gray", linestyle="--", alpha=0.65)
            ax.set(xlabel=parameter, ylabel=ylabel)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Чувствительность к {parameter}: {scenario}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{scenario}_{parameter}.png", dpi=160)
        plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for scenario in SCENARIOS:
        print(f"\n=== {scenario}: {N_RUNS} парных прогонов ===")
        rows = run_scenario(scenario)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{row['parameter']:17s}={row['value']:5.0f} | "
                f"W={row['mean_wait']:6.2f} c | "
                f"P95(EW)={row['p95_ew']:6.2f} c | "
                f"переключений={row['switches']:5.1f} | "
                f"выигрыш W против исходного: {row['mean_wait_gain_vs_default']:+6.2f} c"
            )
        plot_results(rows, scenario)

    with (OUTPUT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nРезультаты: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
