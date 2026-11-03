"""Оценка среднего и 95-го процентиля времени ожидания для трёх контроллеров.

Запуск из корня проекта: python tail_wait_experiments.py
Не изменяет файлы или правила модели. Требует существующие main.py,
traffic_sim/model.py и nonstationary_experiments.py.
"""

import csv
from collections import defaultdict
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from main import run
from nonstationary_experiments import make_config as make_nonstationary_config
from traffic_sim.model import Config, DIRECTIONS

N_RUNS = 100
CONTROLLERS = ("fixed", "fixed_proportional", "adaptive")
NAMES = {
    "fixed": "Fixed 30/30",
    "fixed_proportional": "Fixed proportional",
    "adaptive": "Adaptive",
}
GROUPS = ("ALL", "NS", "EW", *DIRECTIONS)
SCENARIOS = ("stationary_80", "nonstationary")
OUTPUT_DIR = Path("results/tail_waits")


def make_config(scenario: str, seed: int) -> Config:
    if scenario == "stationary_80":
        return Config(
            rates={"N": 600, "S": 600, "E": 150, "W": 150},
            seed=seed,
        )
    if scenario == "nonstationary":
        return make_nonstationary_config(seed)
    raise ValueError(f"Неизвестный сценарий: {scenario}")


def empirical_p95(values: list[float]) -> float:
    """Эмпирический P95 (метод ближайшего большего ранга)."""
    if not values:
        raise ValueError("Нет автомобилей, для которых можно вычислить P95")
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


def waiting_metrics(model) -> dict[str, tuple[float, float]]:
    waits = defaultdict(list)
    for vehicle, departure in model.completed:
        delay = departure - vehicle.arrived_at
        waits["ALL"].append(delay)
        waits[vehicle.direction].append(delay)
        waits["NS" if vehicle.direction in ("N", "S") else "EW"].append(delay)
    return {
        group: (mean(waits[group]), empirical_p95(waits[group]))
        for group in GROUPS
    }


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Приближённый 95% ДИ среднего по независимым прогонам."""
    average = mean(values)
    margin = 1.96 * stdev(values) / sqrt(len(values))
    return average, average - margin, average + margin


def arrival_signature(model):
    # Порядок обслуживания может различаться, но поступления должны совпадать.
    return sorted(
        (vehicle.direction, vehicle.arrived_at)
        for vehicle, _ in model.completed
    )


def plot_p95(scenario: str, measurements: dict):
    groups = ("N", "S", "E", "W")
    positions = list(range(len(groups)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, controller in enumerate(CONTROLLERS):
        values = [
            mean(measurements[(scenario, group, controller, "p95")])
            for group in groups
        ]
        offsets = [p + (i - 1) * width for p in positions]
        ax.bar(offsets, values, width=width, label=NAMES[controller])
    ax.set_xticks(positions, groups)
    ax.set_xlabel("Направление")
    ax.set_ylabel("P95 ожидания, с")
    ax.set_title(f"95-й процентиль ожидания: {scenario} (среднее по {N_RUNS} прогонам)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"p95_{scenario}.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements = defaultdict(list)
    metrics_rows = []
    gains_rows = []

    for scenario in SCENARIOS:
        print(f"\n=== {scenario} ({N_RUNS} парных прогонов) ===")
        for seed in range(N_RUNS):
            config = make_config(scenario, seed)
            reference_signature = None
            for controller in CONTROLLERS:
                model = run(config, controller)
                if model.arrived != model.departed:
                    raise AssertionError("Есть необслуженные автомобили")
                signature = arrival_signature(model)
                if reference_signature is None:
                    reference_signature = signature
                elif reference_signature != signature:
                    raise AssertionError("Входящие потоки разных контроллеров не совпали")

                result = waiting_metrics(model)
                for group, (avg, p95) in result.items():
                    measurements[(scenario, group, controller, "mean")].append(avg)
                    measurements[(scenario, group, controller, "p95")].append(p95)

        for group in GROUPS:
            for controller in CONTROLLERS:
                avg_waits = measurements[(scenario, group, controller, "mean")]
                p95_waits = measurements[(scenario, group, controller, "p95")]
                p95_avg, low, high = mean_ci(p95_waits)
                metrics_rows.append({
                    "scenario": scenario,
                    "group": group,
                    "controller": controller,
                    "mean_wait_s": mean(avg_waits),
                    "mean_of_run_p95_s": p95_avg,
                    "run_p95_mean_ci_low_s": low,
                    "run_p95_mean_ci_high_s": high,
                })

            adaptive_values = measurements[(scenario, group, "adaptive", "p95")]
            for baseline in ("fixed", "fixed_proportional"):
                base_values = measurements[(scenario, group, baseline, "p95")]
                differences = [b - a for b, a in zip(base_values, adaptive_values)]
                gain, low, high = mean_ci(differences)
                gains_rows.append({
                    "scenario": scenario,
                    "group": group,
                    "baseline": baseline,
                    "adaptive_p95_gain_s": gain,
                    "gain_ci_low_s": low,
                    "gain_ci_high_s": high,
                })

            prop_mean = mean(measurements[(scenario, group, "fixed_proportional", "p95")])
            adaptive_mean = mean(adaptive_values)
            prop_gain = next(r for r in reversed(gains_rows)
                             if r["scenario"] == scenario and r["group"] == group
                             and r["baseline"] == "fixed_proportional")
            print(f"{group:3s} | Proportional P95: {prop_mean:7.2f} с | "
                  f"Adaptive P95: {adaptive_mean:7.2f} с | "
                  f"выигрыш: {prop_gain['adaptive_p95_gain_s']:+6.2f} с "
                  f"[{prop_gain['gain_ci_low_s']:.2f}; "
                  f"{prop_gain['gain_ci_high_s']:.2f}]")

        plot_p95(scenario, measurements)

    files = (
        ("metrics.csv", metrics_rows),
        ("gains.csv", gains_rows),
    )
    for filename, rows in files:
        with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nРезультаты сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
