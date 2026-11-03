"""Независимая проверка нескольких настроек адаптивного контроллера.

Запуск из корня проекта: python validate_adaptive_parameters.py
Требуется run(..., controller_kwargs=...) в main.py.
"""

import csv
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, stdev

from main import run
from traffic_sim.model import Config

SEEDS = range(1000, 1100)  # новые реализации, не использованные при подборе
OUTPUT = Path("results/validation")

CANDIDATES = {
    "baseline": {"min_green": 10, "max_green": 50, "queue_threshold": 3},
    "moderate": {"min_green": 10, "max_green": 50, "queue_threshold": 8},
    "fast": {"min_green": 5, "max_green": 50, "queue_threshold": 8},
}

STATIONARY = {"N": 600., "S": 600., "E": 150., "W": 150.}
SCHEDULE = (
    (0., {"N": 600., "S": 600., "E": 150., "W": 150.}),
    (1200., {"N": 150., "S": 150., "E": 600., "W": 600.}),
    (2400., {"N": 600., "S": 600., "E": 150., "W": 150.}),
)


def config_for(scenario: str, seed: int) -> Config:
    if scenario == "stationary_80":
        return Config(rates=STATIONARY, seed=seed)
    return Config(rate_schedule=SCHEDULE, seed=seed)


def p95(values: list[float]) -> float:
    """Ближайший ранг: 95% наблюдений не превышают этот порог."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


def metrics(model) -> dict[str, float]:
    waits = {"ALL": [], "NS": [], "EW": []}
    for vehicle, departure in model.completed:
        wait = departure - vehicle.arrived_at
        waits["ALL"].append(wait)
        waits["NS" if vehicle.direction in ("N", "S") else "EW"].append(wait)

    # Считаем смены зелёной фазы за первый час, исключая первоначальный запуск.
    greens = sum(
        phase in ("NS", "EW") and t < model.config.arrival_duration
        for t, phase in model.phase_history
    )
    return {
        "mean_wait": mean(waits["ALL"]),
        "p95_all": p95(waits["ALL"]),
        "p95_NS": p95(waits["NS"]),
        "p95_EW": p95(waits["EW"]),
        "max_wait": max(waits["ALL"]),
        "switches": max(0, greens - 1),
    }


def paired_interval(baseline: list[float], candidate: list[float]):
    differences = [b - c for b, c in zip(baseline, candidate)]
    improvement = mean(differences)
    margin = 1.96 * stdev(differences) / sqrt(len(differences))
    return improvement, improvement - margin, improvement + margin


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []

    for scenario in ("stationary_80", "nonstationary"):
        print(f"\n=== {scenario} ===")
        per_name = {name: [] for name in CANDIDATES}

        for seed in SEEDS:
            config = config_for(scenario, seed)
            arrival_signatures = []
            for name, params in CANDIDATES.items():
                model = run(config, "adaptive", controller_kwargs=params)
                assert model.arrived == model.departed
                arrival_signatures.append(
                    sorted((v.direction, v.arrived_at) for v, _ in model.completed)
                )
                result = metrics(model)
                per_name[name].append(result)
                rows.append({"scenario": scenario, "seed": seed,
                             "candidate": name, **result})
            assert arrival_signatures[0] == arrival_signatures[1] == arrival_signatures[2], (
                "Входные потоки не совпали для парного сравнения"
            )

        for name, data in per_name.items():
            aggregate = {key: mean(row[key] for row in data) for key in data[0]}
            summary.append({"scenario": scenario, "candidate": name, **aggregate})
            print(
                f"{name:9s} | W={aggregate['mean_wait']:6.2f} с | "
                f"P95(ALL)={aggregate['p95_all']:6.2f} с | "
                f"P95(EW)={aggregate['p95_EW']:6.2f} с | "
                f"MAX={aggregate['max_wait']:7.2f} с | "
                f"переключений={aggregate['switches']:6.1f}"
            )

        for name in ("moderate", "fast"):
            print(f"  {name}: выигрыш относительно baseline (положительный — лучше)")
            for key in ("mean_wait", "p95_all", "p95_EW"):
                improvement, low, high = paired_interval(
                    [r[key] for r in per_name["baseline"]],
                    [r[key] for r in per_name[name]],
                )
                print(f"    {key:9s}: {improvement:+.2f} с, 95% ДИ [{low:.2f}; {high:.2f}]")

    for filename, data in (("individual_runs.csv", rows), ("summary.csv", summary)):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    print(f"\nCSV сохранены в {OUTPUT}/")


if __name__ == "__main__":
    main()
