"""Исследование влияния общей интенсивности транспортных потоков.

Доля NS фиксирована (80%), общая интенсивность меняется.
Каждый seed используется для всех трёх контроллеров одного сценария.
"""

import csv
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from main import run
from traffic_sim.model import Config

TOTAL_RATES = (300, 600, 900, 1200, 1500, 1800, 2100)
NS_SHARE = 0.8
N_RUNS = 100
CONTROLLERS = ("fixed", "fixed_proportional", "adaptive")
OUTPUT_DIR = Path("results/intensity")


def make_rates(total_rate: float) -> dict[str, float]:
    return {
        "N": total_rate * NS_SHARE / 2,
        "S": total_rate * NS_SHARE / 2,
        "E": total_rate * (1 - NS_SHARE) / 2,
        "W": total_rate * (1 - NS_SHARE) / 2,
    }


def mean_wait(model) -> float:
    if not model.completed:
        return 0.0
    return mean(departure - vehicle.arrived_at
                for vehicle, departure in model.completed)


def paired_gain(base: list[float], adaptive: list[float]) -> tuple[float, float, float]:
    differences = [x - y for x, y in zip(base, adaptive)]
    gain = mean(differences)
    margin = 1.96 * stdev(differences) / sqrt(len(differences))
    return gain, gain - margin, gain + margin


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for total_rate in TOTAL_RATES:
        waits = {name: [] for name in CONTROLLERS}
        remaining = {name: [] for name in CONTROLLERS}
        clearance = {name: [] for name in CONTROLLERS}

        for seed in range(N_RUNS):
            config = Config(rates=make_rates(total_rate), seed=seed)
            arrivals = []

            for name in CONTROLLERS:
                model = run(config, name)
                arrivals.append(model.arrived)
                waits[name].append(mean_wait(model))
                remaining[name].append(sum(model.at_horizon["queue_lengths"].values()))
                clearance[name].append(model.env.now - config.arrival_duration)

                assert model.arrived == model.departed

            # Проверяем парность эксперимента по входным потокам.
            assert len(set(arrivals)) == 1

        gain_fixed = paired_gain(waits["fixed"], waits["adaptive"])
        gain_proportional = paired_gain(
            waits["fixed_proportional"], waits["adaptive"]
        )

        row = {"total_rate": total_rate}
        for name in CONTROLLERS:
            row[f"{name}_wait"] = mean(waits[name])
            row[f"{name}_remaining_at_hour"] = mean(remaining[name])
            row[f"{name}_clearance_after_hour"] = mean(clearance[name])

        for label, values in (
            ("gain_vs_fixed", gain_fixed),
            ("gain_vs_proportional", gain_proportional),
        ):
            row[label] = values[0]
            row[f"{label}_ci_low"] = values[1]
            row[f"{label}_ci_high"] = values[2]

        rows.append(row)
        print(
            f"Общий поток {total_rate:4d} авт./ч | "
            f"Fixed: {row['fixed_wait']:7.2f} с | "
            f"Proportional: {row['fixed_proportional_wait']:7.2f} с | "
            f"Adaptive: {row['adaptive_wait']:7.2f} с | "
            f"выигрыш над Proportional: {gain_proportional[0]:+.2f} с "
            f"[{gain_proportional[1]:.2f}; {gain_proportional[2]:.2f}]"
        )

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = [row["total_rate"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, label in (
        ("fixed", "Fixed 30/30"),
        ("fixed_proportional", "Fixed proportional"),
        ("adaptive", "Adaptive"),
    ):
        ax.plot(x, [row[f"{name}_wait"] for row in rows], marker="o", label=label)
    ax.set(xlabel="Общая интенсивность, авт./ч", ylabel="Среднее ожидание, с",
           title="Влияние общей интенсивности при NS = 80%")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_wait_vs_intensity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for prefix, label in (
        ("gain_vs_fixed", "Выигрыш над Fixed 30/30"),
        ("gain_vs_proportional", "Выигрыш над Fixed proportional"),
    ):
        y = [row[prefix] for row in rows]
        lower = [row[f"{prefix}_ci_low"] for row in rows]
        upper = [row[f"{prefix}_ci_high"] for row in rows]
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, lower, upper, alpha=0.15)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Общая интенсивность, авт./ч",
           ylabel="Сокращение среднего ожидания, с",
           title="Выигрыш адаптивного управления (95% ДИ)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gain_vs_intensity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, label in (
        ("fixed", "Fixed 30/30"),
        ("fixed_proportional", "Fixed proportional"),
        ("adaptive", "Adaptive"),
    ):
        ax.plot(x, [row[f"{name}_remaining_at_hour"] for row in rows],
                marker="o", label=label)
    ax.set(xlabel="Общая интенсивность, авт./ч",
           ylabel="Среднее число автомобилей в очередях через час",
           title="Остаточные очереди при NS = 80%")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "remaining_vs_intensity.png", dpi=160)
    plt.close(fig)
    print(f"\nРезультаты сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
