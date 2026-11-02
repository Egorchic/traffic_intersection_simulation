"""Сравнение трёх стратегий при изменении распределения потоков."""

import csv
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from experiments import GROUPS, get_mean_waits
from main import run
from traffic_sim.model import Config

TOTAL_RATE = 1500  # Общий поток, автомобилей в час
NS_SHARES = (0.5, 0.6, 0.7, 0.8, 0.9)
N_RUNS = 100
CONTROLLERS = ("fixed", "fixed_proportional", "adaptive")
OUTPUT_DIR = Path("results/sensitivity_three")


def make_rates(ns_share: float) -> dict[str, float]:
    return {
        "N": TOTAL_RATE * ns_share / 2,
        "S": TOTAL_RATE * ns_share / 2,
        "E": TOTAL_RATE * (1 - ns_share) / 2,
        "W": TOTAL_RATE * (1 - ns_share) / 2,
    }


def paired_gain(baseline: list[float], adaptive: list[float]):
    """Средний выигрыш и приближённый 95%-й ДИ парной разности."""
    differences = [b - a for b, a in zip(baseline, adaptive)]
    gain = mean(differences)
    margin = 1.96 * stdev(differences) / sqrt(len(differences))
    return gain, gain - margin, gain + margin


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for share in NS_SHARES:
        waits = {
            group: {controller: [] for controller in CONTROLLERS}
            for group in GROUPS
        }

        for seed in range(N_RUNS):
            config = Config(rates=make_rates(share), seed=seed)
            models = {name: run(config, name) for name in CONTROLLERS}

            # Сравниваем контроллеры на одинаковых входящих потоках.
            assert len({model.arrived for model in models.values()}) == 1

            for name, model in models.items():
                average_waits = get_mean_waits(model)
                for group in GROUPS:
                    waits[group][name].append(average_waits[group])

        for group in GROUPS:
            fixed = waits[group]["fixed"]
            proportional = waits[group]["fixed_proportional"]
            adaptive = waits[group]["adaptive"]

            gain_fixed, fixed_low, fixed_high = paired_gain(fixed, adaptive)
            gain_prop, prop_low, prop_high = paired_gain(proportional, adaptive)

            rows.append({
                "ns_share": round(share, 2),
                "group": group,
                "fixed_wait": mean(fixed),
                "proportional_wait": mean(proportional),
                "adaptive_wait": mean(adaptive),
                "gain_vs_fixed": gain_fixed,
                "fixed_ci_low": fixed_low,
                "fixed_ci_high": fixed_high,
                "gain_vs_proportional": gain_prop,
                "proportional_ci_low": prop_low,
                "proportional_ci_high": prop_high,
            })

        overall = rows[-len(GROUPS)]  # первая запись текущей доли — ALL
        print(
            f"NS {share:.0%} | "
            f"Fixed: {overall['fixed_wait']:.2f} с | "
            f"Proportional: {overall['proportional_wait']:.2f} с | "
            f"Adaptive: {overall['adaptive_wait']:.2f} с\n"
            f"    Выигрыш Adaptive над Proportional: "
            f"{overall['gain_vs_proportional']:+.2f} с "
            f"[{overall['proportional_ci_low']:.2f}; "
            f"{overall['proportional_ci_high']:.2f}]"
        )

    with (OUTPUT_DIR / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    overall_rows = [row for row in rows if row["group"] == "ALL"]
    shares = [row["ns_share"] * 100 for row in overall_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in (
        ("fixed_wait", "Fixed 30/30"),
        ("proportional_wait", "Fixed proportional"),
        ("adaptive_wait", "Adaptive"),
    ):
        ax.plot(
            shares, [row[key] for row in overall_rows], marker="o", label=label
        )
    ax.set(
        xlabel="Доля потока NS, %",
        ylabel="Среднее ожидание, с",
        title="Сравнение трёх контроллеров",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "three_controllers.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for gain_key, low_key, high_key, label in (
        ("gain_vs_fixed", "fixed_ci_low", "fixed_ci_high", "Adaptive − Fixed 30/30"),
        (
            "gain_vs_proportional",
            "proportional_ci_low",
            "proportional_ci_high",
            "Adaptive − Fixed proportional",
        ),
    ):
        ax.plot(
            shares, [row[gain_key] for row in overall_rows],
            marker="o", label=label,
        )
        ax.fill_between(
            shares,
            [row[low_key] for row in overall_rows],
            [row[high_key] for row in overall_rows],
            alpha=0.15,
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Доля потока NS, %",
        ylabel="Сокращение ожидания, с",
        title="Выигрыш адаптивного управления (95% ДИ)",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "adaptive_gain.png", dpi=160)
    plt.close(fig)

    print(f"\nРезультаты сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
