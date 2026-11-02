
"""Влияние неравномерности транспортных потоков."""

import csv
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from experiments import get_mean_waits, GROUPS
from main import run
from traffic_sim.model import Config

TOTAL_RATE = 1500  # авт./час, общий поток всех направлений
NS_SHARES = (0.5, 0.6, 0.7, 0.8, 0.9)
N_RUNS = 100
OUTPUT_DIR = Path("results/sensitivity")


def make_rates(ns_share: float) -> dict[str, float]:
    return {
        "N": TOTAL_RATE * ns_share / 2,
        "S": TOTAL_RATE * ns_share / 2,
        "E": TOTAL_RATE * (1 - ns_share) / 2,
        "W": TOTAL_RATE * (1 - ns_share) / 2,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for ns_share in NS_SHARES:
        waits = {
            group: {"fixed": [], "adaptive": []}
            for group in GROUPS
        }

        for seed in range(N_RUNS):
            config = Config(rates=make_rates(ns_share), seed=seed)
            fixed = run(config, "fixed")
            adaptive = run(config, "adaptive")
            assert fixed.arrived == adaptive.arrived

            for name, model in (("fixed", fixed), ("adaptive", adaptive)):
                avg_waits = get_mean_waits(model)
                for group in GROUPS:
                    waits[group][name].append(avg_waits[group])

        for group in GROUPS:
            fixed_values = waits[group]["fixed"]
            adaptive_values = waits[group]["adaptive"]
            differences = [f - a for f, a in zip(fixed_values, adaptive_values)]
            gain = mean(differences)
            margin = 1.96 * stdev(differences) / sqrt(N_RUNS)
            rows.append({
                "ns_share": round(ns_share, 2),
                "group": group,
                "fixed_wait": mean(fixed_values),
                "adaptive_wait": mean(adaptive_values),
                "gain": gain,
                "ci_low": gain - margin,
                "ci_high": gain + margin,
            })

        total_row = rows[-len(GROUPS)]  # первая строка текущего сценария — ALL
        print(
            f"NS {ns_share:.0%} | "
            f"Fixed: {total_row['fixed_wait']:.2f} с | "
            f"Adaptive: {total_row['adaptive_wait']:.2f} с | "
            f"Выигрыш: {total_row['gain']:.2f} с "
            f"[{total_row['ci_low']:.2f}; {total_row['ci_high']:.2f}]"
        )

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    overall = [row for row in rows if row["group"] == "ALL"]
    shares = [row["ns_share"] * 100 for row in overall]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(shares, [row["fixed_wait"] for row in overall], marker="o", label="Fixed")
    ax.plot(shares, [row["adaptive_wait"] for row in overall], marker="o", label="Adaptive")
    ax.set(xlabel="Доля потока NS, %", ylabel="Среднее ожидание, с",
           title="Влияние неравномерности потоков")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_wait_vs_imbalance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(shares, [row["gain"] for row in overall], marker="o")
    ax.fill_between(
        shares,
        [row["ci_low"] for row in overall],
        [row["ci_high"] for row in overall],
        alpha=0.2,
    )
    ax.axhline(0, linewidth=1, linestyle="--", color="gray")
    ax.set(xlabel="Доля потока NS, %", ylabel="Выигрыш адаптации, с",
           title="Выигрыш адаптивного контроллера (95% ДИ)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gain_vs_imbalance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

