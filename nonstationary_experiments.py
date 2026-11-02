"""Смена доминирующего транспортного потока каждые 20 минут."""

import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from main import run
from traffic_sim.model import Config

N_RUNS = 100
HORIZON = 3600.0
PERIOD_STARTS = (0.0, 1200.0, 2400.0)
CONTROLLERS = ("fixed", "fixed_proportional", "adaptive")
LABELS = {
    "fixed": "Fixed 30/30",
    "fixed_proportional": "Fixed proportional",
    "adaptive": "Adaptive",
}
OUTPUT_DIR = Path("results/nonstationary")

# Общий поток 1500 авт./ч во всех периодах.
RATE_SCHEDULE = (
    (0.0, {"N": 600.0, "S": 600.0, "E": 150.0, "W": 150.0}),
    (1200.0, {"N": 150.0, "S": 150.0, "E": 600.0, "W": 600.0}),
    (2400.0, {"N": 600.0, "S": 600.0, "E": 150.0, "W": 150.0}),
)
PERIOD_LABELS = ("0–20", "20–40", "40–60")
GROUPS = ("ALL", "NS", "EW")


def make_config(seed: int) -> Config:
    # rates нужен для совместимости со старыми конфигурациями;
    # когда задан rate_schedule, входящий поток берётся именно из расписания.
    return Config(arrival_duration=HORIZON, rate_schedule=RATE_SCHEDULE, seed=seed)


def arrival_signature(model):
    """Проверяем идентичность входящих потоков для всех стратегий."""
    return sorted((vehicle.direction, vehicle.arrived_at)
                  for vehicle, _ in model.completed)


def wait_groups(model):
    """Группируем по времени ПРИБЫТИЯ, а не проезда автомобиля."""
    buckets = defaultdict(list)
    for vehicle, departure in model.completed:
        period = min(int(vehicle.arrived_at // 1200), 2)
        wait = departure - vehicle.arrived_at
        buckets[("ALL", period)].append(wait)
        group = "NS" if vehicle.direction in ("N", "S") else "EW"
        buckets[(group, period)].append(wait)
        buckets[("ALL", "total")].append(wait)

    results = {}
    for group in GROUPS:
        for period in range(3):
            values = buckets[(group, period)]
            results[(group, period)] = mean(values) if values else float("nan")
    results[("ALL", "total")] = mean(buckets[("ALL", "total")])
    return results


def confidence_interval(values):
    """Приближённый двусторонний 95% ДИ среднего по независимым прогонам."""
    value = mean(values)
    margin = 1.96 * stdev(values) / sqrt(len(values))
    return value, value - margin, value + margin


def graph_arrival_schedule():
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [0, 20, 40, 60]
    ns = [1200, 300, 1200, 1200]
    ew = [300, 1200, 300, 300]
    ax.step(x, ns, where="post", label="NS")
    ax.step(x, ew, where="post", label="EW")
    ax.set(xlabel="Модельное время, мин", ylabel="Интенсивность, авт./ч",
           title="Заданный входящий транспортный поток")
    ax.set_xlim(0, 60)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "arrival_schedule.png", dpi=160)
    plt.close(fig)


def graph_cohort_waits(results):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, group in zip(axes, GROUPS):
        for controller in CONTROLLERS:
            values = [mean(results[(controller, group, period)])
                      for period in range(3)]
            ax.plot(range(3), values, marker="o", label=LABELS[controller])
        ax.set(ylabel="Ожидание, с", title=f"Группа автомобилей: {group}")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xticks(range(3), PERIOD_LABELS)
    axes[-1].set_xlabel("Интервал прибытия, мин")
    fig.suptitle("Ожидание автомобилей по времени их прибытия")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cohort_waits.png", dpi=160)
    plt.close(fig)


def graph_queue_example(models):
    """Иллюстрация одного seed, не среднее по прогонам."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, controller in zip(axes, CONTROLLERS):
        history = models[controller].queue_history
        data = [(t, q) for t, q in history if t <= HORIZON]
        if data[-1][0] < HORIZON:
            data.append((HORIZON, data[-1][1]))
        x = [t / 60 for t, _ in data]
        ax.step(x, [q[0] + q[1] for _, q in data], where="post", label="NS")
        ax.step(x, [q[2] + q[3] for _, q in data], where="post", label="EW")
        for boundary in (20, 40):
            ax.axvline(boundary, linestyle="--", color="gray", alpha=0.6)
        ax.set(ylabel="Автомобилей", title=LABELS[controller])
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Модельное время, мин")
    axes[-1].set_xlim(0, 60)
    fig.suptitle("Очереди при смене интенсивностей: пример seed=0")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "queues_seed0.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_example = make_config(0)
    print("Средние интенсивности за час:", config_example.average_rates())
    print("Пропорциональный контроллер использует NS=36 с, EW=24 с.\n")

    results = defaultdict(list)
    examples = {}
    remaining = defaultdict(list)

    for seed in range(N_RUNS):
        cfg = make_config(seed)
        signatures = []
        for controller in CONTROLLERS:
            model = run(cfg, controller)
            if seed == 0:
                examples[controller] = model
            assert model.arrived == model.departed
            signatures.append(arrival_signature(model))
            for key, wait in wait_groups(model).items():
                results[(controller, *key)].append(wait)
            remaining[controller].append(sum(model.at_horizon["queue_lengths"].values()))
        assert signatures[0] == signatures[1] == signatures[2], (
            "Входящие потоки контроллеров не совпали"
        )

    rows = []
    for group in GROUPS:
        for period in range(3):
            for controller in CONTROLLERS:
                m, low, high = confidence_interval(results[(controller, group, period)])
                rows.append({"group": group, "period": PERIOD_LABELS[period],
                             "controller": controller, "mean_wait": m,
                             "ci_low": low, "ci_high": high})

    with (OUTPUT_DIR / "summary_by_period.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for controller in CONTROLLERS:
        print(f"{LABELS[controller]:20s} "
              f"ожидание={mean(results[(controller, 'ALL', 'total')]):6.2f} с, "
              f"остаток через час={mean(remaining[controller]):5.1f} авт.")

    for baseline in ("fixed", "fixed_proportional"):
        differences = [b - a for b, a in zip(
            results[(baseline, "ALL", "total")],
            results[("adaptive", "ALL", "total")],
        )]
        m, low, high = confidence_interval(differences)
        print(f"Выигрыш Adaptive над {LABELS[baseline]}: "
              f"{m:+.2f} с, 95% ДИ [{low:.2f}; {high:.2f}]")

    graph_arrival_schedule()
    graph_cohort_waits(results)
    graph_queue_example(examples)
    print(f"\nГрафики и CSV сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
