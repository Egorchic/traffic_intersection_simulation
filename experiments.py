
"""Статистическое сравнение алгоритмов управления."""

from math import sqrt
from statistics import mean, stdev

from main import run
from traffic_sim.model import Config, DIRECTIONS


N_RUNS = 100
GROUPS = ("ALL", *DIRECTIONS)

RATES = {
    "N": 600,
    "S": 600,
    "E": 150,
    "W": 150,
}


def get_mean_waits(model):
    """Среднее ожидание по каждому направлению и в целом."""
    waits = {group: [] for group in GROUPS}

    for vehicle, departure in model.completed:
        wait = departure - vehicle.arrived_at

        waits["ALL"].append(wait)
        waits[vehicle.direction].append(wait)

    return {
        group: mean(values) if values else float("nan")
        for group, values in waits.items()
    }


def main():
    results = {
        group: {"fixed": [], "adaptive": []}
        for group in GROUPS
    }

    for seed in range(N_RUNS):
        config = Config(rates=RATES, seed=seed)

        fixed = run(config, "fixed")
        adaptive = run(config, "adaptive")

        # Проверяем, что в обоих запусках
        # поступило одинаковое число автомобилей.
        assert fixed.arrived == adaptive.arrived

        for controller, model in (
            ("fixed", fixed),
            ("adaptive", adaptive),
        ):
            waits = get_mean_waits(model)

            for group in GROUPS:
                results[group][controller].append(waits[group])

    print(f"Количество парных экспериментов: {N_RUNS}\n")

    for group in GROUPS:
        fixed = results[group]["fixed"]
        adaptive = results[group]["adaptive"]

        # Положительная разность означает выигрыш
        # адаптивного управления.
        differences = [
            f - a for f, a in zip(fixed, adaptive)
        ]

        avg_fixed = mean(fixed)
        avg_adaptive = mean(adaptive)
        avg_difference = mean(differences)

        # Приближённый 95%-й доверительный интервал
        # среднего парного различия.
        margin = (
            1.96 * stdev(differences) / sqrt(N_RUNS)
        )

        improvement = (
            100 * avg_difference / avg_fixed
            if avg_fixed > 0 else 0
        )

        print(f"=== {group} ===")
        print(f"Fixed:       {avg_fixed:.2f} с")
        print(f"Adaptive:    {avg_adaptive:.2f} с")
        print(f"Выигрыш:     {avg_difference:+.2f} с")
        print(f"Изменение:   {improvement:+.1f}%")
        print(
            f"95% ДИ:      "
            f"[{avg_difference - margin:.2f}; "
            f"{avg_difference + margin:.2f}]"
        )
        print()


if __name__ == "__main__":
    main()
