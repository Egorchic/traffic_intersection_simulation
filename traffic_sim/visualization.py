
from pathlib import Path

import matplotlib.pyplot as plt

from .model import DIRECTIONS, Intersection


def save_plots(
    intersection: Intersection,
    output_dir: str = "results",
):
    """Сохраняет графики работы перекрёстка."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    horizon = intersection.config.arrival_duration

    # 1. Динамика очередей за весь период прибытия

    history = [
        (t, lengths)
        for t, lengths in intersection.queue_history
        if t <= horizon
    ]

    if history[-1][0] < horizon:
        history.append((horizon, history[-1][1]))

    times = [t / 60 for t, _ in history]

    fig, ax = plt.subplots(figsize=(12, 5))

    for index, direction in enumerate(DIRECTIONS):
        lengths = [q[index] for _, q in history]

        ax.step(
            times,
            lengths,
            where="post",
            label=direction,
            linewidth=1.3,
        )

    ax.set_title("Изменение длин очередей")
    ax.set_xlabel("Модельное время, мин")
    ax.set_ylabel("Количество автомобилей")
    ax.set_xlim(0, horizon / 60)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="Направление")

    fig.tight_layout()
    fig.savefig(output / "queues.png", dpi=160)
    plt.close(fig)

    # 2. Переключение светофорных фаз
    # Показываем первые пять минут для наглядности.

    window = min(300.0, horizon)

    phases = [
        (t, phase)
        for t, phase in intersection.phase_history
        if t <= window
    ]

    if phases[-1][0] < window:
        phases.append((window, phases[-1][1]))

    phase_values = {"NS": 1, "EW": -1, None: 0}

    fig, ax = plt.subplots(figsize=(12, 3.5))

    ax.step(
        [t for t, _ in phases],
        [phase_values[p] for _, p in phases],
        where="post",
        linewidth=2,
    )

    ax.set_title("Работа светофора (первые 5 минут)")
    ax.set_xlabel("Модельное время, с")
    ax.set_yticks(
        [-1, 0, 1],
        ["EW", "Переход", "NS"],
    )
    ax.set_xlim(0, window)
    ax.set_ylim(-1.4, 1.4)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output / "traffic_light.png", dpi=160)
    plt.close(fig)

    print(f"Графики сохранены в папку {output}/")
