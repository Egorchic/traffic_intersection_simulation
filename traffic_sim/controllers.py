"""Алгоритмы управления светофором."""

from .model import Config, Intersection


def fixed_time_controller(intersection: Intersection, config: Config):
    """Фиксированный цикл: NS -> жёлтый -> все красные -> EW -> ..."""
    env = intersection.env

    while True:
        for phase in ("NS", "EW"):
            intersection.set_green_phase(phase, until=env.now + config.green_duration)
            yield env.timeout(config.green_duration)

            intersection.set_green_phase(None)  # Жёлтый
            yield env.timeout(config.yellow_duration)

            intersection.set_green_phase(None)  # Все красные
            yield env.timeout(config.all_red_duration)


def adaptive_controller(
    intersection: Intersection,
    config: Config,
    min_green: float = 10.0,
    max_green: float = 50.0,
    queue_threshold: int = 3,
    check_interval: float = 1.0,
):
    """Адаптивное управление по длинам очередей."""
    env = intersection.env
    current_phase = "NS"

    while True:
        intersection.set_green_phase(current_phase)
        started_at = env.now
        other_phase = "EW" if current_phase == "NS" else "NS"

        while True:
            yield env.timeout(check_interval)
            elapsed = env.now - started_at
            if elapsed < min_green:
                continue

            current_queue = sum(len(intersection.queues[d]) for d in current_phase)
            other_queue = sum(len(intersection.queues[d]) for d in other_phase)

            if other_queue == 0:
                continue

            if (current_queue == 0
                    or other_queue > current_queue + queue_threshold
                    or elapsed >= max_green):
                break

        intersection.set_green_phase(None)
        yield env.timeout(config.yellow_duration)
        intersection.set_green_phase(None)
        yield env.timeout(config.all_red_duration)
        current_phase = other_phase


def proportional_fixed_controller(
    intersection: Intersection,
    config: Config,
    min_green: float = 10.0,
):
    """Постоянные фазы по средним интенсивностям за весь горизонт."""
    env = intersection.env

    # Для нестационарного потока НЕ используем rates первого периода.
    mean_rates = config.average_rates()
    ns_rate = mean_rates["N"] + mean_rates["S"]
    ew_rate = mean_rates["E"] + mean_rates["W"]
    total_rate = ns_rate + ew_rate

    total_green = 2 * config.green_duration
    if not 0 < min_green < total_green / 2:
        raise ValueError("Некорректное минимальное время зелёного")

    if total_rate == 0:
        ns_green = total_green / 2
    else:
        ns_green = total_green * ns_rate / total_rate

    ns_green = max(min_green, min(ns_green, total_green - min_green))
    ew_green = total_green - ns_green

    while True:
        for phase, duration in (("NS", ns_green), ("EW", ew_green)):
            intersection.set_green_phase(phase, until=env.now + duration)
            yield env.timeout(duration)

            intersection.set_green_phase(None)
            yield env.timeout(config.yellow_duration)
            intersection.set_green_phase(None)
            yield env.timeout(config.all_red_duration)

