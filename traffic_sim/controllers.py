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
        # В отличие от фиксированного контроллера,
        # заранее неизвестно, когда закончится зелёный.
        intersection.set_green_phase(current_phase)
        started_at = env.now

        other_phase = "EW" if current_phase == "NS" else "NS"

        while True:
            yield env.timeout(check_interval)

            elapsed = env.now - started_at

            # Нельзя переключать фазу слишком рано.
            if elapsed < min_green:
                continue

            current_queue = sum(
                len(intersection.queues[d])
                for d in current_phase
            )
            other_queue = sum(
                len(intersection.queues[d])
                for d in other_phase
            )

            # Если на красном нет автомобилей,
            # переключение не требуется.
            if other_queue == 0:
                continue

            should_switch = (
                current_queue == 0
                or other_queue > current_queue + queue_threshold
                or elapsed >= max_green
            )

            if should_switch:
                break

        # Завершаем зелёный сигнал.
        intersection.set_green_phase(None)
        yield env.timeout(config.yellow_duration)

        # Защитный интервал.
        intersection.set_green_phase(None)
        yield env.timeout(config.all_red_duration)

        current_phase = other_phase
