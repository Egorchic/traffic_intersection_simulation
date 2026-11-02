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
