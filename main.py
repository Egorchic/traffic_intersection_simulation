"""Точка входа для одного запуска моделирования."""

import simpy

from traffic_sim.controllers import fixed_time_controller
from traffic_sim.model import Config, DIRECTIONS, Intersection

from traffic_sim.visualization import save_plots


def run(config: Config) -> Intersection:
    env = simpy.Environment()
    intersection = Intersection(env, config)

    # Сначала создаём монитор, потом остальные процессы.
    env.process(intersection.snapshot_at_horizon())
    env.process(fixed_time_controller(intersection, config))
    for direction in DIRECTIONS:
        env.process(intersection.generate(direction))
        env.process(intersection.serve(direction))

    # После часа поступлений продолжаем обслуживать накопленные очереди.
    env.run(until=intersection.done)
    return intersection


def main():
    intersection = run(
        Config(
            rates={
                "N": 600,
                "S": 600,
                "E": 150,
                "W": 150,
            }
        )
    )
    result = intersection.report()

    print("=== Фиксированный светофор ===")
    print(f"Прибыло: {result['arrived']}")
    print(f"Проехало после опустошения очередей: {result['departed']}")
    print(f"Среднее ожидание: {result['mean_wait']:.2f} с")
    print(f"Проехало за первый час: {result['horizon']['departed']}")
    print(f"Осталось в очередях через час: {result['horizon']['queue_lengths']}")
    print(f"Очереди полностью опустели к {result['cleared_at']:.1f} с")
    print("Средние длины очередей за первый час:")
    for direction, value in result["mean_queue_lengths"].items():
        print(f"  {direction}: {value:.2f} авт.")

    save_plots(intersection)


if __name__ == "__main__":
    main()

