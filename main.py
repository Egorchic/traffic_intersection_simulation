"""Точка входа для одного запуска моделирования."""

import simpy

from traffic_sim.controllers import (
    fixed_time_controller,
    adaptive_controller,
)

from traffic_sim.model import Config, DIRECTIONS, Intersection

from traffic_sim.visualization import save_plots


def run(
    config: Config,
    controller: str = "fixed",
) -> Intersection:
    env = simpy.Environment()
    intersection = Intersection(env, config)

    if controller == "fixed":
        controller_function = fixed_time_controller
    elif controller == "adaptive":
        controller_function = adaptive_controller
    else:
        raise ValueError(f"Неизвестный контроллер: {controller}")

    env.process(intersection.snapshot_at_horizon())
    env.process(controller_function(intersection, config))

    for direction in DIRECTIONS:
        env.process(intersection.generate(direction))
        env.process(intersection.serve(direction))

    env.run(until=intersection.done)
    return intersection



def main():
    config = Config(
        rates={
            "N": 600,
            "S": 600,
            "E": 150,
            "W": 150,
        },
        seed=42,
    )

    for controller in ("fixed", "adaptive"):
        intersection = run(config, controller)
        result = intersection.report()

        print(f"\n=== {controller.upper()} ===")
        print(f"Прибыло: {result['arrived']}")
        print(f"Проехало: {result['departed']}")
        print(f"Среднее ожидание: {result['mean_wait']:.2f} с")
        print(
            "Осталось через час:",
            result["horizon"]["queue_lengths"],
        )
        print("Средние длины очередей:")

        for direction, value in result["mean_queue_lengths"].items():
            print(f"  {direction}: {value:.2f}")

        save_plots(
            intersection,
            output_dir=f"results/{controller}",
        )



if __name__ == "__main__":
    main()

