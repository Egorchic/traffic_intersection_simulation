"""Проверки инвариантов имитационной модели.

Скопировать в tests/test_model_invariants.py.
Запуск из корня проекта:
    python -m unittest discover -s tests -v

Использует только стандартную библиотеку и существующий код проекта.
"""

from bisect import bisect_right
from collections import defaultdict
import unittest

from main import run
from traffic_sim.model import Config, DIRECTIONS, DIRECTION_PHASE


class ModelInvariantTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            arrival_duration=600,
            rates={"N": 600, "S": 600, "E": 150, "W": 150},
            seed=1234,
        )

    def test_vehicle_conservation_and_queue_emptying(self):
        """Все прибывшие автомобили обслужены, потерь и дублирования нет."""
        for controller in ("fixed", "fixed_proportional", "adaptive"):
            with self.subTest(controller=controller):
                model = run(self.config, controller)
                self.assertEqual(model.arrived, model.departed)
                self.assertEqual(model.arrived, len(model.completed))
                self.assertEqual(
                    sorted(vehicle.id for vehicle, _ in model.completed),
                    list(range(1, model.arrived + 1)),
                )
                self.assertTrue(all(len(q) == 0 for q in model.queues.values()))
                self.assertEqual(
                    model.at_horizon["arrived"],
                    model.at_horizon["departed"]
                    + sum(model.at_horizon["queue_lengths"].values()),
                )

    def test_fifo_on_every_approach(self):
        """На каждом въезде более поздний автомобиль не обгоняет ранний."""
        for controller in ("fixed", "fixed_proportional", "adaptive"):
            with self.subTest(controller=controller):
                model = run(self.config, controller)
                by_direction = defaultdict(list)
                for vehicle, departure in model.completed:
                    self.assertGreaterEqual(departure, vehicle.arrived_at)
                    by_direction[vehicle.direction].append((departure, vehicle))
                for direction, departures in by_direction.items():
                    arrivals_in_departure_order = [
                        vehicle.arrived_at
                        for _, vehicle in sorted(departures)
                    ]
                    self.assertEqual(
                        arrivals_in_departure_order,
                        sorted(arrivals_in_departure_order),
                        f"Нарушен FIFO на {direction} для {controller}",
                    )

    def test_departures_only_during_green_phase(self):
        """В момент проезда разрешена соответствующая фаза."""
        for controller in ("fixed", "fixed_proportional", "adaptive"):
            with self.subTest(controller=controller):
                model = run(self.config, controller)
                history = model.phase_history
                times = [time for time, _ in history]
                self.assertEqual(times, sorted(times))
                for vehicle, departure in model.completed:
                    idx = bisect_right(times, departure) - 1
                    phase = history[idx][1]
                    self.assertEqual(
                        phase,
                        DIRECTION_PHASE[vehicle.direction],
                        f"Проезд {vehicle.direction} в t={departure:.6f} "
                        f"при фазе {phase} ({controller})",
                    )

    def test_four_second_transition_between_green_phases(self):
        """Между разными зелёными фазами не меньше 3+1 секунд."""
        gap = self.config.yellow_duration + self.config.all_red_duration
        for controller in ("fixed", "fixed_proportional", "adaptive"):
            with self.subTest(controller=controller):
                model = run(self.config, controller)
                history = model.phase_history
                for i in range(2, len(history)):
                    t, phase = history[i]
                    if phase not in ("NS", "EW"):
                        continue
                    previous_t, previous_phase = history[i - 1]
                    self.assertIsNone(previous_phase)
                    self.assertGreaterEqual(t - previous_t + 1e-9, gap)
                    self.assertIn(history[i - 2][1], ("NS", "EW"))
                    self.assertNotEqual(history[i - 2][1], phase)

    def test_adaptive_respects_minimum_green(self):
        """Завершённое зелёное не короче min_green=10 с по умолчанию."""
        model = run(self.config, "adaptive")
        history = model.phase_history
        for i in range(1, len(history)):
            t, phase = history[i]
            if phase is not None:
                continue
            start, prev_phase = history[i - 1]
            if prev_phase in ("NS", "EW"):
                self.assertGreaterEqual(t - start + 1e-9, 10.0)

    def test_queue_length_time_integral(self):
        """Средняя длина очереди совпадает с площадью под ступенчатым графиком."""
        for controller in ("fixed", "fixed_proportional", "adaptive"):
            with self.subTest(controller=controller):
                model = run(self.config, controller)
                horizon = self.config.arrival_duration
                data = [(t, q) for t, q in model.queue_history if t <= horizon]
                if data[-1][0] < horizon:
                    data.append((horizon, data[-1][1]))
                reported = model.report()["mean_queue_lengths"]
                for index, direction in enumerate(DIRECTIONS):
                    area = sum(
                        (next_t - t) * q[index]
                        for (t, q), (next_t, _) in zip(data, data[1:])
                    )
                    self.assertAlmostEqual(
                        area / horizon, reported[direction], places=8,
                        msg=f"Ошибка среднего Q[{direction}] ({controller})",
                    )

    def test_fixed_controller_phase_schedule(self):
        """Для фиксированного цикла проверяем первые переключения."""
        model = run(Config(arrival_duration=180, rates={d: 0 for d in DIRECTIONS}))
        first_six = model.phase_history[:6]
        expected = [
            (0.0, None), (0.0, "NS"), (30.0, None),
            (34.0, "EW"), (64.0, None), (68.0, "NS"),
        ]
        self.assertEqual(first_six, expected)


if __name__ == "__main__":
    unittest.main()
