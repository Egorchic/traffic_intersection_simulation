"""Проверки нестационарного потока и парности эксперимента."""

import unittest

from main import run
from traffic_sim.model import Config, DIRECTIONS


class NonstationaryTests(unittest.TestCase):
    def test_average_rates(self):
        schedule = (
            (0.0, {"N": 600, "S": 600, "E": 150, "W": 150}),
            (1200.0, {"N": 150, "S": 150, "E": 600, "W": 600}),
            (2400.0, {"N": 600, "S": 600, "E": 150, "W": 150}),
        )
        rates = Config(rate_schedule=schedule).average_rates()
        self.assertEqual(rates, {"N": 450, "S": 450, "E": 300, "W": 300})

    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            Config(rate_schedule=((20.0, {d: 1 for d in DIRECTIONS}),))
        with self.assertRaises(ValueError):
            Config(arrival_duration=100.0, rate_schedule=(
                (0.0, {d: 1 for d in DIRECTIONS}),
                (120.0, {d: 1 for d in DIRECTIONS}),
            ))

    def test_no_arrivals_in_zero_intensity_period(self):
        initial = {d: 0.0 for d in DIRECTIONS}
        busy = {d: 0.0 for d in DIRECTIONS}
        busy["N"] = 3600.0
        config = Config(
            arrival_duration=180.0,
            rate_schedule=((0.0, initial), (60.0, busy), (120.0, initial)),
            seed=123,
        )
        model = run(config, "adaptive")
        self.assertGreater(model.arrived, 0)
        self.assertTrue(all(60 <= vehicle.arrived_at < 120
                            for vehicle, _ in model.completed))
        self.assertEqual(model.arrived, model.departed)

    def test_controllers_get_identical_arrivals(self):
        high = {"N": 600.0, "S": 600.0, "E": 150.0, "W": 150.0}
        reversed_ = {"N": 150.0, "S": 150.0, "E": 600.0, "W": 600.0}
        cfg = Config(
            arrival_duration=180.0,
            rate_schedule=((0.0, high), (60.0, reversed_), (120.0, high)),
            seed=77,
        )
        signatures = []
        for controller in ("fixed", "fixed_proportional", "adaptive"):
            model = run(cfg, controller)
            signatures.append(sorted((v.direction, v.arrived_at)
                                     for v, _ in model.completed))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])


if __name__ == "__main__":
    unittest.main()
