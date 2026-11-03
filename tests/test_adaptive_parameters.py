"""Проверки передачи параметров адаптивного управления."""

import unittest

from main import run
from traffic_sim.model import Config


class AdaptiveParameterTests(unittest.TestCase):
    def test_default_kwargs_do_not_change_simulation(self):
        cfg = Config(arrival_duration=300, seed=7)
        a = run(cfg, "adaptive")
        b = run(cfg, "adaptive", controller_kwargs={
            "min_green": 10.0,
            "max_green": 50.0,
            "queue_threshold": 3,
        })
        self.assertEqual(a.completed, b.completed)

    def test_different_parameters_preserve_arrival_stream(self):
        cfg = Config(arrival_duration=300, seed=31)
        a = run(cfg, "adaptive", controller_kwargs={"queue_threshold": 0})
        b = run(cfg, "adaptive", controller_kwargs={"queue_threshold": 8})
        signature = lambda model: sorted(
            (vehicle.direction, vehicle.arrived_at)
            for vehicle, _ in model.completed
        )
        self.assertEqual(signature(a), signature(b))
        self.assertEqual(a.arrived, a.departed)
        self.assertEqual(b.arrived, b.departed)

    def test_nonadaptive_rejects_adaptive_kwargs(self):
        with self.assertRaises(ValueError):
            run(Config(arrival_duration=100), "fixed",
                controller_kwargs={"queue_threshold": 3})


if __name__ == "__main__":
    unittest.main()
