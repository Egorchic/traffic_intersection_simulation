"""Базовые проверки корректности модели (stdlib unittest)."""

import unittest

from main import run
from traffic_sim.model import Config, DIRECTIONS


class IntersectionTests(unittest.TestCase):
    def test_empty_traffic(self):
        model = run(Config(arrival_duration=120, rates={d: 0 for d in DIRECTIONS}))
        self.assertEqual(model.arrived, 0)
        self.assertEqual(model.departed, 0)
        self.assertEqual(model.env.now, 120)

    def test_all_vehicles_eventually_depart(self):
        model = run(Config(arrival_duration=300))
        self.assertEqual(model.arrived, model.departed)
        self.assertTrue(all(not q for q in model.queues.values()))
        self.assertTrue(all(departure >= vehicle.arrived_at
                            for vehicle, departure in model.completed))

    def test_seed_reproducibility(self):
        config = Config(arrival_duration=300, seed=123)
        model1 = run(config)
        model2 = run(config)
        self.assertEqual(model1.completed, model2.completed)

    def test_no_vehicles_appear_on_wrong_direction(self):
        rates = {d: 0 for d in DIRECTIONS}
        rates["N"] = 600
        model = run(Config(arrival_duration=300, rates=rates))
        self.assertTrue(model.arrived > 0)
        self.assertTrue(all(vehicle.direction == "N"
                            for vehicle, _ in model.completed))


if __name__ == "__main__":
    unittest.main()
