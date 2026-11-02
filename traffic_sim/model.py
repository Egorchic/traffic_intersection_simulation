"""Автомобили, потоки, очереди и обслуживание перекрёстка."""

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
import random
from typing import Optional

import simpy


DIRECTIONS = ("N", "S", "E", "W")
DIRECTION_PHASE = {"N": "NS", "S": "NS", "E": "EW", "W": "EW"}


@dataclass(frozen=True)
class Config:
    # Интенсивности прибытия, автомобилей в час.
    rates: dict[str, float] = field(
        default_factory=lambda: {direction: 300.0 for direction in DIRECTIONS}
    )
    arrival_duration: float = 3600.0
    green_duration: float = 30.0
    yellow_duration: float = 3.0
    all_red_duration: float = 1.0
    first_vehicle_delay: float = 2.0
    headway_min: float = 1.8
    headway_max: float = 2.6
    seed: int = 42
    # Пары (начало периода в секундах, интенсивности авт./ч).
    # None сохраняет прежний стационарный поток из rates.
    rate_schedule: tuple[tuple[float, dict[str, float]], ...] | None = None

    def __post_init__(self):
        if set(self.rates) != set(DIRECTIONS) or any(v < 0 for v in self.rates.values()):
            raise ValueError("Нужны неотрицательные интенсивности для N, S, E и W")
        if any(value <= 0 or not isfinite(value) for value in (
            self.arrival_duration, self.green_duration, self.yellow_duration,
            self.all_red_duration, self.first_vehicle_delay, self.headway_min,
        )) or not isfinite(self.headway_max) or self.headway_max < self.headway_min:
            raise ValueError("Некорректные длительности в Config")

        if any(not isfinite(v) for v in self.rates.values()):
            raise ValueError("Интенсивности должны быть конечными числами")

        if self.rate_schedule is not None:
            if not self.rate_schedule or self.rate_schedule[0][0] != 0:
                raise ValueError("Расписание должно начинаться с t=0")

            last_start = -1.0
            for start, period_rates in self.rate_schedule:
                if not isfinite(start) or not last_start < start < self.arrival_duration:
                    raise ValueError("Начала периодов должны возрастать и быть меньше горизонта")
                if set(period_rates) != set(DIRECTIONS) or any(
                    not isfinite(v) or v < 0 for v in period_rates.values()
                ):
                    raise ValueError("Некорректные интенсивности в расписании")
                last_start = start

    def average_rates(self) -> dict[str, float]:
        """Средние интенсивности за весь час; используются в фиксированном плане."""
        if self.rate_schedule is None:
            return dict(self.rates)

        total = {direction: 0.0 for direction in DIRECTIONS}
        for index, (start, period_rates) in enumerate(self.rate_schedule):
            end = (self.rate_schedule[index + 1][0]
                   if index + 1 < len(self.rate_schedule)
                   else self.arrival_duration)
            for direction in DIRECTIONS:
                total[direction] += period_rates[direction] * (end - start)

        return {d: total[d] / self.arrival_duration for d in DIRECTIONS}


@dataclass(frozen=True)
class Vehicle:
    id: int
    direction: str
    arrived_at: float


class Intersection:
    def __init__(self, env: simpy.Environment, config: Config):
        self.env = env
        self.config = config
        self.queues = {direction: deque() for direction in DIRECTIONS}
        self.queue_history = [(0.0, (0, 0, 0, 0))]
        self.phase_history = [(0.0, None)]
        self.arrival_rng = {
            direction: random.Random(config.seed + index)
            for index, direction in enumerate(DIRECTIONS)
        }
        self.service_rng = {
            direction: random.Random(config.seed + 1000 + index)
            for index, direction in enumerate(DIRECTIONS)
        }

        self.green_phase: Optional[str] = None
        self.green_until: Optional[float] = None
        self.green_epoch = 0
        self.phase_changed = env.event()
        self.state_changed = env.event()

        self.arrived = 0
        self.departed = 0
        self.completed: list[tuple[Vehicle, float]] = []
        self.finished_generators = 0
        self.done = env.event()

        # Интегралы длин очередей на интервале поступления автомобилей.
        self.queue_area = {direction: 0.0 for direction in DIRECTIONS}
        self.last_area_update = 0.0
        self.at_horizon: Optional[dict] = None

    def _update_queue_area(self):
        now = min(self.env.now, self.config.arrival_duration)
        elapsed = now - self.last_area_update
        for direction in DIRECTIONS:
            self.queue_area[direction] += len(self.queues[direction]) * elapsed
        self.last_area_update = now

    def _notify_state(self):
        lengths = tuple(len(self.queues[d]) for d in DIRECTIONS)
        if lengths != self.queue_history[-1][1]:
            self.queue_history.append((self.env.now, lengths))

        previous = self.state_changed
        self.state_changed = self.env.event()
        previous.succeed()

    def set_green_phase(self, phase: Optional[str], until: Optional[float] = None):
        """phase=None означает жёлтый или общий красный сигнал."""
        if phase not in ("NS", "EW", None):
            raise ValueError("Неизвестная светофорная фаза")
        self.green_phase = phase
        self.green_until = until
        if phase != self.phase_history[-1][1]:
            self.phase_history.append((self.env.now, phase))
        if phase is not None:
            self.green_epoch += 1
        previous = self.phase_changed
        self.phase_changed = self.env.event()
        previous.succeed()
        self._notify_state()

    def generate(self, direction: str):
        """Пуассоновский поток с кусочно-постоянной интенсивностью."""
        schedule = self.config.rate_schedule
        if schedule is None:
            schedule = ((0.0, self.config.rates),)

        rng = self.arrival_rng[direction]
        deadline = self.config.arrival_duration

        for index, (start, period_rates) in enumerate(schedule):
            end = schedule[index + 1][0] if index + 1 < len(schedule) else deadline
            rate_per_second = period_rates[direction] / 3600.0

            if self.env.now < start:
                yield self.env.timeout(start - self.env.now)

            while self.env.now < end:
                if rate_per_second == 0:
                    yield self.env.timeout(end - self.env.now)
                    break

                interval = rng.expovariate(rate_per_second)
                if self.env.now + interval >= end:
                    # Следующее прибытие не попало в текущий период.
                    # Переходим к границе и используем новую интенсивность.
                    yield self.env.timeout(end - self.env.now)
                    break

                yield self.env.timeout(interval)
                self._update_queue_area()
                self.arrived += 1
                self.queues[direction].append(
                    Vehicle(self.arrived, direction, self.env.now)
                )
                self._notify_state()

        self.finished_generators += 1
        self._check_finished()

    def serve(self, direction: str):
        """Обслуживает FIFO-очередь, только когда её направлению горит зелёный."""
        required_phase = DIRECTION_PHASE[direction]
        rng = self.service_rng[direction]
        previous_epoch = -1
        departed_in_green = 0

        while True:
            if self.green_phase != required_phase or not self.queues[direction]:
                yield self.state_changed
                continue

            if previous_epoch != self.green_epoch:
                previous_epoch = self.green_epoch
                departed_in_green = 0

            if departed_in_green == 0:
                interval = self.config.first_vehicle_delay
            else:
                interval = rng.uniform(self.config.headway_min, self.config.headway_max)

            # Если времени зелёного не хватает, не начинаем проезд.
            if self.green_until is not None and self.env.now + interval >= self.green_until:
                yield self.phase_changed
                continue

            phase_change_event = self.phase_changed
            result = yield self.env.timeout(interval) | phase_change_event
            if phase_change_event in result or self.green_phase != required_phase:
                continue

            self._update_queue_area()
            vehicle = self.queues[direction].popleft()
            self.departed += 1
            self.completed.append((vehicle, self.env.now))
            departed_in_green += 1
            self._notify_state()
            self._check_finished()

    def snapshot_at_horizon(self):
        """Сохраняет состояние ровно в момент окончания прибытия автомобилей."""
        yield self.env.timeout(self.config.arrival_duration)
        self._update_queue_area()
        self.at_horizon = {
            "arrived": self.arrived,
            "departed": self.departed,
            "queue_lengths": {d: len(self.queues[d]) for d in DIRECTIONS},
        }

    def _check_finished(self):
        if (self.finished_generators == len(DIRECTIONS)
                and self.arrived == self.departed
                and not self.done.triggered):
            self.done.succeed()

    def report(self) -> dict:
        waits = [departure - vehicle.arrived_at for vehicle, departure in self.completed]
        return {
            "arrived": self.arrived,
            "departed": self.departed,
            "mean_wait": sum(waits) / len(waits) if waits else 0.0,
            "mean_queue_lengths": {
                direction: self.queue_area[direction] / self.config.arrival_duration
                for direction in DIRECTIONS
            },
            "horizon": self.at_horizon,
            "cleared_at": self.env.now,
        }

