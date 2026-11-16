"""Camera detector error model used inside the simulator.

The simulator never runs YOLO itself. Instead each camera produces what a
real detector would: sometimes nothing, sometimes a box with a confidence.
How often it misses, how late it fires, how often it raises false alarms and
what its confidences look like all come from a JSON file written by
detection/measure_detector.py from REAL videos. That is what keeps the
evaluation from being circular.

`configs/detector_errors_placeholder.json` holds made-up numbers so the code
runs before you have measured anything. Never report results from it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fire import CLEAR, FIRE, SMOKE


@dataclass
class HazardErrors:
    event_miss_rate: float     # fraction of fire/smoke events never detected at all
    delay_s: list[float]       # measured seconds from onset to first detection
    frame_recall: float        # after first detection, chance each sampled frame detects it
    conf_beta: tuple[float, float]  # Beta(a, b) fitted to calibrated true-positive confidences


@dataclass
class FalseAlarmErrors:
    rate_per_step: float       # chance a clear zone produces a false detection per sampled frame
    conf_beta: tuple[float, float]
    confuser_zone_fraction: float  # share of zones containing a fire-like object (window at sunset, red light)
    confuser_rate_per_step: float  # false-alarm rate in those zones


@dataclass
class DetectorErrors:
    source: str
    step_seconds: float
    fire: HazardErrors
    smoke: HazardErrors
    false_alarm: FalseAlarmErrors
    is_placeholder: bool

    @classmethod
    def load(cls, path: str | Path) -> "DetectorErrors":
        d = json.loads(Path(path).read_text())

        def hz(x):
            return HazardErrors(x["event_miss_rate"], list(x["delay_s"]), x["frame_recall"], tuple(x["conf_beta"]))

        fa = d["false_alarm"]
        return cls(
            source=d["source"],
            step_seconds=float(d["step_seconds"]),
            fire=hz(d["fire"]),
            smoke=hz(d["smoke"]),
            false_alarm=FalseAlarmErrors(
                fa["rate_per_step"], tuple(fa["conf_beta"]),
                fa["confuser_zone_fraction"], fa["confuser_rate_per_step"],
            ),
            is_placeholder=bool(d.get("is_placeholder", False)),
        )


class CameraNetwork:
    """One simulated camera per zone, driven by measured error statistics."""

    def __init__(self, zones: list[str], errors: DetectorErrors, rng: np.random.Generator):
        self.e = errors
        self.rng = rng
        n_conf = int(round(errors.false_alarm.confuser_zone_fraction * len(zones)))
        self.confusers = set(rng.choice(zones, size=n_conf, replace=False)) if n_conf else set()
        self._event: dict[str, tuple[bool, int]] = {}  # zone -> (missed?, delay in steps)

    def _hz(self, s: int) -> HazardErrors:
        return self.e.fire if s == FIRE else self.e.smoke

    def observe(self, zone: str, true_state: int, t: int, onset: int | None) -> float:
        """Confidence of the strongest fire/smoke box this step, or 0.0 for no box."""
        if true_state == CLEAR:
            fa = self.e.false_alarm
            rate = fa.confuser_rate_per_step if zone in self.confusers else fa.rate_per_step
            if self.rng.random() < rate:
                return float(self.rng.beta(*fa.conf_beta))
            return 0.0

        hz = self._hz(true_state)
        key = f"{zone}:{true_state}"  # a zone that turns from smoke to fire gets a new event
        if key not in self._event:
            missed = self.rng.random() < hz.event_miss_rate
            delay = float(self.rng.choice(hz.delay_s)) if hz.delay_s else 0.0
            self._event[key] = (missed, int(np.ceil(delay / self.e.step_seconds)))
        missed, delay_steps = self._event[key]
        if missed or onset is None or t - onset < delay_steps:
            return 0.0
        if self.rng.random() < hz.frame_recall:
            return float(self.rng.beta(*hz.conf_beta))
        return 0.0
