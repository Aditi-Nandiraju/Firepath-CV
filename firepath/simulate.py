"""Evacuation simulation: fire spreads, cameras report, occupants walk and re-plan.

Common random numbers: for a given seed the fire, the camera reports and the
occupants' start rooms are generated ONCE and replayed for every policy, so
policies are compared on exactly the same emergencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .building import EXITS, rooms, zones
from .detector_model import CameraNetwork, DetectorErrors
from .fire import CLEAR, FIRE, SMOKE, FireParams, FireState
from .planners import Planner, with_sink


@dataclass
class SimConfig:
    n_occupants: int = 30
    walk_speed: float = 1.2       # m/s, a common design walking speed
    max_steps: int = 120          # 120 x 5 s = 10 minutes
    preburn_steps: tuple[int, int] = (4, 10)  # fire grows 20-50 s before the evacuation starts
    fire: FireParams = None

    def __post_init__(self):
        if self.fire is None:
            self.fire = FireParams()


@dataclass
class Scenario:
    seed: int
    origin: str
    truth: list[dict[str, int]]      # true zone states per step (after pre-burn)
    onset: dict[str, int]            # step each zone became hazardous (pre-burn shifted, may be negative)
    obs: list[dict[str, float]]      # camera confidences per step
    starts: list[str]                # occupant start rooms
    shortest: dict[str, float]       # hazard-blind shortest distance from each start room


def make_scenario(g: nx.Graph, errors: DetectorErrors, cfg: SimConfig, seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    zs = zones(g)
    origin = str(rng.choice(zs))
    fire = FireState.ignite(g, origin, cfg.fire, rng)
    pre = int(rng.integers(cfg.preburn_steps[0], cfg.preburn_steps[1] + 1))
    cams = CameraNetwork(zs, errors, rng)

    truth, obs = [], []
    for t in range(-pre, cfg.max_steps):
        st = {z: fire.hazard(z) for z in zs}
        o = {z: cams.observe(z, st[z], fire.t, fire.onset.get(z)) for z in zs}
        if t >= 0:
            truth.append(st)
            obs.append(o)
        fire.step()
    onset = {z: s - pre for z, s in fire.onset.items()}

    safe_rooms = [r for r in rooms(g) if truth[0][r] == CLEAR]
    starts = [str(x) for x in rng.choice(safe_rooms, size=cfg.n_occupants, replace=True)]
    gs = with_sink(g)
    base = nx.single_source_dijkstra_path_length(gs, "__SINK__", weight="length")
    return Scenario(seed, origin, truth, onset, obs, starts, {r: base[r] for r in set(starts)})


def run_policy(g: nx.Graph, sc: Scenario, planner: Planner, cfg: SimConfig, step_seconds: float) -> dict:
    step_m = cfg.walk_speed * step_seconds
    n = len(sc.starts)
    at = list(sc.starts)              # zone the occupant is in (or leaving)
    nxt = [None] * n                  # zone they are walking toward
    left = [0.0] * n                  # metres left on the current edge
    edge_len = [0.0] * n
    plan = [None] * n                 # remaining planned path
    done = [False] * n
    dead = [False] * n
    t_out = [math.nan] * n
    exposure = [0.0] * n              # seconds spent in smoke
    walked = [0.0] * n
    flips = [0] * n

    for t in range(cfg.max_steps):
        truth = sc.truth[t]
        planner.update(sc.obs[t], truth)
        planner.plan()
        for i in range(n):
            if done[i] or dead[i]:
                continue
            if nxt[i] is None:  # standing on a node: choose the next hop
                new = planner.paths[at[i]]
                if plan[i] is not None and new != plan[i]:
                    flips[i] += 1
                plan[i] = new
                nxt[i] = new[1]
                edge_len[i] = left[i] = g.edges[at[i], nxt[i]]["length"]
            move = min(step_m, left[i])
            left[i] -= move
            walked[i] += move
            # which zone is the walker in? the one they left for the first half of the edge
            here = at[i] if left[i] > edge_len[i] / 2 else nxt[i]
            s = truth.get(here, CLEAR)
            if s == FIRE:
                dead[i] = True
                continue
            if s == SMOKE:
                exposure[i] += step_seconds
            if left[i] <= 1e-9:
                at[i], nxt[i] = nxt[i], None
                plan[i] = plan[i][1:]
                if at[i] in EXITS:
                    done[i] = True
                    t_out[i] = (t + 1) * step_seconds

    detour = [walked[i] / sc.shortest[sc.starts[i]] for i in range(n) if done[i]]
    return {
        "seed": sc.seed,
        "policy": planner.name,
        "success_rate": float(np.mean(done)),
        "fire_casualty_rate": float(np.mean(dead)),
        "trapped_rate": float(np.mean([not d and not x for d, x in zip(done, dead)])),
        "smoke_exposure_s": float(np.mean(exposure)),
        "evac_time_s": float(np.nanmean(t_out)) if any(done) else math.nan,
        "detour_ratio": float(np.mean(detour)) if detour else math.nan,
        "route_flips": float(np.mean(flips)),
    }
