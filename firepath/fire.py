"""Simple fire and smoke spread on the building graph.

Each zone is CLEAR (0), SMOKE (1) or FIRE (2). Every time step:
  * a FIRE zone sets each neighbour on fire with probability p_fire
  * a FIRE or SMOKE zone fills each clear neighbour with smoke with probability
    p_smoke, and smoke climbs stairs faster (p_smoke_up)
This is a deliberately simple model (state it as a limitation in the paper);
it is not a physics-based fire simulator like FDS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .building import EXITS, node_floor

CLEAR, SMOKE, FIRE = 0, 1, 2


@dataclass
class FireParams:
    p_fire: float = 0.10       # chance per step that fire jumps to a neighbour
    p_smoke: float = 0.30      # chance per step that smoke enters a neighbour
    p_smoke_up: float = 0.55   # smoke moving UP a staircase
    smoke_to_fire_steps: int = 0  # unused hook for future refinement


@dataclass
class FireState:
    graph: nx.Graph
    params: FireParams
    rng: np.random.Generator
    state: dict[str, int] = field(default_factory=dict)
    onset: dict[str, int] = field(default_factory=dict)  # step a zone first became hazardous
    t: int = 0

    @classmethod
    def ignite(cls, graph, origin, params, rng):
        fs = cls(graph=graph, params=params, rng=rng)
        fs.state = {n: CLEAR for n in graph.nodes}
        fs.state[origin] = FIRE
        fs.onset[origin] = 0
        return fs

    def step(self) -> None:
        self.t += 1
        new = dict(self.state)
        p = self.params
        for n, s in self.state.items():
            if s == CLEAR:
                continue
            for m in self.graph.neighbors(n):
                if m in EXITS:
                    continue  # exits are outside, they never burn
                if s == FIRE and self.state[m] != FIRE and self.rng.random() < p.p_fire:
                    new[m] = FIRE
                elif self.state[m] == CLEAR and new[m] == CLEAR:
                    going_up = (node_floor(m) or 0) > (node_floor(n) or 0)
                    if self.rng.random() < (p.p_smoke_up if going_up else p.p_smoke):
                        new[m] = SMOKE
        for n, s in new.items():
            if s != CLEAR and n not in self.onset:
                self.onset[n] = self.t
        self.state = new

    def hazard(self, n: str) -> int:
        return self.state.get(n, CLEAR)
