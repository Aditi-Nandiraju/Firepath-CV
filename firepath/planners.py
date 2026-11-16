"""Evacuation routing policies.

All policies share one routine: Dijkstra from every zone to the nearest exit,
where entering a zone costs  edge length (m) + penalty(zone).  They differ only
in how they turn camera confidences into that penalty:

  static        penalty 0 (ignores fire, sanity floor)
  threshold     BLOCK if this frame's confidence >= tau            (main baseline)
  latched       like threshold, but a zone stays blocked once alarmed
  voting        BLOCK if >= k of the last n frames have confidence >= tau
  confidence    lambda x confidence                                 (our method)
  confidence+smooth  lambda x EMA of confidence over frames          (our method)
  oracle        uses the TRUE fire state (ceiling, not a real method)

BLOCK is a large finite cost so a route always exists, even when every exit
looks unsafe.
"""
from __future__ import annotations

from collections import deque

import networkx as nx

from .building import EXITS
from .fire import FIRE, SMOKE

BLOCK = 10_000.0
SINK = "__SINK__"


def with_sink(g: nx.Graph) -> nx.Graph:
    h = g.copy()
    for e in EXITS:
        h.add_edge(e, SINK, length=0.0)
    return h


class Planner:
    name = "base"

    def __init__(self, graph_with_sink: nx.Graph):
        self.g = graph_with_sink
        self.next_hop: dict[str, str] = {}
        self.paths: dict[str, list[str]] = {}

    def update(self, obs: dict[str, float], truth: dict[str, int]) -> None:
        """Called once per time step with camera confidences and (oracle only) the truth."""

    def penalty(self, zone: str) -> float:
        return 0.0

    def plan(self) -> None:
        """Recompute the best route from every zone to the nearest exit."""
        def w(u, v, d):
            # Search runs backwards from the sink: expanding u means the walker ENTERS u.
            return d["length"] + (0.0 if u == SINK else self.penalty(u))

        _, paths = nx.single_source_dijkstra(self.g, SINK, weight=w)
        self.paths = {n: list(reversed(p))[:-1] for n, p in paths.items() if n != SINK}


class StaticPlanner(Planner):
    name = "static"


class ThresholdPlanner(Planner):
    def __init__(self, g, tau=0.5, latch=False):
        super().__init__(g)
        self.tau, self.latch = tau, latch
        self.name = f"{'latched' if latch else 'threshold'}@{tau:g}"
        self.blocked: set[str] = set()

    def update(self, obs, truth):
        now = {z for z, c in obs.items() if c >= self.tau}
        self.blocked = (self.blocked | now) if self.latch else now

    def penalty(self, zone):
        return BLOCK if zone in self.blocked else 0.0


class VotingPlanner(Planner):
    def __init__(self, g, tau=0.5, k=2, n=3):
        super().__init__(g)
        self.tau, self.k, self.n = tau, k, n
        self.name = f"voting{k}of{n}@{tau:g}"
        self.hist: dict[str, deque] = {}

    def update(self, obs, truth):
        for z, c in obs.items():
            self.hist.setdefault(z, deque(maxlen=self.n)).append(c >= self.tau)

    def penalty(self, zone):
        h = self.hist.get(zone)
        return BLOCK if h is not None and sum(h) >= self.k else 0.0


class ConfidencePlanner(Planner):
    """cost = length + lambda x belief, belief = EMA of calibrated confidence.

    alpha = 1.0 means no smoothing (belief = this frame's confidence).
    """

    def __init__(self, g, lam=40.0, alpha=1.0):
        super().__init__(g)
        self.lam, self.alpha = lam, alpha
        self.name = f"confidence@{lam:g}" + ("" if alpha >= 1.0 else f"+ema{alpha:g}")
        self.belief: dict[str, float] = {}

    def update(self, obs, truth):
        a = self.alpha
        for z, c in obs.items():
            self.belief[z] = a * c + (1 - a) * self.belief.get(z, 0.0)

    def penalty(self, zone):
        return self.lam * self.belief.get(zone, 0.0)


class OraclePlanner(Planner):
    """Perfect knowledge of the true hazard. Upper bound, not a real system."""

    def __init__(self, g, smoke_cost=40.0):
        super().__init__(g)
        self.name = "oracle"
        self.smoke_cost = smoke_cost
        self.truth: dict[str, int] = {}

    def update(self, obs, truth):
        self.truth = truth

    def penalty(self, zone):
        s = self.truth.get(zone, 0)
        return BLOCK if s == FIRE else (self.smoke_cost if s == SMOKE else 0.0)


def make_policy(spec: str, g: nx.Graph) -> Planner:
    """Build a planner from a short spec string, e.g. 'threshold:0.5', 'confidence:40:0.5'."""
    kind, *args = spec.split(":")
    a = [float(x) for x in args]
    if kind == "static":
        return StaticPlanner(g)
    if kind == "threshold":
        return ThresholdPlanner(g, tau=a[0] if a else 0.5)
    if kind == "latched":
        return ThresholdPlanner(g, tau=a[0] if a else 0.5, latch=True)
    if kind == "voting":
        tau, k, n = (a + [0.5, 2, 3][len(a):])[:3]
        return VotingPlanner(g, tau=tau, k=int(k), n=int(n))
    if kind == "confidence":
        lam, alpha = (a + [40.0, 1.0][len(a):])[:2]
        return ConfidencePlanner(g, lam=lam, alpha=alpha)
    if kind == "oracle":
        return OraclePlanner(g)
    raise ValueError(f"unknown policy spec: {spec}")
