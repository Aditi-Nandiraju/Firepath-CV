"""Multi-floor office building as a graph.

Every node is a zone watched by one camera. Edges are walkable connections
with a length in metres. Two staircases (west and east) join the floors, and
two exits sit on the ground floor.

Layout of one floor (n_corridor = 6):

    R0   R1   R2   R3   R4   R5      <- rooms
    |    |    |    |    |    |
    C0 - C1 - C2 - C3 - C4 - C5      <- corridor
    |                        |
    SW                       SE      <- stair landings (connect floors)
"""
from __future__ import annotations

import networkx as nx

EXITS = ("EXIT_W", "EXIT_E")


def node_floor(node: str) -> int | None:
    """Floor number of a node, or None for exits."""
    if node.startswith("F"):
        return int(node[1:node.index("_")])
    return None


def node_kind(node: str) -> str:
    if node in EXITS:
        return "exit"
    part = node.split("_", 1)[1]
    if part.startswith("R"):
        return "room"
    if part.startswith("C"):
        return "corridor"
    return "stair"


def build_building(
    n_floors: int = 5,
    n_corridor: int = 8,
    corridor_len: float = 10.0,
    room_len: float = 6.0,
    stair_len: float = 15.0,
) -> nx.Graph:
    """Return an undirected graph of the building with 'length' on each edge."""
    g = nx.Graph()
    for f in range(n_floors):
        for i in range(n_corridor):
            g.add_node(f"F{f}_C{i}", floor=f, kind="corridor", pos=(i, 0))
            g.add_node(f"F{f}_R{i}", floor=f, kind="room", pos=(i, 1))
            g.add_edge(f"F{f}_C{i}", f"F{f}_R{i}", length=room_len)
            if i > 0:
                g.add_edge(f"F{f}_C{i-1}", f"F{f}_C{i}", length=corridor_len)
        g.add_node(f"F{f}_SW", floor=f, kind="stair", pos=(0, -1))
        g.add_node(f"F{f}_SE", floor=f, kind="stair", pos=(n_corridor - 1, -1))
        g.add_edge(f"F{f}_SW", f"F{f}_C0", length=4.0)
        g.add_edge(f"F{f}_SE", f"F{f}_C{n_corridor-1}", length=4.0)
        if f > 0:
            g.add_edge(f"F{f}_SW", f"F{f-1}_SW", length=stair_len, stairs=True)
            g.add_edge(f"F{f}_SE", f"F{f-1}_SE", length=stair_len, stairs=True)
    g.add_node("EXIT_W", floor=0, kind="exit", pos=(-1, -1))
    g.add_node("EXIT_E", floor=0, kind="exit", pos=(n_corridor, -1))
    g.add_edge("EXIT_W", "F0_SW", length=5.0)
    g.add_edge("EXIT_E", "F0_SE", length=5.0)
    return g


def zones(g: nx.Graph) -> list[str]:
    """All camera zones (every node except the exits)."""
    return [n for n in g.nodes if n not in EXITS]


def rooms(g: nx.Graph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d["kind"] == "room"]
