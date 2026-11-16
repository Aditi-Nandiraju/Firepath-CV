"""Draw the building, the fire, what the cameras report, and each planner's route.

Usage:
    python -m firepath.visualize --seed 7 --step 6 --room F3_R2 --out results/route_example.png
    python -m firepath.visualize --find-interesting     # searches for a case where planners disagree
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .building import EXITS, build_building, rooms
from .detector_model import DetectorErrors
from .fire import FIRE, SMOKE
from .planners import make_policy, with_sink
from .simulate import SimConfig, make_scenario

DEFAULT_POLICIES = ["static", "threshold:0.5", "confidence:160:0.5", "oracle"]
COLORS = {"clear": "#e8e8e8", "smoke": "#9e9e9e", "fire": "#e4572e"}


def layout(g):
    pos = {}
    for n, d in g.nodes(data=True):
        x, y = d["pos"]
        pos[n] = (x, d["floor"] * 3.2 + y)
    return pos


def routes_at(g, sc, specs, step):
    gs = with_sink(g)
    out = {}
    for spec in specs:
        pl = make_policy(spec, gs)
        for t in range(step + 1):
            pl.update(sc.obs[t], sc.truth[t])
        pl.plan()
        out[spec] = pl
    return out


def draw(g, sc, specs, step, room, path_out, note=""):
    pos = layout(g)
    planners = routes_at(g, sc, specs, step)
    truth, obs = sc.truth[step], sc.obs[step]
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 7.5), sharey=True)
    for ax, spec in zip(axes, specs):
        for u, v, d in g.edges(data=True):
            (x1, y1), (x2, y2) = pos[u], pos[v]
            ax.plot([x1, x2], [y1, y2], color="#cfcfcf", lw=1.5 if d.get("stairs") else 1, zorder=1)
        for n in g.nodes:
            x, y = pos[n]
            if n in EXITS:
                ax.scatter(x, y, s=160, marker="s", color="#2e7d32", zorder=3)
                continue
            s = truth[n]
            c = COLORS["fire"] if s == FIRE else COLORS["smoke"] if s == SMOKE else COLORS["clear"]
            ax.scatter(x, y, s=110, color=c, edgecolor="none", zorder=2)
            if obs[n] > 0:  # camera reported something this step
                ax.scatter(x, y, s=260, facecolor="none", edgecolor="#1565c0", lw=0.8 + 2.5 * obs[n], zorder=4)
        path = planners[spec].paths[room]
        xs, ys = zip(*[pos[n] for n in path])
        ax.plot(xs, ys, color="#1565c0" if spec != "oracle" else "#2e7d32", lw=3, alpha=0.85, zorder=5)
        ax.scatter(*pos[room], s=220, marker="*", color="black", zorder=6)
        length = sum(g.edges[a, b]["length"] for a, b in zip(path, path[1:]))
        ax.set_title(f"{planners[spec].name}\nroute {length:.0f} m via {path[-1]}", fontsize=10)
        ax.set_yticks([f * 3.2 for f in range(max(d['floor'] for _, d in g.nodes(data=True)) + 1)])
        ax.set_yticklabels([f"Floor {f}" for f in range(len(ax.get_yticks()))])
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    legend = [
        Line2D([], [], marker="o", ls="", color=COLORS["fire"], label="fire (true)"),
        Line2D([], [], marker="o", ls="", color=COLORS["smoke"], label="smoke (true)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#1565c0", mew=2, label="camera detection (ring width = confidence)"),
        Line2D([], [], marker="s", ls="", color="#2e7d32", label="exit"),
        Line2D([], [], marker="*", ls="", color="black", label="person"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Same fire, different planners - scenario {sc.seed}, t = {step * 5} s{note}", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_out, dpi=150)
    plt.close(fig)


def find_interesting(g, errors, cfg, specs, n=200):
    """First (seed, step, room) where the static route walks into fire and the planners disagree."""
    for seed in range(n):
        sc = make_scenario(g, errors, cfg, seed)
        for step in range(2, 12):
            pls = routes_at(g, sc, specs, step)
            for room in rooms(g):
                paths = {s: tuple(p.paths[room]) for s, p in pls.items()}
                fire_on = lambda p: any(sc.truth[step][z] == FIRE for z in p[:-1])
                escapable = "oracle" not in paths or not fire_on(paths["oracle"])
                if fire_on(paths["static"]) and escapable and len(set(paths.values())) >= 3:
                    return sc, step, room
    raise RuntimeError("no interesting case found")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--errors", default="configs/detector_errors_placeholder.json")
    p.add_argument("--seed", type=int)
    p.add_argument("--step", type=int, default=6)
    p.add_argument("--room")
    p.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    p.add_argument("--find-interesting", action="store_true")
    p.add_argument("--out", default="results/route_example.png")
    a = p.parse_args(argv)
    g = build_building()
    errors = DetectorErrors.load(a.errors)
    cfg = SimConfig()
    note = "  (placeholder detector errors)" if errors.is_placeholder else ""
    if a.find_interesting or a.seed is None or a.room is None:
        sc, step, room = find_interesting(g, errors, cfg, a.policies)
    else:
        sc, step, room = make_scenario(g, errors, cfg, a.seed), a.step, a.room
    draw(g, sc, a.policies, step, room, a.out, note)
    print(f"saved {a.out}  (seed={sc.seed}, step={step}, room={room})")


if __name__ == "__main__":
    main()
