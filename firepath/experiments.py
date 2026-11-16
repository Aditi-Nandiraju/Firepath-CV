"""Run every policy on many simulated fires and summarise the results.

Usage:
    python -m firepath.experiments --errors configs/detector_errors_placeholder.json --scenarios 300

Outputs (in --out, default results/):
    per_scenario.csv        one row per (scenario, policy)
    summary.csv / summary.md   mean and 95% bootstrap CI per policy
    tradeoff.png            smoke exposure vs detour, the paper's key figure
    error_types.csv/.md     ablation: false alarms only / misses only / delay only / all
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .building import build_building
from .detector_model import DetectorErrors
from .planners import make_policy, with_sink
from .simulate import SimConfig, make_scenario, run_policy

MAIN_POLICIES = [
    "static",
    "threshold:0.5",
    "threshold:0.25",
    "latched:0.5",
    "voting:0.5:2:3",
    "confidence:40",
    "confidence:40:0.5",
    "confidence:160:0.5",
    "oracle",
]
LAMBDAS = [0, 5, 10, 20, 40, 80, 160, 320]
TAUS = [0.25, 0.4, 0.5, 0.6, 0.75]
METRICS = ["success_rate", "fire_casualty_rate", "smoke_exposure_s", "detour_ratio", "evac_time_s", "route_flips"]


def run(errors: DetectorErrors, policies: list[str], n_scenarios: int, cfg: SimConfig, building_kw: dict, seed0=0):
    g = build_building(**building_kw)
    gs = with_sink(g)
    rows = []
    for k in range(n_scenarios):
        sc = make_scenario(g, errors, cfg, seed0 + k)
        for spec in policies:
            r = run_policy(g, sc, make_policy(spec, gs), cfg, errors.step_seconds)
            r["spec"] = spec
            rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_ci(x: np.ndarray, n=2000, rng=np.random.default_rng(0)):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarise(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    out = []
    for spec in order:
        d = df[df.spec == spec]
        row = {"policy": d.policy.iloc[0]}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(d[m].to_numpy(dtype=float))
            row[m] = mean
            row[m + "_ci"] = f"[{lo:.3g}, {hi:.3g}]"
        out.append(row)
    return pd.DataFrame(out)


def error_variant(e: DetectorErrors, kind: str) -> DetectorErrors:
    """Keep only one error type so its effect on routing can be isolated."""
    v = copy.deepcopy(e)
    perfect = lambda h: (setattr(h, "event_miss_rate", 0.0), setattr(h, "delay_s", [0.0]), setattr(h, "frame_recall", 1.0))
    if kind in ("false_alarms_only", "none"):
        perfect(v.fire); perfect(v.smoke)
    if kind in ("misses_only", "delay_only", "none"):
        v.false_alarm.rate_per_step = 0.0
        v.false_alarm.confuser_rate_per_step = 0.0
    if kind == "misses_only":
        for h in (v.fire, v.smoke):
            h.delay_s = [0.0]
    if kind == "delay_only":
        for h in (v.fire, v.smoke):
            h.event_miss_rate, h.frame_recall = 0.0, 1.0
    return v


def plot_tradeoff(df: pd.DataFrame, path: Path, title_note: str):
    agg = df.groupby("spec")[["smoke_exposure_s", "detour_ratio", "fire_casualty_rate"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, ycol, ylab in [
        (axes[0], "smoke_exposure_s", "Mean smoke exposure per person (s)"),
        (axes[1], "fire_casualty_rate", "Share of people reaching a fire zone"),
    ]:
        for alpha, style, lab in [(1.0, "o-", "confidence (no smoothing)"), (0.5, "s-", "confidence + EMA 0.5")]:
            specs = [f"confidence:{l}" if alpha == 1.0 else f"confidence:{l}:{alpha}" for l in LAMBDAS]
            pts = agg.loc[[s for s in specs if s in agg.index]]
            ax.plot(pts.detour_ratio, pts[ycol], style, label=lab)
            if alpha < 1.0:  # label one curve only, and only a few λ values, so labels stay readable
                for s, r in pts.iterrows():
                    if float(s.split(":")[1]) in (10, 40, 160, 320):
                        ax.annotate(f"λ={s.split(':')[1]}", (r.detour_ratio, r[ycol]), fontsize=7,
                                    xytext=(4, -10), textcoords="offset points")
        th = agg.loc[[f"threshold:{t}" for t in TAUS if f"threshold:{t}" in agg.index]]
        ax.plot(th.detour_ratio, th[ycol], "^--", color="grey", label="threshold (τ sweep)")
        for name, mk in [("static", "x"), ("oracle", "*"), ("voting:0.5:2:3", "D"), ("latched:0.5", "v")]:
            if name in agg.index:
                ax.plot(agg.loc[name, "detour_ratio"], agg.loc[name, ycol], mk, ms=10, label=name)
        ax.set_xlabel("Mean detour ratio (walked / shortest)  →  longer routes")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Safety vs detour trade-off" + title_note, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--errors", default="configs/detector_errors_placeholder.json")
    p.add_argument("--scenarios", type=int, default=300)
    p.add_argument("--out", default="results")
    p.add_argument("--floors", type=int, default=5)
    p.add_argument("--corridor", type=int, default=8)
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--seed0", type=int, default=0,
                   help="first scenario seed. Pick λ on validation seeds (e.g. --seed0 100000), "
                        "then report test results on different seeds (default 0)")
    a = p.parse_args(argv)

    errors = DetectorErrors.load(a.errors)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    note = "  (PLACEHOLDER detector errors - not real results)" if errors.is_placeholder else ""
    if errors.is_placeholder:
        print("WARNING: using PLACEHOLDER detector errors. Do not report these numbers.")

    cfg = SimConfig()
    bkw = dict(n_floors=a.floors, n_corridor=a.corridor)
    sweep = sorted(set(MAIN_POLICIES
                       + [f"confidence:{l}" for l in LAMBDAS]
                       + [f"confidence:{l}:0.5" for l in LAMBDAS]
                       + [f"threshold:{t}" for t in TAUS]))
    df = run(errors, sweep, a.scenarios, cfg, bkw, seed0=a.seed0)
    df.to_csv(out / "per_scenario.csv", index=False)

    summ = summarise(df, MAIN_POLICIES)
    summ.to_csv(out / "summary.csv", index=False)
    cols = ["policy"] + [c for m in METRICS for c in (m, m + "_ci")]
    (out / "summary.md").write_text(
        f"Detector errors: {errors.source}\nScenarios: {a.scenarios}\n\n"
        + summ[cols].to_markdown(index=False, floatfmt=".3f"))
    plot_tradeoff(df, out / "tradeoff.png", note)
    print(summ[["policy"] + METRICS].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not a.skip_ablation:
        abl = []
        for kind in ["none", "false_alarms_only", "misses_only", "delay_only", "all"]:
            ev = errors if kind == "all" else error_variant(errors, kind)
            d = run(ev, MAIN_POLICIES, max(50, a.scenarios // 3), cfg, bkw, seed0=a.seed0)
            s = summarise(d, MAIN_POLICIES)[["policy", "smoke_exposure_s", "fire_casualty_rate", "detour_ratio", "route_flips"]]
            s.insert(0, "errors", kind)
            abl.append(s)
        abl = pd.concat(abl)
        abl.to_csv(out / "error_types.csv", index=False)
        (out / "error_types.md").write_text(abl.to_markdown(index=False, floatfmt=".3f"))
        print("\nAblation by error type written to", out / "error_types.md")


if __name__ == "__main__":
    main()
