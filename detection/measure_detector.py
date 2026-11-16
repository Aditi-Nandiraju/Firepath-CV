"""Measure a trained detector's REAL errors on videos and write them for the simulator.

This is the step that makes the study non-circular: the simulator's misses,
false alarms, delays and confidences all come from here.

Input: a CSV listing videos (do NOT use videos the model was trained or tuned on)
    path,label,onset_s
    videos/fire_01.mp4,fire,0        # fire visible from 0 s
    videos/smoke_03.mp4,smoke,4.5    # smoke first visible at 4.5 s
    videos/office_07.mp4,none,       # normal scene, no fire
    videos/sunset_02.mp4,confuser,   # fire-LIKE scene (sunset, red light, orange clothes)

How a camera "step" works: the simulator advances in steps of --step-seconds.
Within each step we keep the highest box confidence over the sampled frames
(0 when no box). That is exactly what the simulator's cameras report.

Usage:
    python detection/measure_detector.py --weights runs/firepath/train/weights/best.pt \
        --videos videos.csv --calib-videos calib.csv --out configs/detector_errors_measured.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- inference
def video_step_confidences(model, path: str, step_s: float, sample_fps: float, det_conf: float, imgsz: int):
    """Return (step_start_times, max_confidence_per_step) for one video."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    every = max(1, int(round(fps / sample_fps)))
    times, confs, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            r = model.predict(frame, conf=det_conf, imgsz=imgsz, verbose=False)[0]
            c = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
            times.append(i / fps)
            confs.append(c)
        i += 1
    cap.release()
    if not times:
        return np.array([]), np.array([]), np.array([]), np.array([])
    times, confs = np.array(times), np.array(confs)
    n_steps = int(math.ceil((times[-1] + 1e-9) / step_s)) or 1
    step_t = np.arange(n_steps) * step_s
    step_c = np.array([confs[(times >= t) & (times < t + step_s)].max(initial=0.0) for t in step_t])
    return step_t, step_c, times, confs


# ----------------------------------------------------------------- statistics
def _logit(c):
    c = np.clip(c, 1e-4, 1 - 1e-4)
    return np.log(c / (1 - c))


def fit_platt(confs: np.ndarray, labels: np.ndarray, reg: float = 0.01):
    """Fit p = sigmoid(a * logit(conf) + b) on boxes only; returns (a, b).

    A small penalty pulls (a, b) toward (1, 0) = 'trust the raw score', so a
    handful of videos cannot produce an absurdly sharp calibration curve.
    """
    from scipy.optimize import minimize

    m = confs > 0
    x, y = _logit(confs[m]), labels[m].astype(float)
    if len(np.unique(y)) < 2:
        print("WARNING: calibration needs both true and false detections; using identity.")
        return 1.0, 0.0
    if len(y) < 50:
        print(f"WARNING: only {len(y)} detections to calibrate on; collect more videos for a reliable fit.")

    def loss(w):
        z = w[0] * x + w[1]
        return np.mean(np.logaddexp(0, z) - y * z) + reg * ((w[0] - 1) ** 2 + w[1] ** 2)

    w = minimize(loss, x0=[1.0, 0.0], method="Nelder-Mead").x
    return float(w[0]), float(w[1])


def calibrate(c: np.ndarray, a: float, b: float) -> np.ndarray:
    from scipy.special import expit

    out = np.zeros_like(c, dtype=float)
    m = c > 0
    out[m] = expit(a * _logit(c[m]) + b)
    return out


def fit_beta(x: np.ndarray, fallback):
    x = np.clip(np.asarray(x, dtype=float), 1e-3, 1 - 1e-3)
    if len(x) < 5:
        print(f"WARNING: only {len(x)} samples to fit a confidence distribution; using fallback {fallback}.")
        return list(fallback)
    m, v = x.mean(), x.var()
    v = max(v, 1e-4)
    k = m * (1 - m) / v - 1
    if k <= 0:
        return list(fallback)
    return [round(float(m * k), 3), round(float((1 - m) * k), 3)]


def ece(conf: np.ndarray, y: np.ndarray, bins=10):
    m = conf > 0
    conf, y = conf[m], y[m]
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - y[s].mean())
    return float(e)


def step_labels(step_t, label, onset):
    if label in ("fire", "smoke"):
        return (step_t + 1e-9 >= (onset or 0.0)).astype(int)
    return np.zeros(len(step_t), dtype=int)


def summarise(videos: list[dict], step_s: float, a: float, b: float, confuser_fraction: float):
    """videos: dicts with label, onset_s, step_t, step_c (raw). Returns (json_dict, per_video_rows)."""
    rows = []
    out = {}
    for hz in ("fire", "smoke"):
        vs = [v for v in videos if v["label"] == hz]
        delays, missed, recalls, tp = [], 0, [], []
        for v in vs:
            c = calibrate(v["step_c"], a, b)
            after = v["step_t"] + 1e-9 >= (v["onset_s"] or 0.0)
            idx = np.where(after & (c > 0))[0]
            if len(idx) == 0:
                missed += 1
                rows.append({"video": v["path"], "label": hz, "detected": False})
                continue
            first = idx[0]
            # delay measured at frame level when frame times are available, else at step level
            onset = v["onset_s"] or 0.0
            ft, fc = v.get("frame_t"), v.get("frame_c")
            hit = np.where((ft >= onset) & (fc > 0))[0] if ft is not None and len(ft) else []
            first_t = float(ft[hit[0]]) if len(hit) else float(v["step_t"][first])
            delay = max(0.0, first_t - onset)
            delays.append(delay)
            window = c[first:][after[first:]]
            recalls.append(float((window > 0).mean()))
            tp.extend(window[window > 0].tolist())
            rows.append({"video": v["path"], "label": hz, "detected": True, "delay_s": delay,
                         "step_recall": recalls[-1], "mean_conf": float(window[window > 0].mean())})
        n = len(vs)
        if n == 0:
            print(f"WARNING: no '{hz}' videos; {hz} statistics fall back to the other hazard type.")
            continue
        out[hz] = {
            "n_videos": n,
            "event_miss_rate": round(missed / n, 4),
            "delay_s": [round(d, 2) for d in delays] or [0.0],
            "frame_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "conf_beta": fit_beta(tp, (4.0, 2.0)),
        }
    if "fire" not in out and "smoke" in out:
        out["fire"] = dict(out["smoke"])
    if "smoke" not in out and "fire" in out:
        out["smoke"] = dict(out["fire"])

    fa = {}
    for kind in ("none", "confuser"):
        vs = [v for v in videos if v["label"] == kind]
        steps, hits, confs = 0, 0, []
        for v in vs:
            c = calibrate(v["step_c"], a, b)
            steps += len(c)
            hits += int((c > 0).sum())
            confs.extend(c[c > 0].tolist())
            rows.append({"video": v["path"], "label": kind, "false_alarm_steps": int((c > 0).sum()), "steps": len(c)})
        fa[kind] = (hits / steps if steps else float("nan"), confs, len(vs))
    rate_none, conf_none, n_none = fa["none"]
    rate_conf, conf_conf, n_conf = fa["confuser"]
    if n_none == 0:
        raise SystemExit("Need at least some 'none' (normal scene) videos to measure false alarms.")
    out["false_alarm"] = {
        "n_videos_normal": n_none,
        "n_videos_confuser": n_conf,
        "rate_per_step": round(rate_none, 5),
        "conf_beta": fit_beta(conf_none + conf_conf, (2.5, 4.0)),
        "confuser_zone_fraction": confuser_fraction,
        "confuser_rate_per_step": round(rate_conf, 5) if n_conf else round(rate_none, 5),
    }
    return out, rows


# ----------------------------------------------------------------- main
def read_list(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = r["label"].strip().lower()
        assert r["label"] in ("fire", "smoke", "none", "confuser"), f"bad label {r['label']}"
        r["onset_s"] = float(r["onset_s"]) if r.get("onset_s", "").strip() else 0.0
    return rows


def run_videos(model, rows, a):
    out = []
    for r in rows:
        t, c, ft, fc = video_step_confidences(model, r["path"], a.step_seconds, a.sample_fps, a.det_conf, a.imgsz)
        print(f"{r['label']:9s} {Path(r['path']).name:40s} steps={len(c):4d} detections={(c > 0).sum():4d}")
        out.append({**r, "step_t": t, "step_c": c, "frame_t": ft, "frame_c": fc})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--videos", required=True, help="CSV of evaluation videos")
    p.add_argument("--calib-videos", help="separate CSV for calibration (recommended)")
    p.add_argument("--step-seconds", type=float, default=5.0)
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--det-conf", type=float, default=0.10, help="lowest box confidence the detector reports")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--confuser-fraction", type=float, default=0.10,
                   help="ASSUMED share of building zones containing a fire-like object")
    p.add_argument("--out", default="configs/detector_errors_measured.json")
    a = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)
    evals = run_videos(model, read_list(a.videos), a)
    if a.calib_videos:
        calib = run_videos(model, read_list(a.calib_videos), a)
    else:
        print("WARNING: no --calib-videos; calibrating on the evaluation videos (optimistic).")
        calib = evals
    cc = np.concatenate([v["step_c"] for v in calib])
    yy = np.concatenate([step_labels(v["step_t"], v["label"], v["onset_s"]) for v in calib])
    pa, pb = fit_platt(cc, yy)
    print(f"Calibration (Platt): a={pa:.3f} b={pb:.3f}  ECE raw={ece(cc, yy):.3f} -> calibrated={ece(calibrate(cc, pa, pb), yy):.3f}")

    stats, rows = summarise(evals, a.step_seconds, pa, pb, a.confuser_fraction)
    result = {
        "source": f"Measured with {a.weights} on {a.videos} ({len(evals)} videos)",
        "is_placeholder": False,
        "step_seconds": a.step_seconds,
        "detector_threshold": a.det_conf,
        "calibration": {"method": "platt", "a": pa, "b": pb,
                        "fitted_on": a.calib_videos or a.videos},
        **stats,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    report = Path(a.out).with_suffix(".per_video.csv")
    keys = sorted({k for r in rows for k in r})
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {a.out} and {report}")
    for hz in ("fire", "smoke"):
        s = stats[hz]
        print(f"{hz:6s} miss rate={s['event_miss_rate']:.1%}  median delay={np.median(s['delay_s']):.1f}s  "
              f"step recall={s['frame_recall']:.1%}")
    fa = stats["false_alarm"]
    print(f"false alarms per step: normal={fa['rate_per_step']:.2%}  fire-like scenes={fa['confuser_rate_per_step']:.2%}")


if __name__ == "__main__":
    main()
