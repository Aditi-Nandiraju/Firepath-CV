# FirePath AI

Fire and smoke detection from cameras, plus evacuation routing in a multi-floor
building, built to answer one research question:

> How do a real fire detector's mistakes (misses, false alarms, delays) affect
> evacuation routes, and does confidence-aware routing keep people safer than a
> normal threshold-based planner without making routes too long?

The project has two halves that connect through one JSON file:

```
 PART A - detection (Colab, GPU)                 PART B - simulation (any laptop)
 ──────────────────────────────                  ─────────────────────────────────
 D-Fire images ──► train_yolo.py ──► best.pt     building graph + fire spread
                                        │              │
 held-out videos ──► measure_detector.py ──► detector_errors_measured.json
                                                       │
                                     cameras behave like YOUR detector
                                                       │
                                  7 routing policies ──► experiments.py ──► tables + plots
```

**Important:** `configs/detector_errors_placeholder.json` contains made-up
numbers so that Part B runs on day 1. Every plot made with it says
"PLACEHOLDER". Do not report those numbers. Replace them with the file Part A
produces.

---

## Folder layout

```
firepath/            simulation package
  building.py        5-floor office as a graph (rooms, corridors, 2 staircases, 2 exits)
  fire.py            simple fire + smoke spread model (smoke climbs stairs faster)
  detector_model.py  simulated cameras driven by MEASURED detector errors
  planners.py        routing policies (static, threshold, latched, voting, confidence, oracle)
  simulate.py        occupants walk, re-plan every 5 s, metrics are recorded
  experiments.py     runs everything, writes tables and the trade-off plot
  visualize.py       draws the building, fire, detections and each planner's route
detection/
  check_dataset.py   confirms which class id is fire and which is smoke
  train_yolo.py      fine-tunes YOLO on D-Fire
  measure_detector.py measures misses, false alarms, delay, calibration on videos
configs/
  detector_errors_placeholder.json   FAKE numbers, for testing only
  videos_template.csv                how to list your evaluation videos
results/             example outputs (made with placeholder numbers)
results/notebooks/   notebook used for the SageMaker training and measurement run
```

---

## Part B first: run the simulation (5 minutes)

```bash
pip install -r requirements.txt
python -m firepath.experiments --scenarios 300        # about 4 minutes
python -m firepath.visualize --find-interesting        # one example picture
```

Outputs in `results/`:

| File | What it is |
|---|---|
| `summary.md` / `summary.csv` | main table: each policy's mean and 95% bootstrap CI |
| `tradeoff.png` | key figure: safety vs detour for every λ and threshold |
| `error_types.md` | ablation: false alarms only / misses only / delay only / all |
| `per_scenario.csv` | every run, for your own analysis |
| `route_example.png` | same fire, routes chosen by four planners |

### The routing policies

| Policy (spec) | How it decides | Role in the paper |
|---|---|---|
| `static` | shortest path, ignores fire | floor |
| `threshold:0.5` | block a zone if this frame's confidence ≥ 0.5 | **main baseline** |
| `threshold:0.25` | same, lower threshold (catches more, more false alarms) | baseline |
| `latched:0.5` | once a zone alarms it stays blocked | realistic alarm-panel baseline |
| `voting:0.5:2:3` | block if 2 of the last 3 frames ≥ 0.5 | temporal-verification baseline |
| `confidence:λ` | cost = length + λ × confidence | **our method** |
| `confidence:λ:0.5` | same, confidence smoothed over frames (EMA) | **our method + smoothing** |
| `oracle` | knows the true fire | ceiling, not a real system |

### Metrics

| Metric | Meaning |
|---|---|
| `success_rate` | share of people who reach an exit within 10 minutes |
| `fire_casualty_rate` | share who end up in a fire zone (the most important safety number) |
| `smoke_exposure_s` | seconds per person spent in smoke |
| `detour_ratio` | metres walked ÷ shortest possible route |
| `evac_time_s` | time to reach an exit |
| `route_flips` | how often a person's route changed (flickering detections cause this) |

### Choosing λ fairly

Do not pick λ by looking at the test results. Choose it on separate
validation scenarios, then report test scenarios:

```bash
python -m firepath.experiments --seed0 100000 --scenarios 200 --out results_val --skip-ablation
# look at results_val/tradeoff.png, choose λ (e.g. lowest casualty rate at detour ≤ threshold's)
python -m firepath.experiments --seed0 0 --scenarios 300 --out results_test
```

---

## Part A: train the detector and measure its real errors (GPU)

Run these on a machine with a GPU (the results here were made on SageMaker,
see `results/notebooks/`):

**1. Get D-Fire** (21,527 images, YOLO format) from the dataset's GitHub page
(gaiasd/DFireDataset) and unzip it so you have `D-Fire/train/` and `D-Fire/test/`.

**2. Check the class ids** (D-Fire has more fire boxes than smoke boxes):
```bash
python detection/check_dataset.py --root /content/D-Fire
```
It prints the `--names` to use.

**3. Fine-tune** (a few hours on a free Colab GPU):
```bash
python detection/train_yolo.py --root /content/D-Fire --names smoke fire --epochs 50
```
It prints the test-set mAP for your paper's detection table.

**4. Collect evaluation videos.** These must NOT be D-Fire training images.
Good sources: MIVIA/Foggia fire videos, FIRESENSE, D-Fire's surveillance
videos, plus your own short clips of fire-like scenes (sunsets through windows,
red lights, orange clothing). List them in a CSV like
`configs/videos_template.csv`:
```
path,label,onset_s
videos/fire_01.mp4,fire,0
videos/smoke_02.mp4,smoke,4.5
videos/office_01.mp4,none,
videos/sunset_01.mp4,confuser,
```
`onset_s` = second when fire/smoke first becomes visible. Make a second,
separate CSV for calibration if you can (`--calib-videos`).

**5. Measure the errors:**
```bash
python detection/measure_detector.py --weights runs/firepath/train/weights/best.pt \
    --videos videos.csv --calib-videos calib.csv --out configs/detector_errors_measured.json
```
It prints miss rate, median delay, recall and false-alarm rates (a table for
your paper), writes a per-video CSV, and saves the JSON the simulator needs.

**6. Run the simulation with your real numbers:**
```bash
python -m firepath.experiments --errors configs/detector_errors_measured.json --scenarios 300
```

---

## How the simulation works (for the Method section)

* **Building:** 5 floors × 8 rooms on a corridor, west and east staircases,
  2 ground-floor exits; 90 zones, one camera each. Edge lengths in metres.
* **Fire:** starts in a random zone and grows for 20–50 s before evacuation
  starts. Each 5-s step, fire spreads to a neighbour with p = 0.10; smoke
  spreads with p = 0.30 (0.55 up a staircase).
* **Cameras:** each step, each camera reports the highest box confidence or
  nothing. Misses, delay to first detection, per-step recall, false-alarm
  rate, extra false alarms in zones with fire-like objects, and confidence
  distributions are all taken from `measure_detector.py`. Confidences are
  Platt-calibrated so they behave like probabilities.
* **People:** 30 per scenario, starting in random rooms, walking at 1.2 m/s.
  Every 5 s each policy re-plans with Dijkstra (cost of entering a zone =
  length + penalty). A person who enters a fire zone counts as a casualty.
* **Fair comparison:** for each scenario the fire, the camera reports and the
  start rooms are generated once and replayed for every policy.

## Limitations to state in the paper

* Simple probabilistic fire/smoke spread, not a physics model such as FDS.
* No crowding or congestion between people.
* One camera per zone with full coverage; real buildings have blind spots.
* Camera errors are independent between zones; real errors may be correlated
  (for example, the same sunset affecting a whole side of the building).
* `confuser_zone_fraction` (share of zones with fire-like objects) is an
  assumption; vary it in a sensitivity analysis.
* A research prototype, not a certified evacuation system.

## Ideas for extra results

* Run Part A with a zero-shot detector (for example Grounding DINO with the
  prompts "fire" and "smoke") and compare its error profile and routing
  outcomes with the fine-tuned YOLO.
* Sensitivity analysis: change `p_fire`, `p_smoke` in `firepath/fire.py`, or
  `--floors` / `--corridor`, and check that conclusions hold.
* Day vs night videos: measure two JSON files and compare.
