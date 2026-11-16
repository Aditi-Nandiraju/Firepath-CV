"""Fine-tune a small YOLO model on D-Fire (run on Google Colab with a GPU).

Usage:
    python detection/train_yolo.py --root /content/D-Fire --names smoke fire --epochs 50

Run check_dataset.py first to confirm which class id is fire and which is smoke.
D-Fire has train/ and test/ folders; we split 10% of train off as validation so
the test set stays untouched for the final numbers.
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml


def make_val_split(root: Path, frac: float, seed: int) -> Path:
    out = root.parent / (root.name + "_split")
    if out.exists():
        print("Reusing existing split at", out)
        return out
    imgs = sorted((root / "train" / "images").glob("*"))
    random.Random(seed).shuffle(imgs)
    n_val = int(len(imgs) * frac)
    for split, files in [("val", imgs[:n_val]), ("train", imgs[n_val:])]:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        for im in files:
            shutil.copy(im, out / split / "images" / im.name)
            lab = root / "train" / "labels" / (im.stem + ".txt")
            if lab.exists():
                shutil.copy(lab, out / split / "labels" / lab.name)
    shutil.copytree(root / "test", out / "test")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--names", nargs=2, required=True, help="class names in id order, e.g. smoke fire")
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt / yolov8s.pt / yolo11n.pt ...")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default="runs/firepath")
    a = p.parse_args()

    from ultralytics import YOLO

    data_root = make_val_split(Path(a.root), a.val_frac, a.seed)
    cfg = {"path": str(data_root), "train": "train/images", "val": "val/images",
           "test": "test/images", "names": {0: a.names[0], 1: a.names[1]}}
    yaml_path = data_root / "dfire.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))

    model = YOLO(a.model)
    model.train(data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                seed=a.seed, project=a.project, name="train", exist_ok=True)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("\nBest weights:", best)
    metrics = YOLO(str(best)).val(data=str(yaml_path), split="test", imgsz=a.imgsz)
    print("Test-set mAP50:", metrics.box.map50, " mAP50-95:", metrics.box.map)
    print("Per-class mAP50:", dict(zip(a.names, metrics.box.ap50)))


if __name__ == "__main__":
    main()
