"""Check a YOLO-format dataset before training: images, labels, and which class id is which.

D-Fire's page reports 14,692 fire boxes and 11,865 smoke boxes, so the class
with ~14.7k boxes is fire. Run this, then set --names in train_yolo.py so the
names match the ids.

Usage:
    python detection/check_dataset.py --root /content/D-Fire
"""
import argparse
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset folder containing train/ and test/")
    a = p.parse_args()
    root = Path(a.root)
    total = Counter()
    for split in sorted(d for d in root.iterdir() if d.is_dir()):
        imgs = [f for f in (split / "images").glob("*") if f.suffix.lower() in IMG_EXT]
        labels = list((split / "labels").glob("*.txt"))
        boxes, empty = Counter(), 0
        for lf in labels:
            lines = [l for l in lf.read_text().splitlines() if l.strip()]
            empty += not lines
            for l in lines:
                boxes[int(l.split()[0])] += 1
        total.update(boxes)
        print(f"{split.name:8s} images={len(imgs):6d} label_files={len(labels):6d} "
              f"empty(negative)={empty:6d} boxes_per_class={dict(sorted(boxes.items()))}")
        missing = {f.stem for f in imgs} - {f.stem for f in labels}
        if missing:
            print(f"  {len(missing)} images have no label file (YOLO treats them as negatives)")
    print("\nAll splits, boxes per class id:", dict(sorted(total.items())))
    if len(total) == 2 and len(set(total.values())) == 1:
        print("Both classes have the same number of boxes, so the ids can't be told apart this way. "
              "Open a few images with their labels to check.")
    elif len(total) == 2:
        fire_id = max(total, key=total.get)
        print(f"Class {fire_id} has the most boxes, so it is most likely FIRE (D-Fire: 14,692 fire vs 11,865 smoke).")
        names = ["fire", "smoke"] if fire_id == 0 else ["smoke", "fire"]
        print(f"Use:  --names {' '.join(names)}")


if __name__ == "__main__":
    main()
