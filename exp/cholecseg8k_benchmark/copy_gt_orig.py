#!/usr/bin/env python3
"""Copy files from cholecseg8k_test_{fold}.csv into GT_orig_{fold} with same structure as ground_truth_{fold}."""
import argparse
import csv
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
SPLITS_DIR = BASE / "splits"
OUTPUTS_DIR = BASE / "infer_outputs"


def run_fold(fold: int) -> None:
    csv_path = SPLITS_DIR / f"cholecseg8k_test_{fold}.csv"
    out_dir = OUTPUTS_DIR / f"GT_orig_{fold}"
    if not csv_path.exists():
        print(f"Skip fold {fold}: {csv_path} not found")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            video = row["video"]
            src = Path(row["file"].strip())
            frame_id = row["frame_id"].strip()
            dst_dir = out_dir / video
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"frame_{int(frame_id):06d}_gt_idx.png"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing.append(str(src))
    print(f"Fold {fold}: copied {copied} files to {out_dir}")
    if missing:
        print(f"  Missing {len(missing)} source files (first 5):")
        for p in missing[:5]:
            print(f"    {p}")


def main():
    parser = argparse.ArgumentParser(description="Copy GT files from split CSV to GT_orig_{fold}")
    parser.add_argument("--fold", type=int, default=None, help="Single fold (default: all folds 0..4)")
    args = parser.parse_args()
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = list(range(5))  # 0..4
    for fold in folds:
        run_fold(fold)


if __name__ == "__main__":
    main()
