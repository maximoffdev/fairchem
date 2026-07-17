"""Split a directory of IQA .pkl samples into train/test/val sets by sum formula.

The split is *grouped* by sum formula so that all conformers/samples sharing the
same stoichiometry land in exactly one split (no formula leakage between splits).

Example
-------
    python -m fairchem.core.scripts.iqa_split_dataset \
        --input_dir /data/iqa/all \
        --output_dir /data/iqa/split \
        --train_size 0.9 --test_size 0.095 --val_size 0.005
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import os
import pickle
import shutil

from sklearn.model_selection import GroupShuffleSplit


def _sum_formula(path: str) -> str:
    """Return a canonical 'Z:count_Z:count' sum-formula string for a .pkl sample."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    z_list = data.atomic_numbers.view(-1).tolist()
    counts = collections.Counter(z_list)
    return "_".join(f"{z}:{count}" for z, count in sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset by sum formula into train/test/val."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input .pkl files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save train/test/val splits"
    )
    parser.add_argument(
        "--train_size", type=float, default=0.9, help="Proportion of data for training"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.095, help="Proportion of data for testing"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.005, help="Proportion of data for validation"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducible splits"
    )
    args = parser.parse_args()

    # Validate fractions.
    total = args.train_size + args.test_size + args.val_size
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        parser.error(
            f"train_size + test_size + val_size must sum to 1.0 (got {total:.6f})"
        )
    if min(args.train_size, args.test_size, args.val_size) < 0:
        parser.error("Split fractions must be non-negative.")

    # 1. Collect files.
    file_paths = glob.glob(os.path.join(args.input_dir, "*.pkl"))
    if not file_paths:
        print(f"No .pkl files found in {args.input_dir}")
        return

    # 2. Extract sum formulas as grouping keys.
    print("Extracting sum formulas...")
    groups = []
    valid_files = []
    for path in file_paths:
        groups.append(_sum_formula(path))
        valid_files.append(path)

    # 3. First grouped split: train vs. (test + val).
    gss_train = GroupShuffleSplit(
        n_splits=1, train_size=args.train_size, random_state=args.random_state
    )
    train_idx, rest_idx = next(gss_train.split(X=valid_files, groups=groups))

    train_files = [valid_files[i] for i in train_idx]
    rest_files = [valid_files[i] for i in rest_idx]
    rest_groups = [groups[i] for i in rest_idx]

    # 4. Second grouped split: split the remainder into test vs. val.
    #    Fractions are renormalized relative to the held-out remainder.
    rest_fraction = args.test_size + args.val_size
    if rest_fraction <= 0 or not rest_files:
        test_files, val_files = rest_files, []
    else:
        test_fraction = args.test_size / rest_fraction
        gss_test = GroupShuffleSplit(
            n_splits=1, train_size=test_fraction, random_state=args.random_state
        )
        # Guard: a second grouped split needs at least 2 distinct groups.
        if len(set(rest_groups)) < 2:
            print(
                "Warning: fewer than 2 unique formulas in the held-out remainder; "
                "assigning all remaining files to the test split."
            )
            test_files, val_files = rest_files, []
        else:
            test_sub_idx, val_sub_idx = next(
                gss_test.split(X=rest_files, groups=rest_groups)
            )
            test_files = [rest_files[i] for i in test_sub_idx]
            val_files = [rest_files[i] for i in val_sub_idx]

    # 5. Create output directories.
    split_dirs = {
        "train": os.path.join(args.output_dir, "train"),
        "test": os.path.join(args.output_dir, "test"),
        "val": os.path.join(args.output_dir, "val"),
    }
    for d in split_dirs.values():
        os.makedirs(d, exist_ok=True)

    # 6. Copy files.
    for name, files in (
        ("train", train_files),
        ("test", test_files),
        ("val", val_files),
    ):
        print(f"Copying {len(files)} files to {split_dirs[name]}...")
        for f in files:
            shutil.copy(f, split_dirs[name])

    # 7. Summary.
    print("\n--- Split Summary ---")
    print(f"Total files: {len(valid_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Number of unique sum formulas: {len(set(groups))}")


if __name__ == "__main__":
    main()
