"""Compare two splits and report whether weights trained on one can be evaluated on the other.

A model may only be scored on images it never saw. Because classes 2-4 carry horizontally flipped
copies, "never saw" has to be judged by source image, not by filename: if `MildG2 (193).png` was in the
training subset the model has also effectively seen `aug_flipped_7_MildG2 (193).png`.

    python scripts/check_split_overlap.py SplittedDataset SplittedDatasetV2

The first path is the split the weights were trained on, the second the one you want to evaluate on.
"""
import argparse
import collections
import pathlib
import re
import sys

AUG = re.compile(r"^aug_flipped_\d+_(.+)$")
SPLITS = ("train", "val", "test")


def read_split(root):
    """source image -> the subsets it appears in, counting a flipped copy as its original."""
    root = pathlib.Path(root)
    members = collections.defaultdict(set)
    counts = {}
    for split in SPLITS:
        n = 0
        for cls_dir in sorted((root / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.iterdir():
                if not f.is_file():
                    continue
                m = AUG.match(f.name)
                members[(cls_dir.name, m.group(1) if m else f.name)].add(split)
                n += 1
        counts[split] = n
    return members, counts


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("trained_on")
parser.add_argument("evaluate_on")
args = parser.parse_args()

trained, trained_counts = read_split(args.trained_on)
evaluate, evaluate_counts = read_split(args.evaluate_on)

for name, counts, members in ((args.trained_on, trained_counts, trained),
                              (args.evaluate_on, evaluate_counts, evaluate)):
    leaks = sum(1 for v in members.values() if len(v) > 1)
    print(f"{name}")
    print(f"  files        : train {counts['train']:,}  val {counts['val']:,}  test {counts['test']:,}")
    print(f"  source images: {len(members):,}")
    print(f"  images in more than one of its own subsets: {leaks:,}")

train_sources = {k for k, v in trained.items() if "train" in v}
eval_test = {k for k, v in evaluate.items() if "test" in v}
overlap = train_sources & eval_test
identical = all(trained.get(k) == v for k, v in evaluate.items()) and len(trained) == len(evaluate)

print()
print(f"test images of {args.evaluate_on} that were trained on in {args.trained_on}: "
      f"{len(overlap):,} of {len(eval_test):,}")

if identical:
    print("\nThe two splits are identical. The weights can be evaluated as they are.")
    sys.exit(0)
if not overlap:
    print("\nDifferent splits, but nothing in the evaluation test set was trained on.")
    print("The weights can be evaluated on it.")
    sys.exit(0)
print(f"\n{len(overlap) / len(eval_test):.0%} of the evaluation test set was in the training subset of "
      f"the other split.")
print("Scores from this combination would be optimistic. Retrain on the evaluation split instead:")
print("  ./scripts/rerun_reported.sh")
sys.exit(2)
