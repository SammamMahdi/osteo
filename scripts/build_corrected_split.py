"""Build a leakage-free 7:1:2 split of the original 1,650 radiographs.

The previous split was made over a folder that already contained one horizontally flipped copy of every
image in classes 2, 3 and 4. splitfolders then scattered originals and their copies at random, so 302
source images had their two versions in different subsets and 179 of the 466 test images had their mirror
in training. This script fixes that by splitting the originals first and augmenting only afterwards, and
only the training subset:

  1. split the 1,650 original images per class into 7:1:2  ->  1,155 / 165 / 330
  2. leave validation and test exactly as they are, originals only
  3. duplicate classes 2, 3 and 4 of the training subset with a horizontal flip

Because the split happens before the flip, no image can appear in two subsets in any form.

    python scripts/build_corrected_split.py                    # writes ../SplittedDatasetV2
    python scripts/build_corrected_split.py --out /path/to/dir --seed 1337
"""
import argparse
import csv
import pathlib
import random
import shutil
import sys

import cv2

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "Original Dataset" / "MedicalExpert-II (httpsdata.mendeley.comdatasetst9ndx37v5h1)"
CLASS_DIRS = {"0": "0Normal", "1": "1Doubtful", "2": "2Mild", "3": "3Moderate", "4": "4Severe"}
AUGMENT_CLASSES = ("2", "3", "4")
RATIO = (0.7, 0.1, 0.2)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default=str(REPO / "SplittedDatasetV2"))
parser.add_argument("--source", default=str(SOURCE))
parser.add_argument("--seed", type=int, default=1337, help="the seed the original split used")
parser.add_argument("--force", action="store_true", help="rebuild even if the output folder exists")
args = parser.parse_args()

out = pathlib.Path(args.out)
source = pathlib.Path(args.source)
assert source.is_dir(), f"original dataset not found: {source}"
if out.exists() and not args.force:
    print(f"{out} already exists; pass --force to rebuild it")
    sys.exit(0)
if out.exists():
    shutil.rmtree(out)


SPLITS = ("train", "val", "test")


def allocate(class_sizes, ratio, targets):
    """Allocate each class across the three subsets, hitting both the per-class and the overall totals.

    Flooring alone lands on 1,153 / 163 / 328 rather than the 1,155 / 165 / 330 the design calls for, so
    each class's leftover images go to whichever subset is furthest below its overall target, ties broken
    by the largest fractional remainder. Every class still keeps exactly its own number of images.
    """
    counts = {c: [int(n * r) for r in ratio] for c, n in class_sizes.items()}
    deficit = [targets[i] - sum(counts[c][i] for c in class_sizes) for i in range(3)]
    for c, n in class_sizes.items():
        remainder = [n * r - int(n * r) for r in ratio]
        for _ in range(n - sum(counts[c])):
            best = max(range(3), key=lambda i: (deficit[i], remainder[i]))
            counts[c][best] += 1
            deficit[best] -= 1
    for c, n in class_sizes.items():
        assert sum(counts[c]) == n, f"class {c}: {counts[c]} does not sum to {n}"
    return counts


class_files = {}
for cls, folder in CLASS_DIRS.items():
    class_files[cls] = sorted(p for p in (source / folder).iterdir() if p.is_file())
n_total = sum(len(v) for v in class_files.values())
targets = [round(n_total * r) for r in RATIO]
alloc = allocate({c: len(v) for c, v in class_files.items()}, RATIO, targets)

rng = random.Random(args.seed)
rows, totals = [], {s: {} for s in SPLITS}

for cls, folder in CLASS_DIRS.items():
    files = list(class_files[cls])
    rng.shuffle(files)
    n_train, n_val, n_test = alloc[cls]
    # the remainder rule above is applied per class, so the subset totals come out exact overall
    chunks = {"train": files[:n_train],
              "val": files[n_train:n_train + n_val],
              "test": files[n_train + n_val:]}
    assert sum(len(v) for v in chunks.values()) == len(files)

    for split, chunk in chunks.items():
        dest = out / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for p in chunk:
            shutil.copy2(p, dest / p.name)
            rows.append(dict(split=split, cls=cls, file=p.name, kind="original", source=p.name))
        totals[split][cls] = len(chunk)

    # augmentation happens only here, after the split, and only on the training subset
    if cls in AUGMENT_CLASSES:
        dest = out / "train" / cls
        for i, p in enumerate(sorted(chunks["train"])):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            assert img is not None, f"could not read {p}"
            name = f"aug_flipped_{i}_{p.name}"
            cv2.imwrite(str(dest / name), cv2.flip(img, 1))
            rows.append(dict(split="train", cls=cls, file=name, kind="flipped", source=p.name))
        totals["train"][cls] += len(chunks["train"])

manifest = out / "split_manifest.csv"
with manifest.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["split", "cls", "file", "kind", "source"])
    w.writeheader()
    w.writerows(rows)

# ── report, and check the guarantee the whole exercise is about ─────────────
print(f"source: {source}")
print(f"output: {out}\n")
header = f"{'class':<8}" + "".join(f"{s:>12}" for s in ("train", "val", "test")) + f"{'total':>12}"
print(header + "\n" + "-" * len(header))
for cls in CLASS_DIRS:
    line = f"{cls:<8}" + "".join(f"{totals[s][cls]:>12,}" for s in ("train", "val", "test"))
    print(line + f"{sum(totals[s][cls] for s in totals):>12,}")
print("-" * len(header))
print(f"{'total':<8}" + "".join(f"{sum(totals[s].values()):>12,}" for s in ("train", "val", "test"))
      + f"{sum(sum(t.values()) for t in totals.values()):>12,}")

originals = {s: sum(1 for r in rows if r["split"] == s and r["kind"] == "original") for s in totals}
print(f"\noriginals only: train {originals['train']:,}, val {originals['val']:,}, "
      f"test {originals['test']:,}  (= {sum(originals.values()):,})")
print(f"flipped copies added to training: {sum(1 for r in rows if r['kind'] == 'flipped'):,}")

by_source = {}
for r in rows:
    by_source.setdefault((r["cls"], r["source"]), set()).add(r["split"])
leaks = [k for k, v in by_source.items() if len(v) > 1]
assert not leaks, f"{len(leaks)} source images appear in more than one subset: {leaks[:5]}"
print(f"\nleakage check: {len(by_source):,} source images, none appearing in more than one subset")
print(f"manifest: {manifest}")
