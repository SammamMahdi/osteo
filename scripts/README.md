# Re-running the reported results on a leakage-free split

## Why

`SplittedDataset/` was built by flip-augmenting classes 2, 3 and 4 and splitting afterwards, with
`splitfolders.ratio(..., seed=1337, ratio=(0.7, 0.1, 0.2))`. The splitter had no way to know that
`MildG2 (193).png` and `aug_flipped_12_MildG2 (193).png` are the same radiograph, so the two versions were
scattered across subsets:

| | |
|---|---|
| source images with their two versions in different subsets | 302 of 1,650 |
| test images whose mirror is in the training set | 179 of 466 (38.4%) |
| affected classes | KL 2 (70.2%), KL 3 (69.7%), KL 4 (61.4%) |
| unaffected classes | KL 0 and KL 1, which were never augmented |

Training augmentation includes horizontal flipping, so for those three grades the network very likely saw
the test image's pixel content during training. Their reported scores are optimistically biased.

## The corrected split

`build_corrected_split.py` reverses the order of operations:

1. split the **1,650 originals** per class, 7:1:2 → **1,155 train / 165 validation / 330 test**
2. leave validation and test as originals only
3. flip-augment classes 2, 3 and 4 of the **training subset only**

| class | train | val | test | total |
|---|---|---|---|---|
| 0 Normal | 352 | 50 | 101 | 503 |
| 1 Doubtful | 342 | 49 | 97 | 488 |
| 2 Mild | 324 (162 + 162 flipped) | 23 | 47 | 394 |
| 3 Moderate | 310 (155 + 155 flipped) | 22 | 44 | 376 |
| 4 Severe | 288 (144 + 144 flipped) | 21 | 41 | 350 |
| **total** | **1,616** | **165** | **330** | **2,111** |

The script asserts before finishing that no source image appears in more than one subset, and writes
`split_manifest.csv` recording the subset, class, filename and source image of every file. The images
themselves are not committed; the split is rebuilt deterministically from `Original Dataset/` with
seed 1337.

## What gets re-run

`reported_runs.json` lists every run that produced a confusion matrix in the paper, with the configuration
it used. `make_reported_runs.py` regenerates it from a results tree.

| stage | runs | directories |
|---|---|---|
| Screening | 17 architectures | `teacher/results/<Model>/` |
| Tuning | 15 models at their reported configuration | `tuning/results/<Model>/best/` |
| Tuning | Custom-ENB5 and 5 classical baselines | `tuning/results/<Name>/` |
| Distillation | 4 teachers, 2 fused ensembles, 36 student runs, 8 sweep points | `kd/results/...` |

The grid search is **not** repeated: each tuned model is trained once, at the configuration the paper
reports, which is what `tuning/results/<Model>/best/` holds. Names and directories are unchanged, so the
new results are drop-in replacements.

## Running it

```bash
cd /mnt/d/FA019/new
./scripts/rerun_reported.sh               # everything, then commit and push
./scripts/rerun_reported.sh split         # rebuild the dataset only
./scripts/rerun_reported.sh tuning kd     # resume from a stage
./scripts/rerun_reported.sh all nopush    # run everything, leave the commit to you
```

From Windows, `scripts\rerun_reported.bat` does the same through WSL.

Budget roughly **45–55 GPU-hours** on one RTX 4090: about 11 h screening, 4–6 h tuning, 31 h
distillation. Every stage is resumable — a run counts as finished only once its `metrics.json` exists —
so the script can be stopped and started again.

Before training, the script moves `teacher/results`, `tuning/results` and `kd/results` to
`results_leaky_split` so the old numbers stay available for comparison. Logs go to `_rerun_logs/`,
one per stage, together with the executed notebooks.

Environment variables worth knowing: `OSTEO_VENV` (default `/mnt/d/FA011/tf_gpu_env`) and
`OSTEO_SPLIT_DIR` (default `SplittedDatasetV2`).

## Expect the numbers to fall

Grades 2–4 carried the leakage, so their per-class scores should drop the most, and the macro averages
with them. The comparisons between models are unaffected, since every model shared the same split.
