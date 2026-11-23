#!/usr/bin/env bash
# Re-run every result the paper reports, on the leakage-free split, and push the outcome.
#
# The previous SplittedDataset was built by augmenting classes 2-4 and only then splitting, so 179 of
# the 466 test images had their mirror image in training. This script rebuilds the split the other way
# round - split the 1,650 originals 7:1:2, then flip-augment classes 2, 3 and 4 of the training subset
# only - and retrains the runs that produced a confusion matrix in the paper, writing them to the same
# directories under the same names.
#
#   ./scripts/rerun_reported.sh              # everything, then commit and push
#   ./scripts/rerun_reported.sh split        # just rebuild the dataset
#   ./scripts/rerun_reported.sh tuning kd    # pick up from a particular stage
#   ./scripts/rerun_reported.sh all nopush   # run everything but leave the commit to you
#
# Budget roughly 45-55 GPU-hours end to end on one RTX 4090. Every stage is resumable: a run counts as
# finished only once its metrics.json exists, so re-running the script continues where it stopped.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${OSTEO_VENV:-/mnt/d/FA011/tf_gpu_env}"
DATA="${OSTEO_SPLIT_DIR:-$REPO/SplittedDatasetV2}"
LOGS="$REPO/_rerun_logs"
ARCHIVE_SUFFIX="_leaky_split"
# Where each stage writes. Setting OSTEO_RESULTS_NAME to something else leaves the old results exactly
# where they are, which is the way out when a Windows process holds a file inside them and the archive
# step cannot move them aside.
RESULTS_NAME="${OSTEO_RESULTS_NAME:-results}"
export OSTEO_OUTPUT_DIR="$RESULTS_NAME"
export OSTEO_TUNING_DIR="$REPO/tuning/$RESULTS_NAME"
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(all)

mkdir -p "$LOGS"
want() { for s in "${STAGES[@]}"; do [ "$s" = "all" ] && return 0; [ "$s" = "$1" ] && return 0; done; return 1; }
say() { printf '\n\033[1m== %s\033[0m  %s\n' "$1" "$(date '+%H:%M:%S')"; }

# ── preflight ───────────────────────────────────────────────────────────────
say "preflight"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  echo "environment : $VENV"
else
  echo "WARNING: $VENV not found, using whatever python is on PATH" >&2
fi
python -c "import tensorflow as tf; print('tensorflow  :', tf.__version__); \
print('gpus        :', tf.config.list_physical_devices('GPU'))" || {
  echo "TensorFlow is not importable in this environment" >&2; exit 1; }
python -c "import jupyter_core" 2>/dev/null || { echo "jupyter is missing: pip install jupyter" >&2; exit 1; }
cd "$REPO"
git rev-parse --is-inside-work-tree >/dev/null || { echo "not a git repository" >&2; exit 1; }

run_notebook() {   # run_notebook <name> <notebook-dir> <notebook>
  local name="$1" dir="$2" nb="$3"
  say "$name"
  ( cd "$dir" && jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=python3 \
      --ExecutePreprocessor.timeout=-1 \
      --output-dir "$LOGS" --output "executed_${name}.ipynb" "$nb" ) \
    2>&1 | tee "$LOGS/${name}.log"
}

# ── 1. the corrected split ──────────────────────────────────────────────────
if want split; then
  say "building the corrected split"
  python "$REPO/scripts/build_corrected_split.py" --out "$DATA" 2>&1 | tee "$LOGS/split.log"
fi
[ -d "$DATA/train" ] || { echo "no dataset at $DATA - run the split stage first" >&2; exit 1; }
export OSTEO_DATA_DIR="$DATA"
echo "data        : $OSTEO_DATA_DIR"
echo "results     : <stage>/$RESULTS_NAME"

# ── 2. set the old results aside, once ──────────────────────────────────────
if want archive && [ "$RESULTS_NAME" = "results" ]; then
  say "archiving the results of the leaky split"
  # Renaming a directory on /mnt/d fails while any process holds a handle inside it, which on this
  # machine means a Jupyter kernel left running from an earlier session. Moving the contents instead
  # usually succeeds, and if even that fails the holder has to go before training can start.
  archive_dir() {
    local src="$1" dst="$2"
    mv "$src" "$dst" 2>/dev/null && return 0
    mkdir -p "$dst" 2>/dev/null || return 1
    if (shopt -s dotglob nullglob; mv "$src"/* "$dst"/ 2>/dev/null); then
      rmdir "$src" 2>/dev/null || true
      return 0
    fi
    return 1
  }

  for stage in teacher tuning kd; do
    if [ ! -d "$REPO/$stage/results" ] || [ -d "$REPO/$stage/results$ARCHIVE_SUFFIX" ]; then
      echo "  $stage: nothing to archive"
      continue
    fi
    if archive_dir "$REPO/$stage/results" "$REPO/$stage/results$ARCHIVE_SUFFIX"; then
      echo "  $stage/results -> $stage/results$ARCHIVE_SUFFIX"
    else
      echo "ERROR: cannot move $stage/results aside; something is holding a file inside it." >&2
      echo "       Usually a Jupyter kernel from an earlier session:" >&2
      echo "         pkill -f jupyter; pkill -f ipykernel" >&2
      echo "       Then close any editor or Explorer window on that folder and run this script again." >&2
      command -v lsof >/dev/null && lsof +D "$REPO/$stage/results" 2>/dev/null | head -5 >&2
      exit 1
    fi
  done
fi

# Running a single stage without archiving first would silently keep the old results: every run is
# skipped when its metrics.json already exists, so nothing would be retrained.
if ! want keepresults && [ "$RESULTS_NAME" = "results" ]; then
  for stage in teacher tuning kd; do
    want "$(printf '%s' "$stage" | sed 's/^teacher$/screening/')" || continue
    if [ -d "$REPO/$stage/results" ] && [ ! -d "$REPO/$stage/results$ARCHIVE_SUFFIX" ]; then
      echo "ERROR: $stage/results still holds the results of the old split." >&2
      echo "       Run the archive stage first, pass keepresults to train into it deliberately," >&2
      echo "       or set OSTEO_RESULTS_NAME=results_v2 to write beside it and leave it alone:" >&2
      echo "         ./scripts/rerun_reported.sh archive $* " >&2
      exit 1
    fi
  done
fi

# ── 3. Stage II, the screened architectures ─────────────────────────────────
if want screening; then
  # One process per architecture. TensorFlow does not return host memory between models, so running
  # all seventeen in a single kernel exhausts RAM part way through and the kernel is killed; the
  # tuning and distillation stages already work this way.
  say "screened architectures, one process each"
  mapfile -t SCREEN_MODELS < <(python - "$REPO/scripts/reported_runs.json" <<'PY'
import json, sys
for name in json.load(open(sys.argv[1]))["screening"]:
    print(name)
PY
)
  for model in "${SCREEN_MODELS[@]}"; do
    if [ -f "$REPO/teacher/$RESULTS_NAME/$model/metrics.json" ]; then
      echo "  $model: already finished"
      continue
    fi
    say "screening: $model"
    ( cd "$REPO/teacher" && OSTEO_MODELS="$model" \
      jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 \
        --output-dir "$LOGS" --output "executed_screening_${model}.ipynb" \
        Teacher_Screening_EarlyStopping.ipynb ) 2>&1 | tee "$LOGS/screening_${model}.log" || \
      echo "WARNING: $model did not finish; re-run the script to retry it" >&2
  done
fi

# ── 4. Stage III, each architecture at the configuration the paper reports ──
if want tuning; then
  say "tuned models, one configuration each"
  python - "$REPO/scripts/reported_runs.json" <<'PY' > "$LOGS/tuning_plan.sh"
import json, sys
spec = json.load(open(sys.argv[1]))
for r in spec["tuning_dl"]:
    print(f'run_tuned "{r["model"]}" {r["learning_rate"]} {r["dropout_rate"]} '
          f'{r["optimizer"]} {r["batch_size"]}')
PY
  run_tuned() {    # run_tuned <model> <lr> <dropout> <optimiser> <batch>
    local model="$1"
    say "tuning: $model  (lr $2, dropout $3, $4, batch $5)"
    ( cd "$REPO/tuning" && \
      OSTEO_MODELS="$model" OSTEO_RUN_CUSTOM=0 OSTEO_RUN_ML=0 \
      OSTEO_GRID_LR="$2" OSTEO_GRID_DROPOUT="$3" OSTEO_GRID_OPTIMIZER="$4" OSTEO_GRID_BATCH="$5" \
      jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 \
        --output-dir "$LOGS" --output "executed_tuning_${model}.ipynb" \
        Tuning_GridSearch_And_Baselines.ipynb ) 2>&1 | tee "$LOGS/tuning_${model}.log"
  }
  # shellcheck disable=SC1090
  source "$LOGS/tuning_plan.sh"

  # A bare comma selects no grid models: the notebook's _env() treats an empty string as "use the
  # default", which is "all", so the list has to be non-empty but parse to nothing.
  say "tuning: Custom-ENB5 and the classical baselines"
  ( cd "$REPO/tuning" && \
    OSTEO_MODELS="," OSTEO_RUN_CUSTOM=1 OSTEO_RUN_ML=1 \
    jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 \
      --output-dir "$LOGS" --output "executed_tuning_custom_ml.ipynb" \
      Tuning_GridSearch_And_Baselines.ipynb ) 2>&1 | tee "$LOGS/tuning_custom_ml.log"
fi

# ── 5. Stage IV, teachers, 36 distillations, the sweep ──────────────────────
if want kd; then
  run_notebook kd "$REPO/kd" KD_MultiTeacher_Distillation.ipynb
fi

# ── 6. commit and push ──────────────────────────────────────────────────────
if want push && ! want nopush; then
  say "committing and pushing"
  cd "$REPO"
  # The images themselves stay out of git: the split is rebuilt deterministically from the original
  # dataset by build_corrected_split.py, and the manifest records exactly which file went where.
  git add -A teacher tuning kd scripts
  git add -f "$DATA/split_manifest.csv" 2>/dev/null || true
  if git diff --cached --quiet; then
    echo "nothing to commit"
  else
    git commit -q -F - <<'MSG'
Re-run every reported result on the leakage-free split

The earlier SplittedDataset was produced by flip-augmenting classes 2, 3 and 4
and splitting afterwards, so originals and their mirrors were scattered across
the subsets: 302 source images had their two versions in different subsets and
179 of the 466 test images had their mirror in training, which inflated the
scores for exactly those three grades.

SplittedDatasetV2 splits the 1,650 originals 7:1:2 first (1,155 / 165 / 330)
and only then flips classes 2-4 of the training subset, so no image can appear
in two subsets in any form. The split script asserts that before it finishes.

Every run that produced a confusion matrix in the paper is retrained on that
split and written to the same directories: 17 screened architectures, 15 tuned
models at their reported configurations plus Custom-ENB5 and the five
classical baselines, and the distillation stage with its four teachers, two
fused ensembles, 36 student runs and the temperature sweep. The previous
results are kept beside them under results_leaky_split for comparison.
MSG
    git push origin HEAD
    echo "pushed: $(git log --oneline -1)"
  fi
fi

say "done"
echo "logs in $LOGS"
