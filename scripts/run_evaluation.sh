#!/usr/bin/env bash
# Evaluate the trained checkpoints on the corrected split and push the refreshed results.
#
# Nothing is retrained. Every checkpoint already in the results tree - best.keras, teacher.keras,
# student.keras - is loaded, run over the test subset, and its metrics, reports, confusion matrix,
# predictions and figures are written back into the same folder. The _summary tables and comparison
# figures of each stage are rebuilt from those files afterwards. The five classical baselines keep no
# weights file, so they are refitted, which takes seconds.
#
#   ./scripts/run_evaluation.sh                # all three stages, then commit and push
#   ./scripts/run_evaluation.sh kd             # one stage
#   ./scripts/run_evaluation.sh all nopush     # evaluate, leave the commit to you
#   ./scripts/run_evaluation.sh dryrun         # list the checkpoints that would be evaluated
#
# Expect well under an hour for all 88 runs on one RTX 4090.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${OSTEO_VENV:-/mnt/d/FA011/tf_gpu_env}"
DATA="${OSTEO_SPLIT_DIR:-$REPO/SplittedDatasetV2}"
LOGS="$REPO/_rerun_logs"
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(all)

has() { for a in "${ARGS[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
say() { printf '\n\033[1m== %s\033[0m  %s\n' "$1" "$(date '+%H:%M:%S')"; }

STAGE=all
for a in "${ARGS[@]}"; do
  case "$a" in screening|tuning|kd|all) STAGE="$a" ;; esac
done
EXTRA=()
has dryrun && EXTRA+=(--dry-run)

mkdir -p "$LOGS"

say "preflight"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  echo "environment : $VENV"
else
  echo "WARNING: $VENV not found, using whatever python is on PATH" >&2
fi
python -c "import tensorflow as tf; print('tensorflow  :', tf.__version__); \
print('gpus        :', tf.config.list_physical_devices('GPU'))"
[ -d "$DATA/test" ] || { echo "no test subset at $DATA" >&2; exit 1; }
echo "data        : $DATA"
echo "stage       : $STAGE"

say "evaluating the saved checkpoints"
cd "$REPO"
python scripts/evaluate_weights.py --stage "$STAGE" --data-dir "$DATA" "${EXTRA[@]}" \
  2>&1 | tee "$LOGS/evaluate_${STAGE}.log"

if has dryrun; then
  say "dry run, nothing written"
  exit 0
fi

if ! has nopush; then
  say "committing and pushing"
  git add -A teacher tuning kd
  if git diff --cached --quiet; then
    echo "nothing to commit"
  else
    git commit -q -F - <<MSG
Evaluate the trained checkpoints on the corrected split

Every saved checkpoint was re-run over the test subset of the leakage-free
split and its metrics, classification report, confusion matrix, per-image
predictions and figures rewritten in place; no model was retrained. The
_summary tables and comparison figures of each stage were rebuilt from those
files, and the classical baselines, which keep no weights file, were refitted.

Stage: $STAGE
Data : $(basename "$DATA")
MSG
    git push origin HEAD
    echo "pushed: $(git log --oneline -1)"
  fi
fi

say "done"
echo "log: $LOGS/evaluate_${STAGE}.log"
