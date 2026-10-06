"""Train chosen models under one chosen hyperparameter configuration and report TEST metrics.

The grid search records validation metrics for every combination but evaluates the test set only for each
model's winner. This script fills that gap: it trains a specific configuration and evaluates it on the test
set, writing the usual metrics, report and figures.

It reuses the notebook's own code (the same trick the grid worker uses), so the recipe cannot drift.

Default configuration = Custom-ENB5's (rmsprop, lr 1e-4, dropout 0.3, batch 32).

    python cross_config_test.py                                    # the two models, Custom-ENB5's config
    python cross_config_test.py --seeds 3                          # repeat with 3 seeds (recommended)
    python cross_config_test.py --models ResNet50 --optimizer adam --lr 0.0001 --dropout 0.3 --batch 32

Results go to results/_cross_config/<Model>__<combo>__seed<N>/ plus a summary.csv alongside them.
"""
import argparse
import json
import os
from pathlib import Path

import nbformat

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--notebook", default="Tuning_GridSearch_And_Baselines.ipynb")
parser.add_argument("--models", default="InceptionResNetV2,ResNet50")
parser.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "adam", "sgd"])
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--seeds", type=int, default=1, help="repeat each run with seeds 42, 43, ... (variance check)")
parser.add_argument("--skip-existing", type=int, default=1)
args = parser.parse_args()

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SKIP_TAG = "# worker" + ": skip"     # assembled, so this line cannot match itself
ns = {}
for cell in nbformat.read(args.notebook, as_version=4).cells:
    if cell.cell_type != "code":
        continue
    src = chr(10).join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
    if SKIP_TAG in src:
        continue
    exec(compile(src, "<notebook>", "exec"), ns)
for needed in ("train_one", "TUNED_MODELS", "finalize_run", "make_eval_generator", "OUT_DIR"):
    assert needed in ns, f"the notebook cell defining {needed} was skipped"

CFG, OUT_DIR = ns["CFG"], ns["OUT_DIR"]
combo = ns["combo_id"](args.lr, args.dropout, args.optimizer, args.batch)
cross_dir = OUT_DIR / "_cross_config"
cross_dir.mkdir(parents=True, exist_ok=True)
rows = []

for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
    assert model_name in ns["TUNED_MODELS"], f"Unknown model {model_name}"
    spec = ns["TUNED_MODELS"][model_name]
    for i in range(args.seeds):
        seed = CFG["SEED"] + i
        label = f"{model_name} [{combo}] seed {seed}"
        out = cross_dir / f"{model_name}__{combo}__seed{seed}"
        if args.skip_existing and (out / "metrics.json").exists():
            print(f"{label}: already done - loading")
            rows.append(json.loads((out / "metrics.json").read_text()))
            continue

        print(f"\n=== {label}")
        CFG["SEED"] = seed                       # train_one seeds from CFG
        model, hist, info = ns["train_one"](model_name, spec, args.lr, args.dropout, args.optimizer,
                                            args.batch, tag="cross-config")
        test_gen = ns["make_eval_generator"](ns["TEST_DIR"], spec["pre"], args.batch)
        prob = model.predict(test_gen, verbose=0)
        info["seed"] = seed
        res = ns["finalize_run"](label, out, info, test_gen.classes, prob, test_gen.filenames, hist=hist)
        rows.append(res)
        del model
        ns["cleanup_backend"]()

CFG["SEED"] = 42
if rows:
    import pandas as pd
    df = pd.DataFrame(rows)[["model", "seed", "optimizer", "learning_rate", "dropout_rate", "batch_size",
                             "epochs_run", "best_epoch", "val_accuracy", "val_f1_macro", "test_accuracy",
                             "test_f1_macro", "test_f1_weighted", "test_auc_macro", "fit_time_min"]]
    df.to_csv(cross_dir / "summary.csv", index=False)
    print(f"\n{'=' * 100}\nTEST results for configuration {combo}\n{'=' * 100}")
    print(df.round(4).to_string(index=False))
    if args.seeds > 1:
        print("\nmean +/- sd per model:")
        print(df.groupby("model")[["test_accuracy", "test_f1_macro"]].agg(["mean", "std"]).round(4).to_string())
    print("\nSaved:", cross_dir / "summary.csv")
