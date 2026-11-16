"""Re-evaluate the saved checkpoints and rebuild every reported artifact, without retraining anything.

Each run's folder already holds the weights the paper reports, under the names the notebooks gave them:

    teacher/results/<Model>/best.keras                 screened architectures
    teacher/results/<Model>/head_best.pt               the two timm models
    tuning/results/<Model>/best/best.keras             each tuned model at its winning configuration
    tuning/results/Custom-ENB5/best.keras              the customised EfficientNet-B5
    kd/results/teachers/<Teacher>/teacher.keras        the distillation teachers
    kd/results/<Student>/<config>/student.keras        the 36 distillation runs
    kd/results/_sweep/.../student.keras                the temperature sweep

This script loads each of them, runs the test subset through it, and writes the metrics, reports,
confusion matrix, predictions and figures back into the same folder, by calling the finalising function
of the notebook that produced them in the first place, so nothing about the artifact format can drift.

The non-test fields of an existing metrics.json (configuration, parameter counts, epochs, timings) are
carried over; only the test-set block and the subset sizes are recomputed.

    python scripts/evaluate_weights.py --stage all
    python scripts/evaluate_weights.py --stage kd --only EfficientNetB0
    python scripts/evaluate_weights.py --stage screening --dry-run
"""
import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SKIP_TAG = "# worker" + ": skip"          # assembled, so this line cannot match itself
TEST_KEYS = ("test_", "auc_per_class", "confusion_matrix", "finished_at")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--stage", default="all", choices=["all", "screening", "tuning", "kd"])
parser.add_argument("--data-dir", default=os.environ.get("OSTEO_DATA_DIR", str(REPO / "SplittedDatasetV2")))
parser.add_argument("--only", default="", help="substring filter on the run name")
parser.add_argument("--dry-run", action="store_true", help="list the checkpoints that would be evaluated")
args = parser.parse_args()

os.environ["OSTEO_DATA_DIR"] = args.data_dir
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["OSTEO_WORKER_MODE"] = "0"

import nbformat                                                        # noqa: E402
import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402

done, failed, skipped = [], [], []


def notebook_namespace(folder, notebook):
    """Execute a notebook's definition cells in its own directory and hand back the namespace."""
    here = pathlib.Path.cwd()
    os.chdir(REPO / folder)
    ns = {"__name__": "__main__"}
    try:
        for cell in nbformat.read(notebook, as_version=4).cells:
            if cell.cell_type != "code":
                continue
            src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
            if SKIP_TAG in src:
                continue
            exec(compile(src, f"<{notebook}>", "exec"), ns)
    finally:
        os.chdir(here)
    return ns


# The cells that build each notebook's _summary tables and comparison figures. They only read the
# metrics.json files back off disk, so re-running them after the evaluation refreshes the summaries
# without touching a single model.
SUMMARY_CELLS = {
    "Teacher_Screening_EarlyStopping.ipynb": ("REFERENCE = dict(", "FIG_SUM = SUMMARY_DIR"),
    "Tuning_GridSearch_And_Baselines.ipynb": ("def collect_results():", "FIG_SUM = SUMMARY_DIR"),
    "KD_MultiTeacher_Distillation.ipynb": ("def collect(", "students = summary[summary.role",
                                           "FIG_SUM = SUMMARY_DIR"),
}


def rebuild_summary(ns, folder, notebook):
    """Re-run the notebook's own summary cells so _summary matches the new evaluations."""
    markers = SUMMARY_CELLS[notebook]
    here = pathlib.Path.cwd()
    os.chdir(REPO / folder)
    try:
        for cell in nbformat.read(notebook, as_version=4).cells:
            if cell.cell_type != "code" or not any(m in cell.source for m in markers):
                continue
            src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
            exec(compile(src, f"<{notebook} summary>", "exec"), ns)
        print(f"  summary    {folder}/results/_summary rebuilt")
    except Exception as err:                                            # noqa: BLE001
        failed.append(f"{folder}/_summary: {type(err).__name__}: {err}")
    finally:
        os.chdir(here)


def carried_info(out, fallback):
    """Everything an existing metrics.json says except the test-set block."""
    path = pathlib.Path(out) / "metrics.json"
    if not path.exists():
        return dict(fallback)
    stored = json.loads(path.read_text())
    info = {k: v for k, v in stored.items() if not k.startswith(TEST_KEYS)}
    info.update({k: v for k, v in fallback.items() if k not in info})
    return info


def wanted(name):
    return args.only.lower() in name.lower()


def announce(kind, name, ckpt):
    print(f"  {kind:<10} {name:<34} {ckpt.name}")


# ════════════════════════════════════════════════════════════════════════════
def evaluate_screening():
    ns = notebook_namespace("teacher", "Teacher_Screening_EarlyStopping.ipynb")
    import keras
    out_root, test_dir = pathlib.Path(ns["OUT_DIR"]), ns["TEST_DIR"]
    os.chdir(REPO / "teacher")
    for name, spec in ns["KERAS_MODELS"].items():
        out = out_root / name
        ckpt = out / "best.keras"
        if not wanted(name):
            continue
        if not ckpt.exists():
            skipped.append(f"screening/{name}: no best.keras"); continue
        announce("screening", name, ckpt)
        if args.dry_run:
            continue
        try:
            gen = ns["make_eval_generator"](test_dir, spec["size"], ns["make_bphe"](spec["pre"]))
            model = keras.models.load_model(ckpt)
            prob = model.predict(gen, verbose=0)
            hist = pd.read_csv(out / "history.csv") if (out / "history.csv").exists() else pd.DataFrame()
            info = carried_info(out, dict(model=name, framework="Keras", family=spec["family"],
                                          input_size=spec["size"], params_total=int(model.count_params())))
            info["n_test"] = int(gen.samples)
            ns["finalize_run"](name, info, hist, gen.classes, prob, gen.filenames)
            done.append(f"screening/{name}")
            del model
            ns["cleanup_backend"]()
        except Exception as err:                                        # noqa: BLE001
            failed.append(f"screening/{name}: {type(err).__name__}: {err}")
    if not args.dry_run and done:
        rebuild_summary(ns, "teacher", "Teacher_Screening_EarlyStopping.ipynb")
    for name in ns.get("TIMM_MODELS", {}):
        if (out_root / name / "head_best.pt").exists():
            skipped.append(f"screening/{name}: timm checkpoint, evaluate with the notebook's torch path")


# ════════════════════════════════════════════════════════════════════════════
def evaluate_tuning():
    ns = notebook_namespace("tuning", "Tuning_GridSearch_And_Baselines.ipynb")
    import keras
    out_root, test_dir = pathlib.Path(ns["OUT_DIR"]), ns["TEST_DIR"]
    os.chdir(REPO / "tuning")
    jobs = [(name, out_root / name / "best" / "best.keras", out_root / name / "best", spec["pre"])
            for name, spec in ns["TUNED_MODELS"].items()]
    jobs.append(("Custom-ENB5", out_root / "Custom-ENB5" / "best.keras", out_root / "Custom-ENB5",
                 ns["efficientnet"].preprocess_input))

    for name, ckpt, out, pre in jobs:
        if not wanted(name):
            continue
        if not ckpt.exists():
            skipped.append(f"tuning/{name}: no {ckpt.name}"); continue
        announce("tuning", name, ckpt)
        if args.dry_run:
            continue
        try:
            stored = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
            batch = int(stored.get("batch_size", ns["GRID"]["batch_size"][0]))
            gen = ns["make_eval_generator"](test_dir, pre, batch)
            model = keras.models.load_model(ckpt)
            prob = model.predict(gen, verbose=0)
            hist = pd.read_csv(out / "history.csv") if (out / "history.csv").exists() else None
            info = carried_info(out, dict(model=name, params_total=int(model.count_params())))
            info["n_test"] = int(gen.samples)
            ns["finalize_run"](stored.get("model", name), out, info, gen.classes, prob, gen.filenames,
                               hist=hist)
            done.append(f"tuning/{name}")
            del model
            ns["cleanup_backend"]()
        except Exception as err:                                        # noqa: BLE001
            failed.append(f"tuning/{name}: {type(err).__name__}: {err}")

    # the classical baselines keep no weights file, and refitting them takes seconds
    for name, fitter in (("RandomForest", "fit_random_forest"), ("XGBoost", "fit_xgboost"),
                         ("CatBoost", "fit_catboost"), ("SVM", "fit_svm"), ("NaiveBayes", "fit_naive_bayes")):
        if not wanted(name) or fitter not in ns:
            continue
        print(f"  {'tuning':<10} {name:<34} refitting (no weights are saved for it)")
        if args.dry_run:
            continue
        try:
            kwargs = dict(img_size=(96, 96), scale="01") if name == "NaiveBayes" else {}
            ns["run_ml_model"](name, ns[fitter], **kwargs)
            done.append(f"tuning/{name}")
        except Exception as err:                                        # noqa: BLE001
            failed.append(f"tuning/{name}: {type(err).__name__}: {err}")
    if not args.dry_run:
        rebuild_summary(ns, "tuning", "Tuning_GridSearch_And_Baselines.ipynb")


# ════════════════════════════════════════════════════════════════════════════
def evaluate_kd():
    ns = notebook_namespace("kd", "KD_MultiTeacher_Distillation.ipynb")
    import keras
    import tensorflow as tf
    out_root, test_dir = pathlib.Path(ns["OUT_DIR"]), ns["TEST_DIR"]
    os.chdir(REPO / "kd")
    ns["CFG"]["SKIP_EXISTING"] = False          # the point of this script is to redo the evaluation

    # 1. the teachers, which the notebook already knows how to evaluate from a saved file
    for key in ns["TEACHER_KEYS"]:
        model_name = ns["TEACHERS"][key]["model"]
        ckpt = pathlib.Path(ns["TEACHER_DIR"]) / model_name / "teacher.keras"
        if not wanted(model_name):
            continue
        if not ckpt.exists():
            skipped.append(f"kd/teachers/{model_name}: no teacher.keras"); continue
        announce("kd teacher", model_name, ckpt)
        if args.dry_run:
            continue
        try:
            ns["RESULTS"][model_name] = ns["prepare_teacher"](key)
            done.append(f"kd/teachers/{model_name}")
        except Exception as err:                                        # noqa: BLE001
            failed.append(f"kd/teachers/{model_name}: {type(err).__name__}: {err}")

    # 2. the fused ensembles, from the logits the teachers just cached
    for set_name, set_keys in ns["TEACHER_SETS"].items():
        label = f"Fused set {set_name} ({'+'.join(set_keys)})"
        ens_dir = pathlib.Path(ns["TEACHER_DIR"]) / f"_ensemble_{set_name}"
        logits = [pathlib.Path(ns["TEACHER_DIR"]) / ns["TEACHERS"][k]["model"] / "test_logits.npy"
                  for k in set_keys]
        if not wanted(label) or args.dry_run:
            continue
        if not all(p.exists() for p in logits):
            skipped.append(f"kd/_ensemble_{set_name}: teacher logits missing"); continue
        print(f"  {'kd fused':<10} {label:<34} mean of cached logits")
        try:
            fused = np.mean([np.load(p) for p in logits], axis=0)
            prob = tf.nn.softmax(fused).numpy()
            gen = ns["make_raw_eval_generator"](test_dir)
            members = [ns["RESULTS"][ns["TEACHERS"][k]["model"]] for k in set_keys]
            info = carried_info(ens_dir, dict(
                model=label, role="ensemble", teacher_set=set_name, teachers=set_keys,
                fusion="mean of logits",
                params_total=int(sum(m["params_total"] for m in members)),
                size_mb=float(sum(m["size_mb"] for m in members)),
                ms_per_image=float(sum(m["ms_per_image"] for m in members))))
            info["n_test"] = int(gen.samples)
            ns["finalize_run"](label, ens_dir, info, gen.classes, prob, gen.filenames)
            done.append(f"kd/_ensemble_{set_name}")
        except Exception as err:                                        # noqa: BLE001
            failed.append(f"kd/_ensemble_{set_name}: {type(err).__name__}: {err}")

    # 3. every student run, including the sweep
    students = [(p.parent.parent.name, p.parent.name, p.parent) for p in out_root.glob("*/*/student.keras")
                if p.parent.parent.name in ns["STUDENTS"]]
    students += [(p.parent.parent.parent.name, p.parent.parent.name, p.parent)
                 for p in out_root.glob("_sweep/*/*/*/student.keras")]
    for student_name, config, out in sorted(students, key=lambda t: str(t[2])):
        label = f"{student_name} [{config}]" + ("" if "_sweep" not in str(out) else f" {out.name}")
        if not wanted(label):
            continue
        announce("kd student", label, out / "student.keras")
        if args.dry_run:
            continue
        try:
            model = keras.models.load_model(out / "student.keras")
            pre = ns["STUDENTS"][student_name]["pre"]
            gen = ns["make_raw_eval_generator"](test_dir)
            probs = []
            for i in range(len(gen)):
                x, _ = gen[i]
                probs.append(tf.nn.softmax(model(pre(x.copy()), training=False)).numpy())
            prob = np.concatenate(probs)
            hist = pd.read_csv(out / "history.csv") if (out / "history.csv").exists() else None
            info = carried_info(out, dict(model=label, student=student_name, config=config, role="student",
                                          params_total=int(model.count_params()),
                                          size_mb=float((out / "student.keras").stat().st_size / 2**20)))
            info["n_test"] = int(gen.samples)
            ns["finalize_run"](info.get("model", label), out, info, gen.classes, prob, gen.filenames,
                               hist=hist)
            done.append(f"kd/{student_name}/{config}")
            del model
            ns["cleanup_backend"]()
        except Exception as err:                                        # noqa: BLE001
            failed.append(f"kd/{student_name}/{config}: {type(err).__name__}: {err}")
    if not args.dry_run:
        rebuild_summary(ns, "kd", "KD_MultiTeacher_Distillation.ipynb")


# ════════════════════════════════════════════════════════════════════════════
print(f"data: {args.data_dir}\n")
for stage, fn in (("screening", evaluate_screening), ("tuning", evaluate_tuning), ("kd", evaluate_kd)):
    if args.stage in ("all", stage):
        print(f"[{stage}]")
        fn()
        print()

print(f"evaluated : {len(done)}")
for s in skipped:
    print(f"  skipped : {s}")
for f in failed:
    print(f"  FAILED  : {f}")
sys.exit(1 if failed else 0)
