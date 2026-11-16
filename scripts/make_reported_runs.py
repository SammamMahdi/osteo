"""Write reported_runs.json: every run that appears in the paper, with the configuration it used.

"Reported" is defined exactly as the user put it: a run whose confusion matrix was produced, which is
every result folder holding figures/confusion_matrix.pdf. This script reads the existing results tree to
capture those configurations before it is rebuilt on the corrected split, so the re-run reproduces the
same set of models under the same settings.

    python scripts/make_reported_runs.py
"""
import json
import pathlib

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "scripts" / "reported_runs.json"


def folders_with_confusion(root):
    return sorted(p.parent.parent for p in root.glob("**/figures/confusion_matrix.pdf"))


# ── Stage II: every screened architecture ───────────────────────────────────
screening = [p.name for p in folders_with_confusion(REPO / "teacher" / "results")]

# ── Stage III: the winning configuration of each architecture ───────────────
grid_best = pd.read_csv(REPO / "tuning" / "results" / "_summary" / "grid_best.csv")
tuning_dl = [
    dict(model=r.model, learning_rate=float(r.learning_rate), dropout_rate=float(r.dropout_rate),
         optimizer=str(r.optimizer), batch_size=int(r.batch_size))
    for r in grid_best.itertuples()
]
custom = json.loads((REPO / "tuning" / "results" / "Custom-ENB5" / "metrics.json").read_text())
tuning_custom = dict(model="Custom-ENB5", learning_rate=float(custom["learning_rate"]),
                     dropout_rate=float(custom["dropout_rate"]), optimizer=str(custom["optimizer"]),
                     batch_size=int(custom["batch_size"]))
tuning_ml = [p.name for p in folders_with_confusion(REPO / "tuning" / "results")
             if p.name in ("RandomForest", "XGBoost", "CatBoost", "SVM", "NaiveBayes")]

# ── Stage IV: teachers, both ensembles, every student run and the sweep ─────
kd_root = REPO / "kd" / "results"
students, configs = [], []
for p in folders_with_confusion(kd_root):
    rel = p.relative_to(kd_root)
    if rel.parts[0] in ("teachers", "_sweep"):
        continue
    student, config = rel.parts[0], rel.parts[1]
    if student not in students:
        students.append(student)
    if config not in configs:
        configs.append(config)
sweep = sorted(p.name for p in folders_with_confusion(kd_root) if "_sweep" in str(p))
teachers = sorted(p.name for p in folders_with_confusion(kd_root / "teachers"))
final = json.loads((kd_root / "_sweep" / "EfficientNetB0" / "ENB5+R50" / "T3_a0.5" / "metrics.json").read_text())

spec = dict(
    note="Runs reported in the paper: every result folder that produced a confusion matrix.",
    screening=screening,
    tuning_dl=tuning_dl,
    tuning_custom=tuning_custom,
    tuning_ml=tuning_ml,
    kd=dict(students=students, configs=configs, teachers=teachers, sweep_points=sweep,
            temperature=float(final["temperature"]), alpha_main=0.3,
            sweep_T=[2.0, 3.0, 5.0], sweep_alpha=[0.1, 0.3, 0.5],
            final=dict(student="EfficientNetB0", config="ENB5+R50", T=3.0, alpha=0.5)),
)
OUT.write_text(json.dumps(spec, indent=1))

print(f"wrote {OUT}")
print(f"  screening architectures : {len(screening)}")
print(f"  tuned deep models       : {len(tuning_dl)} + Custom-ENB5")
print(f"  classical baselines     : {len(tuning_ml)}")
print(f"  distillation runs       : {len(students)} students x {len(configs)} configurations "
      f"= {len(students) * len(configs)}")
print(f"  sweep points            : {len(sweep)}")
print(f"  teacher evaluations     : {len(teachers)}")
