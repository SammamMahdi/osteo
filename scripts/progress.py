"""Report how much of the re-run is finished, and what is left.

A run counts as finished when its metrics.json exists, which is the same rule the notebooks use to skip
work, so this says exactly what a restart would skip and what it would train.

    python scripts/progress.py                      # the results_v2 tree
    python scripts/progress.py --results results    # or any other
"""
import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SPEC = json.loads((REPO / "scripts" / "reported_runs.json").read_text())

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--results", default="results_v2", help="name of the results directory in each stage")
parser.add_argument("--verbose", action="store_true", help="name every finished run, not just the missing")
args = parser.parse_args()
R = args.results


def check(label, paths):
    """paths: {run name -> metrics.json path}"""
    done = [n for n, p in paths.items() if p.exists()]
    missing = [n for n, p in paths.items() if not p.exists()]
    mark = "done" if not missing else f"{len(done)}/{len(paths)}"
    print(f"  {label:<26} {mark:>7}")
    if missing:
        print(f"    left: {', '.join(missing)}")
    elif args.verbose:
        print(f"    {', '.join(done)}")
    return len(done), len(paths)


total_done = total_all = 0


def add(pair):
    global total_done, total_all
    total_done += pair[0]
    total_all += pair[1]


print(f"results directory: <stage>/{R}\n")

print("Stage II - screening")
add(check("architectures", {m: REPO / "teacher" / R / m / "metrics.json" for m in SPEC["screening"]}))

print("\nStage III - tuning")
add(check("tuned models", {r["model"]: REPO / "tuning" / R / r["model"] / "best" / "metrics.json"
                           for r in SPEC["tuning_dl"]}))
add(check("Custom-ENB5", {"Custom-ENB5": REPO / "tuning" / R / "Custom-ENB5" / "metrics.json"}))
add(check("classical baselines", {m: REPO / "tuning" / R / m / "metrics.json" for m in SPEC["tuning_ml"]}))

print("\nStage IV - distillation")
kd = REPO / "kd" / R
add(check("teachers and ensembles", {t: kd / "teachers" / t / "metrics.json" for t in SPEC["kd"]["teachers"]}))
students = {f"{s} [{c}]": kd / s / c / "metrics.json"
            for s in SPEC["kd"]["students"] for c in SPEC["kd"]["configs"]}
add(check("student runs", students))
add(check("temperature sweep", {t: kd / "_sweep" / "EfficientNetB0" / "ENB5+R50" / t / "metrics.json"
                                for t in SPEC["kd"]["sweep_points"]}))

print(f"\noverall: {total_done}/{total_all} runs finished")
if total_done < total_all:
    print("resume with:")
    print(f"  OSTEO_RESULTS_NAME={R} ./scripts/rerun_reported.sh")
