import json, subprocess, time, statistics, os
os.chdir("/home/user/hackAthon-gUide")
def log_run(cfg, fold_scores, note="", path="/tmp/claude-0/-home-user-hackAthon-gUide/36bb1483-bce8-5c6f-a8a8-7a8d11515bd7/scratchpad/mlgap/runs.jsonl"):
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
    rec = {"ts": time.strftime("%F %T"), "commit": commit, "dirty": dirty, **cfg,
           "folds": fold_scores, "cv_mean": statistics.mean(fold_scores),
           "cv_std": statistics.pstdev(fold_scores), "note": note}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
log_run({"model":"lgbm","lr":0.05},[0.26,0.38,0.2,0.22,0.3],"test")
import pandas as pd; print(pd.read_json("/tmp/claude-0/-home-user-hackAthon-gUide/36bb1483-bce8-5c6f-a8a8-7a8d11515bd7/scratchpad/mlgap/runs.jsonl", lines=True).sort_values("cv_mean").tail(1).T)
