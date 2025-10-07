# 00_build_manifest_multisensor_2d.py
from pathlib import Path
import csv
import random
from collections import defaultdict

# ====== CONFIG ======
CLASSES = [
    "No-leak",
    "Orifice Leak",
    "Gasket Leak",
    "Longitudinal Crack",
    "Circumferential Crack",
]

SENSOR_DIRS = {
    "Accelerometer": Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\processed\Accelerometer\Looped"),
    "Dynamic Pressure Sensor": Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\processed\Dynamic Pressure Sensor\Looped"),
    "Hydrophones": Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\processed\Hydrophones\Looped"),
}

OUT_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\csv_multisensor_2d")
SPLIT = (0.7, 0.15, 0.15)  # train/val/test
SEED = 1337
# ====== /CONFIG ======

def list_csvs_by_class(sensor_root: Path):
    by_cls = defaultdict(list)
    for cls in CLASSES:
        cdir = sensor_root / cls
        if not cdir.exists():
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        files = sorted([p for p in cdir.glob("*.csv")])
        by_cls[cls] = files
    return by_cls

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    acc = list_csvs_by_class(SENSOR_DIRS["Accelerometer"])
    prs = list_csvs_by_class(SENSOR_DIRS["Dynamic Pressure Sensor"])
    hyd = list_csvs_by_class(SENSOR_DIRS["Hydrophones"])

    rows = []
    for cls in CLASSES:
        n = min(len(acc[cls]), len(prs[cls]), len(hyd[cls]))
        if n == 0:
            raise RuntimeError(f"No files for class {cls}")
        for i in range(n):
            rows.append({
                "class": cls,
                "acc": str(acc[cls][i]),
                "prs": str(prs[cls][i]),
                "hyd": str(hyd[cls][i]),
            })

    # stratified-ish split by class (index-based)
    train, val, test = [], [], []
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)

    for cls in CLASSES:
        items = by_cls[cls]
        random.shuffle(items)
        n = len(items)
        n_tr = int(SPLIT[0]*n)
        n_v  = int(SPLIT[1]*n)
        train += items[:n_tr]
        val   += items[n_tr:n_tr+n_v]
        test  += items[n_tr+n_v:]

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class","acc","prs","hyd"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(OUT_ROOT/"manifest_train.csv", train)
    write_csv(OUT_ROOT/"manifest_val.csv", val)
    write_csv(OUT_ROOT/"manifest_test.csv", test)

    print(f"Saved to: {OUT_ROOT}")
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")

if __name__ == "__main__":
    main()
