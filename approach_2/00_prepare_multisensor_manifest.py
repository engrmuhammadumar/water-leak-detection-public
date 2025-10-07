# 00_prepare_multisensor_manifest.py
from pathlib import Path
import pandas as pd
import json

CLASSES = [
    "No-leak",
    "Orifice Leak",
    "Gasket Leak",
    "Longitudinal Crack",
    "Circumferential Crack",
]

ACC_DIR   = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\processed\Accelerometer\Looped")
PRESS_DIR = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\processed\Dynamic Pressure Sensor\Looped")
HYDR_DIR  = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\processed\Hydrophones\Looped")

OUT_ROOT = Path(r"E:\Upwork Project\AI_Leak_Detection_Project\data\csv_multisensor")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def list_by_class(root: Path, cls: str):
    paths = sorted((root/cls).rglob("*.csv"))
    stems = {p.stem: p for p in paths}
    return paths, stems

def build_pairs():
    rows = []
    for cls in CLASSES:
        acc_paths, acc_stems = list_by_class(ACC_DIR, cls)
        pr_paths,  pr_stems  = list_by_class(PRESS_DIR, cls)
        hy_paths,  hy_stems  = list_by_class(HYDR_DIR, cls)

        # match by stem if possible
        common = set(acc_stems).intersection(pr_stems).intersection(hy_stems)
        if common:
            for s in sorted(common):
                rows.append({
                    "acc": str(acc_stems[s]),
                    "press": str(pr_stems[s]),
                    "hydro": str(hy_stems[s]),
                    "label": cls
                })
        else:
            # fallback: warn and align by index
            n = min(len(acc_paths), len(pr_paths), len(hy_paths))
            print(f"[WARN] No common stems for class '{cls}'. Falling back to index match n={n}.")
            for i in range(n):
                rows.append({
                    "acc": str(acc_paths[i]),
                    "press": str(pr_paths[i]),
                    "hydro": str(hy_paths[i]),
                    "label": cls
                })
    return pd.DataFrame(rows)

def stratified_split(df: pd.DataFrame, val_ratio=0.15, test_ratio=0.15, seed=42):
    dfs = []
    for cls in CLASSES:
        sub = df[df.label==cls].sample(frac=1.0, random_state=seed)
        n = len(sub)
        n_test = int(round(n*test_ratio))
        n_val  = int(round(n*val_ratio))
        n_train = n - n_val - n_test
        train = sub.iloc[:n_train].copy(); train["split"]="train"
        val   = sub.iloc[n_train:n_train+n_val].copy(); val["split"]="val"
        test  = sub.iloc[n_train+n_val:].copy(); test["split"]="test"
        dfs += [train,val,test]
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    df = build_pairs()
    print("Total paired samples:", len(df))
    splits = stratified_split(df, 0.15, 0.15)
    splits.to_csv(OUT_ROOT/"all.csv", index=False)
    splits[splits.split=="train"].to_csv(OUT_ROOT/"train.csv", index=False)
    splits[splits.split=="val"].to_csv(OUT_ROOT/"val.csv", index=False)
    splits[splits.split=="test"].to_csv(OUT_ROOT/"test.csv", index=False)
    classes = {c:i for i,c in enumerate(CLASSES)}
    with open(OUT_ROOT/"classes.json","w",encoding="utf-8") as f:
        json.dump(classes,f,ensure_ascii=False,indent=2)
    print("Saved:", OUT_ROOT)
