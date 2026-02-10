import json
from pathlib import Path
import numpy as np
import shutil

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("Dataset/lerobot_v3/pusht_cchi_v1_lerobotv3")
STATS_PATH = ROOT / "meta" / "stats.json"

ds = LeRobotDataset(repo_id="local/pusht_cchi_v1_lerobotv3", root=str(ROOT))

stats = {}
if STATS_PATH.exists():
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)

# sample indices
n = len(ds)
N_SAMPLE = min(3000, n)   # adjust up for better coverage
idxs = np.linspace(0, n-1, N_SAMPLE, dtype=int)

# discover keys
sample = ds[int(idxs[0])]
keys = [k for k, v in sample.items() if hasattr(v, "shape")]

def update_minmax(key, arr, cur):
    a = np.asarray(arr)

    # handle images (uint8 -> [0,1])
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0

    # reduce over all dims -> scalar min/max
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))

    if cur.get("min") is None:
        cur["min"] = mn
        cur["max"] = mx
    else:
        cur["min"] = float(min(cur["min"], mn))
        cur["max"] = float(max(cur["max"], mx))

for k in keys:
    cur = stats.get(k, {})
    cur.setdefault("mean", cur.get("mean"))  # keep whatever exists
    cur.setdefault("std", cur.get("std"))
    cur.setdefault("min", None)
    cur.setdefault("max", None)

    for i in idxs:
        item = ds[int(i)]
        try:
            update_minmax(k, item[k], cur)
        except Exception:
            continue

    stats[k] = cur
    print(f"{k}: min={stats[k]['min']} max={stats[k]['max']}")

# backup + write
bak = STATS_PATH.with_suffix(".json.bak_minmax")
if STATS_PATH.exists() and not bak.exists():
    shutil.copy2(STATS_PATH, bak)

with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print("Wrote", STATS_PATH)
