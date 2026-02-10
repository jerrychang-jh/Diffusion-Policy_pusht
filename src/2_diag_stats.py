# diag_stats.py
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json

ROOT = "Dataset/lerobot_v3/pusht_cchi_v1_lerobotv3"
ds = LeRobotDataset(repo_id="local/pusht_cchi_v1_lerobotv3", root=ROOT)

print("Loaded dataset. Inspecting meta.stats ...")
meta_stats = getattr(ds.meta, "stats", None)
print("type(meta.stats) =", type(meta_stats))
if meta_stats is None:
    print("meta.stats is None")
else:
    # meta_stats may be a dict-like; attempt to inspect keys & values
    try:
        keys = list(meta_stats.keys())
    except Exception as e:
        print("Could not list keys:", e)
        keys = []
    print("keys:", keys)
    for k in keys:
        v = meta_stats.get(k, None)
        print(f" - {k}: type={type(v)}; is None? {v is None}")
