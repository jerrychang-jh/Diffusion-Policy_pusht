# compute_and_write_stats.py
import json
from pathlib import Path
import numpy as np
import math
import shutil
import sys
from collections import defaultdict
from tqdm import trange

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as e:
    print("Failed to import LeRobotDataset:", e)
    raise

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

ROOT = Path("Dataset/lerobot_v3/pusht_cchi_v1_lerobotv3")
REPO_ID = "local/pust_cchi_v1_lerobotv3"
N_SAMPLE_EPISODES = 300        # number of episodes to sample
N_FRAMES_PER_EP = 5            # frames per episode to sample for images
VIDEO_KEY_PREFIX = "observation.images.img"

print("Loading dataset...")
ds = LeRobotDataset(repo_id="local/pust_cchi_v1_lerobotv3", root=str(ROOT))

n_total = len(ds)
n_samples = min(N_SAMPLE_EPISODES, n_total)
print(f"Dataset length: {n_total}. Sampling {n_samples} episodes.")

if n_samples <= 0:
    print("No episodes found. Exiting.")
    sys.exit(1)

indices = np.linspace(0, n_total-1, n_samples, dtype=int).tolist()

class RunningStat:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None

    def update_batch(self, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return
        if arr.ndim > 1:
            flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
        else:
            flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
        batch_n = flat.shape[0]
        batch_mean = flat.mean(axis=0)
        batch_M2 = ((flat - batch_mean)**2).sum(axis=0)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            # combine
            delta = batch_mean - self.mean
            total_n = self.n + batch_n
            self.M2 = self.M2 + batch_M2 + (delta**2) * (self.n * batch_n / total_n)
            self.mean = (self.mean * self.n + batch_mean * batch_n) / total_n
            self.n = total_n

    def finalize(self):
        if self.n < 2:
            var = np.zeros_like(self.mean)
        else:
            var = self.M2 / self.n
        std = np.sqrt(var)
        return self.mean, std

sample_item = ds[indices[0]]
available_keys = list(sample_item.keys())
print("Available keys in one sample:", available_keys)

numeric_keys = []
image_keys = []
for k in available_keys:
    v = sample_item[k]
    if hasattr(v, "shape"):
        if v.ndim >= 3 and (k.startswith("observation") or "image" in k or "img" in k):
            image_keys.append(k)
        else:
            numeric_keys.append(k)

print("Numeric keys:", numeric_keys)
print("Image keys:", image_keys)

stats = {}
# compute numeric keys stats
for k in numeric_keys:
    rs = RunningStat()
    for idx in indices:
        item = ds[idx]
        arr = item[k]
        a = np.asarray(arr)
        if a.ndim == 0:
            a = a.reshape(1)
        elif a.ndim >= 1 and a.shape[0] != 1 and len(indices) == 1:
            pass
        if a.ndim == 1:
            a = a.reshape(1, -1)
        elif a.ndim >= 2 and a.shape[0] != len(indices):
            a = a.reshape(1, -1)
        rs.update_batch(a)
    mean, std = rs.finalize()
    stats[k] = {
        "mean": mean.tolist() if mean is not None else None,
        "std": std.tolist() if std is not None else None,
    }
    print(f"Computed stats for {k}: mean_len={len(stats[k]['mean']) if stats[k]['mean'] is not None else 0}")

for k in image_keys:
    rs = RunningStat()
    for idx in indices:
        item = ds[idx]
        try:
            img = item[k]
            arr = np.asarray(img)
            if arr.ndim == 4:
                frames_idx = np.linspace(0, arr.shape[0]-1, min(arr.shape[0], 3), dtype=int)
                sample_frames = arr[frames_idx]
            elif arr.ndim == 3:
                sample_frames = arr[np.newaxis, ...]
            else:
                sample_frames = arr.reshape(1, *arr.shape)
            # flatten sample axis
            # convert uint8 -> [0,1]
            if sample_frames.dtype == np.uint8:
                sample_frames = sample_frames.astype(np.float32) / 255.0
            rs.update_batch(sample_frames)
        except Exception as e:
            if HAVE_CV2:
                try:
                    ep = item.get("episode_index", None) or item.get("ep_idx", None)
                except Exception:
                    ep = None
                continue
            else:
                continue
    mean, std = rs.finalize()
    if mean is None:
        print(f"Could not compute image stats for {k}. Falling back to mean=0.5,std=0.5")
        stats[k] = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    else:
        mn = np.asarray(mean)
        sd = np.asarray(std)
        sample_img = np.asarray(sample_item[image_keys[0]])
        if sample_img.ndim == 4:
            # T,H,W,C
            _, H, W, C = sample_img.shape
        elif sample_img.ndim == 3:
            H, W, C = sample_img.shape
        else:
            H=W=C=1
        if mn.size == H*W*C and C>1:
            mn = mn.reshape(-1, C).mean(axis=0)
            sd = sd.reshape(-1, C).mean(axis=0)
        else:
            mn = mn[:3]
            sd = sd[:3]
        stats[k] = {"mean": [float(x) for x in mn.tolist()], "std": [float(x) for x in sd.tolist()]}
        print(f"Image stats for {k}: mean(len)={len(stats[k]['mean'])}")

out = ROOT / "meta" / "stats.json"
bak = out.with_suffix(".json.bak")
if out.exists() and not bak.exists():
    shutil.copy2(out, bak)
with open(out, "w") as f:
    json.dump(stats, f, indent=2)
print("Wrote stats to:", out)
print("Done.")
