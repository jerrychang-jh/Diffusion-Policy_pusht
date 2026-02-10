import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="local/pusht_cchi_v1_lerobotv3",
    root="Dataset/lerobot_v3/pusht_cchi_v1_lerobotv3",
)

# ============= Test after patch_video_meta =============
print("len:", len(ds))

x = ds[0]

print(x.keys())
print("img:", x["observation.images.img"].shape, x["observation.images.img"].dtype)
print("action:", x["action"].shape)
# ======================================================

# ============= Test after patch_episodes_global_indices =============
print(ds.meta.episodes.column_names[:50])
print("dataset_from_index" in ds.meta.episodes.column_names)        # Should be True
print("dataset_to_index" in ds.meta.episodes.column_names)          # Should be True
# ======================================================

print("dataset keys:", sorted([k for k in x.keys() if k.startswith("observation")]))
for k in sorted([k for k in x.keys() if k.startswith("observation") and hasattr(x[k], "shape")]):
    v = np.asarray(x[k])
    print(k, v.shape, v.dtype)