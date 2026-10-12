import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from dataset.dexycb_project import DexYCB_Project


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = "/data/datasets/DexYCB"

frame = 100

# ============================================================
# DATASET
# ============================================================

dataset = DexYCB_Project(
    root=DATASET_ROOT
)

print("Dataset size:", len(dataset))

sample = dataset[frame]

# ============================================================
# RGB
# ============================================================

rgb = sample["rgb"]

if torch.is_tensor(rgb):
    rgb = rgb.permute(1, 2, 0).cpu().numpy()

rgb = np.asarray(rgb)

if rgb.max() > 1.0:
    rgb = rgb / 255.0

rgb = np.clip(rgb, 0, 1)

H, W = rgb.shape[:2]

print("\nRGB shape:", rgb.shape)


# ============================================================
# DEPTH
# ============================================================

depth = sample["depth"]

if torch.is_tensor(depth):
    depth = depth.cpu().numpy()

depth = np.asarray(depth)

if depth.ndim == 3:
    depth = depth[0]

depth = depth.astype(np.float32)

valid_depth = depth[depth > 0]

if len(valid_depth) > 0:

    depth_vmin = np.percentile(
        valid_depth,
        2
    )

    depth_vmax = np.percentile(
        valid_depth,
        98
    )

else:

    depth_vmin = 0
    depth_vmax = 1



print("\nDepth shape:", depth.shape)
print("  dtype  :", depth.dtype)
print("  min    :", depth.min(), "mm")
print("  max    :", depth.max(), "mm")

if len(valid_depth) > 0:
    print("  median :", np.median(valid_depth), "mm")

