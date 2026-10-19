import argparse
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from dataset.dexycb import DexYCB


# ============================================================
# SETTINGS & ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser(description="DexYCB Dataset Inspection")
parser.add_argument(
    "--dataset_root", 
    type=str, 
    default="/data/datasets/DexYCB", 
    help="Path to the DexYCB dataset directory"
)
parser.add_argument(
    "--index",
    type=int,
    default=100,
    help="Index of the dataset sample to inspect and plot"
)
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
frame = args.index

# ============================================================
# DATASET
# ============================================================

dataset = DexYCB(
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


# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot RGB Image
axes[0].imshow(rgb)
axes[0].set_title(f"RGB (Index {frame})")
axes[0].axis("off")

# Plot Depth Map with Colorbar
im = axes[1].imshow(depth, cmap="viridis", vmin=depth_vmin, vmax=depth_vmax)
axes[1].set_title(f"Depth Map (Index {frame})")
axes[1].axis("off")

# Add colorbar for depth visualization
cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label("Depth (mm)")

plt.tight_layout()
output_filename = "dexycb_sample.png"
plt.savefig(output_filename, bbox_inches="tight", dpi=150)
plt.close()

print(f"\nSaved visualization plot to: {output_filename}")
