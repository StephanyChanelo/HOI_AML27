from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import yaml

class DexYCB_Project(Dataset):

    def __init__(
        self,
        root,
        subjects=None,
        sequences=None,
        transform=None,
        return_segmentation=True,
    ):
        """
        PyTorch Dataset for DexYCB.

        Args:
            root:
                Path to DexYCB root directory.
        """

        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"DexYCB directory does not exist: {self.root}"
            )

        self.samples = []

        self._build_index(
            subjects=subjects,
            sequences=sequences,
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No DexYCB samples found in {self.root}"
            )
    
    def _build_index(self, subjects=None, sequences=None):

        if subjects is None:
            subjects = sorted(
                [
                    p for p in self.root.iterdir()
                    if p.is_dir()
                    and p.name.startswith("2020")
                ]
            )
        else:
            subjects = [
                self.root / s
                for s in subjects
            ]

        for subject_dir in subjects:

            if not subject_dir.exists():
                continue

            sequence_dirs = sorted(
                [
                    p for p in subject_dir.iterdir()
                    if p.is_dir()
                ]
            )

            if sequences is not None:
                sequence_dirs = [
                    p for p in sequence_dirs
                    if p.name in sequences
                ]

            for sequence_dir in sequence_dirs:

                camera_dirs = sorted(
                    [
                        p for p in sequence_dir.iterdir()
                        if p.is_dir()
                    ]
                )

                for camera_dir in camera_dirs:

                    color_files = sorted(
                        camera_dir.glob("color_*.jpg")
                    )

                    for color_file in color_files:

                        frame_name = color_file.stem
                        frame_id = frame_name.split("_")[-1]

                        depth_file = (
                            camera_dir
                            / f"aligned_depth_to_color_{frame_id}.png"
                        )

                        label_file = (
                            camera_dir
                            / f"labels_{frame_id}.npz"
                        )

                        if not depth_file.exists():
                            continue

                        if not label_file.exists():
                            continue

                        self.samples.append(
                            {
                                "color_file": color_file,
                                "depth_file": depth_file,
                                #"label_file": label_file,
                                #"subject": subject_dir.name,
                                #"sequence": sequence_dir.name,
                                #"camera": camera_dir.name,
                                #"frame_id": int(frame_id),
                            }
                        )


        #...
        #...
        #...

    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]

        # --------------------------------------------------
        # RGB & Depth
        # --------------------------------------------------
        rgb = cv2.imread(str(sample_info["color_file"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Could not read {sample_info['color_file']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(sample_info["depth_file"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Could not read {sample_info['depth_file']}")
        depth = depth.astype(np.float32)
        #...
        #...
        #...

        # --------------------------------------------------
        # Tensors
        # --------------------------------------------------
        rgb_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(depth.copy()).float()
        #...
        #...
        #...

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
        }