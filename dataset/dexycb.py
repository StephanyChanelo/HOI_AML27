from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import yaml


class DexYCB(Dataset):

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
        self.return_segmentation = return_segmentation

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

    def _load_meta(self, sequence_dir):

        meta_file = sequence_dir / "meta.yml"

        if not meta_file.exists():
            raise FileNotFoundError(
                f"Missing meta.yml: {meta_file}"
            )

        with open(meta_file, "r") as f:
            meta = yaml.safe_load(f)

        return meta
    
    def _load_camera_calibration(self, camera):

        calib_file = (
            self.root
            / "calibration"
            / "intrinsics"
            / f"{camera}_640x480.yml"
        )

        if not calib_file.exists():
            raise FileNotFoundError(
                f"Missing calibration file: {calib_file}"
            )

        with open(calib_file, "r") as f:
            calib = yaml.load(f, Loader=yaml.FullLoader)

        color = calib["color"]
        depth = calib["depth"]

        color_K = np.array([
            [color["fx"], 0.0, color["ppx"]],
            [0.0, color["fy"], color["ppy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        depth_K = np.array([
            [depth["fx"], 0.0, depth["ppx"]],
            [0.0, depth["fy"], depth["ppy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        return color_K, depth_K
    
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
                                "label_file": label_file,
                                "subject": subject_dir.name,
                                "sequence": sequence_dir.name,
                                "camera": camera_dir.name,
                                "frame_id": int(frame_id),
                            }
                        )

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

        # --------------------------------------------------
        # Labels & Hand Parameters
        # --------------------------------------------------
        labels = np.load(sample_info["label_file"], allow_pickle=True)
        segmentation = labels["seg"]
        joint_2d = labels["joint_2d"][0].astype(np.float32)
        joint_3d = labels["joint_3d"][0].astype(np.float32)

        pose_m = labels["pose_m"][0].astype(np.float32)
        mano_pose = pose_m[:48]
        mano_trans = pose_m[48:51]

        # --------------------------------------------------
        # Active Object Selection (Fixed 3x4 Shape)
        # --------------------------------------------------
        meta = self._load_meta(
            self.root / sample_info["subject"] / sample_info["sequence"]
        )
        object_ids = np.array(meta["ycb_ids"], dtype=np.int64)
        pose_y = labels["pose_y"].astype(np.float32)

        # 1. Map grasped object YCB ID to index
        grasp_ycb_id = meta.get("ycb_grasp", 0)
        target_idx = -1
        if grasp_ycb_id != 0 and grasp_ycb_id in object_ids:
            target_idx = int(np.where(object_ids == grasp_ycb_id)[0][0])

        # 2. Extract active pose safely
        if target_idx != -1 and target_idx < len(pose_y):
            active_obj_id = object_ids[target_idx]
            raw_pose = pose_y[target_idx]
        elif len(pose_y) > 0 and len(object_ids) > 0:
            hand_wrist = joint_3d[0]
            obj_translations = pose_y[:, :3, 3]
            closest_idx = np.argmin(np.linalg.norm(obj_translations - hand_wrist, axis=1))
            active_obj_id = object_ids[closest_idx]
            raw_pose = pose_y[closest_idx]
        else:
            active_obj_id = 0
            raw_pose = np.eye(4, dtype=np.float32)

        active_obj_pose = np.ascontiguousarray(raw_pose[:3, :4], dtype=np.float32)

        # --------------------------------------------------
        # Tensors
        # --------------------------------------------------
        rgb_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(depth.copy()).float()
        segmentation_tensor = torch.from_numpy(segmentation.copy()).long()

        color_K, depth_K = self._load_camera_calibration(sample_info["camera"])

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "segmentation": segmentation_tensor,
            "intrinsics": torch.from_numpy(color_K),
            "depth_intrinsics": torch.from_numpy(depth_K),
            "joint_2d": torch.from_numpy(joint_2d),
            "joint_3d": torch.from_numpy(joint_3d),
            "mano_pose": torch.from_numpy(mano_pose),
            "mano_trans": torch.from_numpy(mano_trans),
            # STRICT SINGLE-OBJECT OUTPUTS:
            "active_object_id": torch.tensor(active_obj_id, dtype=torch.long),
            "active_object_pose": torch.from_numpy(active_obj_pose),  # Always (3, 4)
            "sample_id": index,
            "frame_id": sample_info["frame_id"],
        }