import torch


def dexycb_collate(batch):

    output = {}

    # --------------------------------------------------
    # Fixed-size tensors
    # --------------------------------------------------

    fixed_keys = [
        "rgb",
        "depth",
        "segmentation",
        "intrinsics",
        "joint_2d",
        "joint_3d",
        "mano_pose",
        "mano_trans",
    ]

    for key in fixed_keys:
        output[key] = torch.stack(
            [
                sample[key]
                for sample in batch
            ],
            dim=0,
        )

    # --------------------------------------------------
    # Object poses
    # --------------------------------------------------

    max_objects = max(
        sample["object_pose"].shape[0]
        for sample in batch
    )

    batch_size = len(batch)

    object_pose = torch.zeros(
        batch_size,
        max_objects,
        3,
        4,
        dtype=torch.float32,
    )

    object_valid = torch.zeros(
        batch_size,
        max_objects,
        dtype=torch.bool,
    )

    for i, sample in enumerate(batch):

        n = sample["object_pose"].shape[0]

        object_pose[
            i,
            :n
        ] = sample["object_pose"]

        object_valid[
            i,
            :n
        ] = True

    output["object_pose"] = object_pose

    output["object_valid"] = object_valid

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    output["sample_id"] = [
        x["sample_id"]
        for x in batch
    ]

    output["frame_id"] = [
        x["frame_id"]
        for x in batch
    ]

    output["subject"] = [
        x["subject"]
        for x in batch
    ]

    output["sequence"] = [
        x["sequence"]
        for x in batch
    ]

    output["camera"] = [
        x["camera"]
        for x in batch
    ]

    return output