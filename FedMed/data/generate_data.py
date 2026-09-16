import os

import nibabel as nib
import numpy as np


def create_sample(hospital_id, sample_id):
    os.makedirs(f"data/{hospital_id}", exist_ok=True)

    shape = (64, 64, 64)

    # Background medical-image-like noise.
    image = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=shape,
    ).astype(np.float32)

    # Create a spherical foreground region.
    mask = np.zeros(shape, dtype=np.uint8)

    center = np.array([32, 32, 32])
    radius = 10

    x, y, z = np.indices(shape)

    distance = np.sqrt(
        (x - center[0]) ** 2
        + (y - center[1]) ** 2
        + (z - center[2]) ** 2
    )

    foreground = distance < radius
    mask[foreground] = 1

    # Add a learnable intensity signal inside the foreground.
    foreground_signal = np.random.normal(
        loc=3.0,
        scale=0.5,
        size=shape,
    ).astype(np.float32)

    image[foreground] += foreground_signal[foreground]

    image_path = f"data/{hospital_id}/image_{sample_id}.nii.gz"
    mask_path = f"data/{hospital_id}/mask_{sample_id}.nii.gz"

    nib.save(
        nib.Nifti1Image(image, np.eye(4)),
        image_path,
    )

    nib.save(
        nib.Nifti1Image(mask, np.eye(4)),
        mask_path,
    )

    print(f"Created: {image_path}")
    print(f"Created: {mask_path}")


if __name__ == "__main__":
    for hospital in ["hospital1", "hospital2", "hospital3"]:
        for sample in range(3):
            create_sample(hospital, sample)