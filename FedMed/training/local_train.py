import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    EnsureTyped,
)

from model.unet3d import create_model


def get_dataset(hospital_id):
    data_dir = f"data/{hospital_id}"

    data = []

    for i in range(3):
        image_path = os.path.join(data_dir, f"image_{i}.nii.gz")
        mask_path = os.path.join(data_dir, f"mask_{i}.nii.gz")

        data.append({
            "image": image_path,
            "label": mask_path,
        })

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    return Dataset(data=data, transform=transforms)


def train_local(hospital_id, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nTraining {hospital_id}")
    print(f"Device: {device}")

    dataset = get_dataset(hospital_id)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    model = create_model().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            labels = labels.squeeze(1).long()

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    print(f"{hospital_id} local training complete.")

    return model


if __name__ == "__main__":
    train_local("hospital1", epochs=1)