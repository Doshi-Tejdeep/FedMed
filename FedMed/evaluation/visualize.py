import os

import torch
import matplotlib.pyplot as plt

from model.unet3d import create_model
from training.local_train import get_dataset


MODEL_PATH = "models/global_model.pth"
OUTPUT_DIR = "outputs/predictions"


def visualize_hospital(model, hospital_id, device):
    print(f"\nProcessing {hospital_id}...")

    dataset = get_dataset(hospital_id)

    # Use the first sample from the hospital
    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)
    label = sample["label"].squeeze(0)

    model.eval()

    with torch.no_grad():
        output = model(image)

        # Convert model output to predicted segmentation
        prediction = torch.argmax(output, dim=1)

    prediction = prediction.squeeze(0).cpu()

    # Select the middle slice of the 3D volume
    depth = image.shape[2]
    slice_index = depth // 2

    image_slice = image[0, 0, slice_index].cpu()
    label_slice = label[slice_index].cpu()
    prediction_slice = prediction[slice_index]

    # Calculate Dice for this visualization sample
    intersection = (
        (prediction == 1) & (label == 1)
    ).sum().float()

    prediction_sum = (prediction == 1).sum().float()
    label_sum = (label == 1).sum().float()

    dice = (
        (2.0 * intersection) /
        (prediction_sum + label_sum + 1e-8)
    ).item()

    print(f"{hospital_id} sample Dice: {dice:.4f}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"{hospital_id} - Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(label_slice, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction_slice, cmap="gray")
    plt.title(f"Predicted Mask\nDice: {dice:.4f}")
    plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{hospital_id}_prediction.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():

    print("====================================")
    print("FedMed Global Model Visualization")
    print("====================================")

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Global model not found!")
        print(f"Expected path: {MODEL_PATH}")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    print("\nLoading global model...")

    model = create_model().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print("Global model loaded successfully.")

    hospitals = [
        "hospital1",
        "hospital2",
        "hospital3",
    ]

    for hospital in hospitals:
        visualize_hospital(
            model,
            hospital,
            device,
        )

    print("\n====================================")
    print("VISUALIZATION COMPLETE")
    print("====================================")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()