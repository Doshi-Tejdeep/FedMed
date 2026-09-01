import os
import torch
from torch.utils.data import DataLoader

from model.unet3d import create_model
from training.local_train import get_dataset
from evaluation.dice import dice_score


MODEL_PATH = "models/global_model.pth"


def evaluate_hospital(model, hospital_id, device):
    print(f"\nEvaluating {hospital_id}...")

    dataset = get_dataset(hospital_id)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model.eval()

    total_dice = 0.0
    count = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            # Convert model output to predicted class
            predictions = torch.argmax(outputs, dim=1)

            # Remove channel dimension from ground-truth mask
            labels = labels.squeeze(1).long()

            # Calculate Dice score
            score = dice_score(
                predictions,
                labels
            )

            total_dice += float(score)
            count += 1

    average_dice = total_dice / count

    print(
        f"{hospital_id} Dice: "
        f"{average_dice:.4f}"
    )

    return average_dice


def main():

    print("====================================")
    print("FedMed Global Model Evaluation")
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

    # Create same U-Net architecture
    model = create_model().to(device)

    # Load federated global parameters
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

    scores = []

    for hospital in hospitals:

        score = evaluate_hospital(
            model,
            hospital,
            device,
        )

        scores.append(score)

    average_score = sum(scores) / len(scores)

    print("\n====================================")
    print("GLOBAL MODEL RESULTS")
    print("====================================")

    print(
        f"Hospital-1 Dice: {scores[0]:.4f}"
    )

    print(
        f"Hospital-2 Dice: {scores[1]:.4f}"
    )

    print(
        f"Hospital-3 Dice: {scores[2]:.4f}"
    )

    print("------------------------------------")

    print(
        f"Average Dice: {average_score:.4f}"
    )

    print("====================================")


if __name__ == "__main__":
    main()