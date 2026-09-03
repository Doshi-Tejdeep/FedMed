import os

import torch
from torch.utils.data import DataLoader

from model.unet3d import create_model
from training.local_train import get_dataset

from evaluation.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)


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
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0

    count = 0

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            labels = labels.squeeze(1).long()

            dice = dice_score(
                predictions,
                labels,
            )

            iou = iou_score(
                predictions,
                labels,
            )

            precision = precision_score(
                predictions,
                labels,
            )

            recall = recall_score(
                predictions,
                labels,
            )

            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall

            count += 1

    results = {
        "dice": total_dice / count,
        "iou": total_iou / count,
        "precision": total_precision / count,
        "recall": total_recall / count,
    }

    print(
        f"{hospital_id} | "
        f"Dice: {results['dice']:.4f} | "
        f"IoU: {results['iou']:.4f} | "
        f"Precision: {results['precision']:.4f} | "
        f"Recall: {results['recall']:.4f}"
    )

    return results


def main():

    print("====================================")
    print("FedMed Global Model Evaluation")
    print("====================================")

    if not os.path.exists(MODEL_PATH):

        print("ERROR: Global model not found!")
        print(f"Expected path: {MODEL_PATH}")

        return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    hospital_results = []

    for hospital in hospitals:

        result = evaluate_hospital(
            model,
            hospital,
            device,
        )

        hospital_results.append(result)

    average_dice = sum(
        result["dice"]
        for result in hospital_results
    ) / len(hospital_results)

    average_iou = sum(
        result["iou"]
        for result in hospital_results
    ) / len(hospital_results)

    average_precision = sum(
        result["precision"]
        for result in hospital_results
    ) / len(hospital_results)

    average_recall = sum(
        result["recall"]
        for result in hospital_results
    ) / len(hospital_results)

    print("\n====================================")
    print("GLOBAL MODEL RESULTS")
    print("====================================")

    for index, result in enumerate(
        hospital_results,
        start=1,
    ):

        print(
            f"Hospital-{index}: "
            f"Dice={result['dice']:.4f}, "
            f"IoU={result['iou']:.4f}, "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}"
        )

    print("------------------------------------")

    print(
        f"Average Dice:      {average_dice:.4f}"
    )

    print(
        f"Average IoU:       {average_iou:.4f}"
    )

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    print(
        f"Average Recall:    {average_recall:.4f}"
    )

    print("====================================")


if __name__ == "__main__":
    main()