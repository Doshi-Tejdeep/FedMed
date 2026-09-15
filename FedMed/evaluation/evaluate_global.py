import argparse
import csv
import os
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)
from model.unet3d import create_model
from training.local_train import get_dataset


MODEL_PATH = "models/global_model.pth"
DEFAULT_OUTPUT_PATH = "outputs/evaluation_results.csv"


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

            total_dice += float(dice)
            total_iou += float(iou)
            total_precision += float(precision)
            total_recall += float(recall)

            count += 1

    if count == 0:
        raise RuntimeError(
            f"No evaluation samples found for {hospital_id}."
        )

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


def save_results(
    hospital_results,
    experiment_name,
    noise_multiplier,
    clipping_norm,
    output_path,
):
    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True,
    )

    rows = []

    hospitals = [
        "Hospital-1",
        "Hospital-2",
        "Hospital-3",
    ]

    for hospital_name, result in zip(
        hospitals,
        hospital_results,
    ):
        rows.append(
            {
                "experiment": experiment_name,
                "noise_multiplier": noise_multiplier,
                "clipping_norm": clipping_norm,
                "hospital": hospital_name,
                "dice": f"{result['dice']:.4f}",
                "iou": f"{result['iou']:.4f}",
                "precision": f"{result['precision']:.4f}",
                "recall": f"{result['recall']:.4f}",
                "model_path": MODEL_PATH,
                "evaluated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

    fieldnames = [
        "experiment",
        "noise_multiplier",
        "clipping_norm",
        "hospital",
        "dice",
        "iou",
        "precision",
        "recall",
        "model_path",
        "evaluated_at_utc",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nEvaluation results saved to: {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the FedMed global model "
            "and save machine-readable results."
        )
    )

    parser.add_argument(
        "--experiment",
        default="DP-FedAvg",
        help="Experiment name.",
    )

    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=0.5,
        help="DP noise multiplier.",
    )

    parser.add_argument(
        "--clipping-norm",
        type=float,
        default=2.5,
        help="DP clipping norm.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the generated CSV file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("====================================")
    print("FedMed Global Model Evaluation")
    print("====================================")

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Global model not found!")
        print(f"Expected path: {MODEL_PATH}")
        return 1

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

    save_results(
        hospital_results=hospital_results,
        experiment_name=args.experiment,
        noise_multiplier=args.noise_multiplier,
        clipping_norm=args.clipping_norm,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())