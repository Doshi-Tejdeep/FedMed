import torch


def dice_score(predictions, targets):
    """
    Calculate Dice coefficient for binary segmentation.
    """

    predictions = predictions.float()
    targets = targets.float()

    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (predictions * targets).sum()

    dice = (2.0 * intersection + 1e-8) / (
        predictions.sum() + targets.sum() + 1e-8
    )

    return dice.item()