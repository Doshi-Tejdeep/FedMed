import torch


def _prepare_binary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Convert predictions and targets into binary foreground masks.
    Any class greater than 0 is treated as foreground.
    """

    predictions = predictions.float()
    targets = targets.float()

    predictions = (predictions > 0).float()
    targets = (targets > 0).float()

    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    return predictions, targets


def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate binary foreground Dice score.
    """

    predictions, targets = _prepare_binary(
        predictions,
        targets,
    )

    intersection = (
        predictions * targets
    ).sum()

    denominator = (
        predictions.sum()
        +
        targets.sum()
    )

    dice = (
        2.0 * intersection + 1e-8
    ) / (
        denominator + 1e-8
    )

    return float(dice.item())


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate binary foreground Intersection over Union.
    """

    predictions, targets = _prepare_binary(
        predictions,
        targets,
    )

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions
        +
        targets
        -
        predictions * targets
    ).sum()

    iou = (
        intersection + 1e-8
    ) / (
        union + 1e-8
    )

    return float(iou.item())


def precision_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate binary foreground precision.
    """

    predictions, targets = _prepare_binary(
        predictions,
        targets,
    )

    true_positive = (
        predictions * targets
    ).sum()

    predicted_positive = predictions.sum()

    precision = (
        true_positive + 1e-8
    ) / (
        predicted_positive + 1e-8
    )

    return float(precision.item())


def recall_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate binary foreground recall.
    """

    predictions, targets = _prepare_binary(
        predictions,
        targets,
    )

    true_positive = (
        predictions * targets
    ).sum()

    actual_positive = targets.sum()

    recall = (
        true_positive + 1e-8
    ) / (
        actual_positive + 1e-8
    )

    return float(recall.item())