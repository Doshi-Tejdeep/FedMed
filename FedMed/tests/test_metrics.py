import torch

from evaluation.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)


def test_perfect_prediction():

    prediction = torch.tensor(
        [[
            [0, 1],
            [1, 0],
        ]]
    )

    target = prediction.clone()

    assert dice_score(prediction, target) == 1.0
    assert iou_score(prediction, target) == 1.0
    assert precision_score(prediction, target) == 1.0
    assert recall_score(prediction, target) == 1.0


def test_no_overlap():

    prediction = torch.tensor(
        [[
            [1, 1],
            [0, 0],
        ]]
    )

    target = torch.tensor(
        [[
            [0, 0],
            [1, 1],
        ]]
    )

    assert dice_score(prediction, target) < 1e-6
    assert iou_score(prediction, target) < 1e-6
    assert precision_score(prediction, target) < 1e-6
    assert recall_score(prediction, target) < 1e-6