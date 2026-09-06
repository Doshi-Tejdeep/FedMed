# FedMed DP Experiment Results

## Experiment Setup

- Federated hospitals: 3
- Federated rounds: 5
- Local epochs: 1
- Clipping norm: 2.5
- DP noise multiplier: 0.5
- Framework: Flower 1.35.0

## Results

| Metric | Average |
|---|---:|
| Dice | 0.0297 |
| IoU | 0.0151 |
| Precision | 0.0154 |
| Recall | 0.4045 |

## Training

- Round 1 loss: 0.7113
- Round 2 loss: 1.7517
- Round 3 loss: 2.2359
- Round 4 loss: 2.5607
- Round 5 loss: 2.3503

## Validation

- 3/3 hospitals participated in every round.
- 0 training failures.
- Central DP noise was applied.
- Global model was saved successfully.
- 11/11 automated tests passed.