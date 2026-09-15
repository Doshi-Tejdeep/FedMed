# FedMed DP Experiment Results

## Experiment Setup

- Federated hospitals: 3
- Federated rounds: 5
- Local epochs: 1
- Learning rate: 0.001
- Clipping norm: 2.5
- DP noise multiplier: 0.5
- Framework: Flower 1.35.0
- Aggregation: FedAvg with native Flower Differential Privacy
- Client failures: 0

## Latest Official DP-FedAvg Results

The results below correspond to the latest successful DP-FedAvg run that generated the current `models/global_model.pth` artifact.

| Hospital | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Hospital-1 | 0.0312 | 0.0158 | 0.0159 | 0.8841 |
| Hospital-2 | 0.0311 | 0.0158 | 0.0158 | 0.8812 |
| Hospital-3 | 0.0312 | 0.0158 | 0.0159 | 0.8830 |
| **Average** | **0.0312** | **0.0158** | **0.0159** | **0.8828** |

## Training

| Round | Train Loss | Dice |
|---:|---:|---:|
| 1 | 0.8261 | 0.0305 |
| 2 | 0.6389 | 0.0313 |
| 3 | 2.7583 | 0.0305 |
| 4 | 3.3267 | 0.0310 |
| 5 | 4.4626 | 0.0312 |

## Privacy Mechanism

- Client-side fixed clipping was applied to training updates.
- Clipping norm: 2.5
- Central Gaussian DP noise was applied during aggregation.
- Noise multiplier: 0.5
- Reported central DP noise standard deviation contribution: approximately 0.4167

## Validation

- 3/3 simulated hospitals participated in every federated round.
- 0 training failures.
- All 5 federated rounds completed successfully.
- Global model was saved successfully to `models/global_model.pth`.
- The saved global model was evaluated across all three hospitals.
- 11/11 automated tests passed.

## Interpretation

The latest DP experiment demonstrates a working privacy-preserving federated training pipeline. The current segmentation quality remains limited, with low Dice and IoU and very low precision relative to recall. The results should therefore be treated as engineering and research validation rather than clinical performance.

## Reproducibility

The experiment uses the project configuration:

- Federated rounds: 5
- Hospitals sampled per round: 3
- Local epochs: 1
- Learning rate: 0.001
- DP noise multiplier: 0.5
- DP clipping norm: 2.5