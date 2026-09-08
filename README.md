# 🏥 FedMed: Cross-Silo Federated Learning Engine

> **Privacy-Preserving Machine Learning (PPML) for Collaborative Healthcare AI**

FedMed is a **privacy-first cross-silo federated learning framework** designed to enable multiple hospitals to collaboratively train a **3D medical-image segmentation model** without directly sharing raw patient MRI data with a centralized server.

Instead of transferring sensitive medical data to a central location, each hospital performs model training locally. The federated server coordinates the training process and aggregates model updates to produce a shared global model.

The current implementation combines **Federated Learning, 3D medical-image segmentation, FedAvg aggregation, Differential Privacy, privacy accounting, and a Streamlit-based monitoring interface** into a unified research-oriented healthcare AI architecture.

Advanced capabilities such as **Homomorphic Encryption, Secure Aggregation, gRPC/TLS deployment, WebSocket monitoring, React dashboards, Docker deployment, and CI/CD** are part of the planned future roadmap and are not presented as completed functionality.

---

# 🎯 Problem Statement

Training accurate machine-learning models for medical imaging often requires large and diverse datasets.

However, hospitals and healthcare institutions may face significant restrictions when sharing raw patient information because of privacy, security, ownership, and regulatory concerns.

FedMed addresses this challenge through a federated learning architecture in which hospitals can participate in collaborative model training while keeping their local datasets within their respective environments.

### Traditional Centralized Approach

    Hospital 1 ──┐
    Hospital 2 ──┼──> Central Server
    Hospital 3 ──┘

           Raw Patient MRI Data
                  ↓
          Centralized Storage
                  ↓
             Privacy Risk

### FedMed Federated Approach

    Hospital 1 ──> Local Training ──┐
                                    │
    Hospital 2 ──> Local Training ──┼──> Federated Aggregation
                                    │
    Hospital 3 ──> Local Training ──┘
                                          ↓
                                  Global Model
                                          ↓
                               Updated Model to Nodes

**Raw training data remains local to the participating hospital nodes in the federated setup.**

---

# 🚀 Key Objectives

The primary objectives of FedMed are:

- Enable collaborative machine-learning training across multiple hospital nodes.
- Keep raw medical imaging data local to each participating node.
- Train a 3D U-Net-based medical-image segmentation model.
- Orchestrate distributed training using Federated Learning.
- Aggregate client models using FedAvg.
- Apply Differential Privacy to the federated training process.
- Calculate privacy budgets using RDP-based accounting.
- Evaluate segmentation performance using standard metrics.
- Provide visibility into the training process through a monitoring dashboard.
- Establish a foundation for future privacy-enhancing technologies.
- Investigate the trade-off between privacy protection and model utility.

---

# 📌 Current Implementation Status

FedMed is being developed in multiple stages.

## ✅ Implemented

The following functionality is currently available and validated:

- PyTorch-based deep-learning model
- MONAI integration
- 3D U-Net segmentation architecture
- Multiple simulated hospital clients
- Local dataset loading
- Local client-side training
- Native Flower `ServerApp`
- Native Flower `ClientApp`
- FedAvg aggregation
- Multi-round federated training
- Global model generation
- Global model evaluation
- Dice Score
- IoU
- Precision
- Recall
- Flower native Differential Privacy integration
- Client-side fixed clipping
- Central Gaussian noise
- RDP privacy accounting
- Configurable federated training parameters
- Streamlit monitoring dashboard
- Automated testing
- Git-based feature development workflow

## 🚧 Planned / Future Enhancements

The following capabilities are planned but are not currently treated as completed features:

- Homomorphic Encryption using TenSEAL
- Secure Aggregation
- Encrypted model aggregation
- gRPC-based production communication
- TLS deployment for distributed hospital infrastructure
- WebSocket-based real-time metric streaming
- React-based production dashboard
- Centralized-vs-federated benchmark experiments
- Advanced MRI data augmentation
- Client failure recovery
- Docker / Docker Compose deployment
- GitHub Actions CI/CD
- Production authentication and authorization
- Production-grade monitoring and audit logging

---

# 🧠 Why Federated Learning?

A centralized machine-learning architecture requires data from multiple institutions to be transferred to a common infrastructure.

For medical applications, this can create significant privacy and governance challenges.

Federated Learning changes the data-flow model.

Instead of moving the data:

    Hospital Data
         ↓
    Central Server
         ↓
       Training

the model moves toward the data:

    Global Model
         ↓
      Hospital
         ↓
    Local Training
         ↓
    Model Update
         ↓
    Federated Server

This allows participating institutions to collaborate while reducing the need to transfer raw training data to the central server.

---

# 🏥 Cross-Silo Federated Architecture

FedMed follows a **cross-silo federated learning** design.

Each hospital is represented as an independent federated client.

Current simulated nodes include:

    Hospital-1
    Hospital-2
    Hospital-3

Each node:

1. Loads its local dataset.
2. Receives the current global model.
3. Performs local training.
4. Generates a local model update.
5. Participates in federated aggregation.

The central server:

1. Initializes the global model.
2. Sends the model to participating nodes.
3. Coordinates training rounds.
4. Aggregates local updates using FedAvg.
5. Applies the configured privacy mechanism.
6. Produces the next global model.
7. Stores the resulting model for evaluation.

---

# 🤖 Machine Learning Pipeline

FedMed uses **PyTorch** and **MONAI** for medical-image machine learning.

The current model architecture is a **3D U-Net**, selected as the foundation for volumetric medical-image segmentation research.

The current pipeline is conceptually:

    Medical Image Dataset
            ↓
       Dataset Loading
            ↓
     Local Hospital Dataset
            ↓
          3D U-Net
            ↓
       Local Training
            ↓
    Segmentation Prediction
            ↓
      Evaluation Metrics

The project is currently focused on establishing a reliable federated training and privacy pipeline before introducing more advanced preprocessing and optimization strategies.

---

# 🧬 3D U-Net Segmentation

The segmentation network is implemented using the MONAI/PyTorch ecosystem.

The model operates on volumetric medical data rather than only 2D individual images.

The intended workflow is:

    Input MRI Volume
           ↓
    3D Feature Extraction
           ↓
         Encoder
           ↓
       Bottleneck
           ↓
         Decoder
           ↓
    Segmentation Prediction

The current implementation provides the model and training pipeline required for federated experimentation.

Further preprocessing, augmentation, architecture optimization, and larger-scale medical datasets remain future improvements.

---

# 🔄 Federated Training Workflow

The current federated training process follows this workflow:

                     Global Model
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
       Hospital-1     Hospital-2     Hospital-3
           │              │              │
           ▼              ▼              ▼
      Local Training  Local Training  Local Training
           │              │              │
           ▼              ▼              ▼
      Local Update    Local Update    Local Update
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                   Privacy Processing
                          │
                          ▼
                    FedAvg Aggregation
                          │
                          ▼
                     Global Model
                          │
                          ▼
                      Next Round

The process repeats for the configured number of federated rounds.

---

# 🧩 Federated Learning Framework

FedMed uses **Flower 1.35.0** and the modern native Flower application architecture.

The project uses:

    ServerApp
    ClientApp
    FedAvg
    Message API
    ArrayRecord
    ConfigRecord
    MetricRecord
    RecordDict

The server is responsible for federated orchestration while individual clients perform local training using their own datasets.

---

# 🖥️ Native Flower Architecture

The server side uses Flower's native `ServerApp` architecture.

Conceptually:

                        ServerApp
                           │
                           ▼
                         FedAvg
                           │
                ┌──────────┼──────────┐
                │          │          │
                ▼          ▼          ▼
            ClientApp  ClientApp  ClientApp
            Hospital-1 Hospital-2 Hospital-3

This architecture provides a modern Flower-based application structure for the federated workflow.

---

# 🔐 Differential Privacy

Differential Privacy is one of the currently implemented privacy mechanisms in FedMed.

The current production training path uses Flower's native Differential Privacy support.

The configured pipeline contains:

    Local Client Training
            ↓
    Client-side Fixed Clipping
            ↓
    Federated Aggregation
            ↓
    Central Gaussian Noise
            ↓
        Global Model

The current implementation uses Flower's native fixed-clipping Differential Privacy mechanism together with the native ClientApp modifier.

---

# 🛡️ Training-Only DP Modifier

The current client implementation uses a training-only wrapper around the fixed clipping modifier.

Conceptually:

    def train_only_fixedclipping_mod(msg, ctxt, call_next):
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)

        return fixedclipping_mod(msg, ctxt, call_next)

The purpose of this wrapper is to ensure that the fixed clipping mechanism is applied to training messages while non-training messages continue through the normal Flower message pipeline.

---

# 📐 Differential Privacy Configuration

The main configuration parameters include:

    num-server-rounds = 5
    local-epochs = 1
    learning-rate = 0.001
    dp-noise-multiplier = 0.5
    dp-clipping-norm = 2.5
    num-sampled-clients = 3

These parameters can be modified to perform different federated-learning and privacy experiments.

---

# 📊 Privacy Accounting

FedMed uses Flower's privacy-accounting utilities to estimate the privacy budget for the configured Differential Privacy mechanism.

The evaluated configuration uses:

    Rounds:               5
    Population:           3
    Sampled Clients:      3
    Delta:                1e-5
    Amplification:        NO_AMPLIFICATION
    Neighboring Relation: ADD_OR_REMOVE_ONE

The observed privacy budgets were approximately:

| Noise Multiplier | Epsilon |
|---:|---:|
| 0.5 | 30.13 |
| 1.0 | 12.30 |
| 2.0 | 5.38 |

These results illustrate the privacy-utility relationship of the implemented Gaussian DP configuration.

Higher noise generally results in a smaller epsilon value and therefore a stronger privacy guarantee under the chosen accounting assumptions.

---

# ⚠️ Privacy Accounting Scope

The privacy values above correspond to the **implemented Flower fixed-clipping Gaussian DP mechanism**.

The project also contains:

    security/differential_privacy.py

which should currently be regarded as reference or experimental privacy code rather than the primary production federated DP mechanism.

The reported privacy accounting should therefore be interpreted specifically in the context of the native Flower DP configuration used during the validated experiments.

---

# 🧪 Federated Learning Baseline

A non-DP federated baseline was successfully executed with:

    Clients:          3
    Rounds:           5
    Failures:         0
    Local Epochs:     1
    Learning Rate:    0.001
    Aggregation:      FedAvg

### Round Loss

| Round | Loss |
|---:|---:|
| 1 | 0.0424 |
| 2 | 0.0651 |
| 3 | 0.0912 |
| 4 | 0.1500 |
| 5 | 0.2086 |

### Global Evaluation

| Hospital | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Hospital-1 | 0.2091 | 0.1167 | 0.1237 | 0.6746 |
| Hospital-2 | 0.2087 | 0.1165 | 0.1234 | 0.6764 |
| Hospital-3 | 0.2080 | 0.1161 | 0.1230 | 0.6749 |
| **Average** | **0.2086** | **0.1164** | **0.1234** | **0.6753** |

The baseline confirms that the complete federated training pipeline can execute successfully across all three simulated clients.

The current segmentation performance remains an area for future model and training improvements.

---

# 🔒 Differential Privacy Experiment — Noise 1.0

A native Flower DP experiment was executed with:

    Clients:          3
    Rounds:           5
    Clipping Norm:    2.5
    Noise Multiplier: 1.0
    Failures:         0

The aggregation process reported central DP noise with an approximately:

    0.8333

standard deviation contribution.

### Round Loss

| Round | Loss |
|---:|---:|
| 1 | 0.7648 |
| 2 | 4.4398 |
| 3 | 5.0521 |
| 4 | 12.0369 |
| 5 | 8.4047 |

### Final Evaluation

| Metric | Average |
|---|---:|
| Dice | 0.0302 |
| IoU | 0.0153 |
| Precision | 0.0156 |
| Recall | 0.4384 |

This experiment demonstrates a substantial utility impact under the tested DP configuration.

---

# 🔒 Differential Privacy Experiment — Noise 0.5

A second native Flower DP experiment was executed with:

    Clients:          3
    Rounds:           5
    Clipping Norm:    2.5
    Noise Multiplier: 0.5
    Failures:         0

The aggregation process reported central DP noise with an approximately:

    0.4167

standard deviation contribution.

### Round Loss

| Round | Loss |
|---:|---:|
| 1 | 0.7113 |
| 2 | 1.7517 |
| 3 | 2.2359 |
| 4 | 2.5607 |
| 5 | 2.3503 |

### Final Evaluation

| Hospital | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Hospital-1 | 0.0297 | 0.0151 | 0.0154 | 0.4040 |
| Hospital-2 | 0.0297 | 0.0151 | 0.0154 | 0.4048 |
| Hospital-3 | 0.0297 | 0.0151 | 0.0154 | 0.4048 |
| **Average** | **0.0297** | **0.0151** | **0.0154** | **0.4045** |

The results provide an initial experimental view of the privacy-utility trade-off in the current training environment.

---

# 📈 Privacy vs Utility

The current experiments provide the following high-level observation:

    Higher Privacy
          ↑
          │
          │    More DP Noise
          │
          │         ↓
          │
          │    Lower Model Utility
          ↓

FedMed therefore treats privacy and segmentation quality as two related objectives rather than optimizing only a single metric.

Future experiments will investigate different privacy parameters, training schedules, model architectures, and data configurations.

---

# 📊 Evaluation Metrics

The project evaluates segmentation performance using standard metrics.

## Dice Score

Measures the overlap between the predicted segmentation mask and the ground-truth mask.

Higher Dice values indicate stronger overlap.

## Intersection over Union

Measures the ratio between the intersection and union of prediction and ground truth.

## Precision

Measures the fraction of positive predictions that are correct.

## Recall

Measures the fraction of actual positive regions that are successfully identified.

These metrics are reported at the hospital level and summarized across participating nodes.

---

# 🖥️ Monitoring Dashboard

The current project includes a **Streamlit-based monitoring interface**.

The dashboard is intended to provide visibility into the federated training process.

The current dashboard serves as the foundation for future advanced visualization features.

Potential monitoring information includes:

- Federated training configuration
- Training progress
- Model evaluation results
- Hospital-level metrics
- Privacy configuration
- Experiment outputs

---

# 🚧 Planned React Dashboard

A production-oriented React dashboard is part of the future roadmap.

The planned dashboard may provide:

    ┌────────────────────────────────────────────────────┐
    │                  FedMed Dashboard                  │
    ├────────────────────────────────────────────────────┤
    │                                                    │
    │ Global Model       Training Round     DP Epsilon   │
    │                                                    │
    ├────────────────────────────────────────────────────┤
    │                                                    │
    │       Global Loss / Dice / IoU Charts              │
    │                                                    │
    ├────────────────────────────────────────────────────┤
    │                                                    │
    │ Hospital-1     ● Online                            │
    │ Hospital-2     ● Online                            │
    │ Hospital-3     ● Online                            │
    │                                                    │
    ├────────────────────────────────────────────────────┤
    │                                                    │
    │       MRI Segmentation Visualization              │
    │                                                    │
    └────────────────────────────────────────────────────┘

This is a planned enhancement and is not currently presented as an implemented React subsystem.

---

# 🔐 Future Homomorphic Encryption

Homomorphic Encryption is planned as a future privacy enhancement.

The project may investigate **TenSEAL** for encrypted computation.

The intended concept is:

    Local Model Update
            ↓
    Homomorphic Encryption
            ↓
      Encrypted Update
            ↓
      Federated Server
            ↓
    Encrypted Computation
            ↓
     Secure Aggregation
            ↓
       Global Model

This functionality is **not currently implemented in the production training path**.

The future goal is to investigate whether encrypted model updates can further reduce the visibility of individual client contributions.

---

# 🤝 Future Secure Aggregation

Secure Aggregation is another planned enhancement.

The future architecture may be extended so that the federated server aggregates client updates without directly observing individual client contributions.

Conceptually:

    Client 1 ──┐
    Client 2 ──┼──> Secure Aggregation ──> Global Update
    Client 3 ──┘

This feature remains part of the future privacy roadmap.

---

# 🌐 Future Distributed Communication

The initial development architecture includes plans for production-style distributed communication using:

- gRPC
- TLS
- REST APIs
- WebSocket

These technologies are intended for future deployment where independent hospital environments communicate over a network.

The current validated implementation focuses on Flower's application-level federated communication and does not claim a completed production gRPC/TLS deployment.

---

# 🧱 Current System Architecture

The current implemented architecture can be summarized as:

                         FEDMED
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
         Streamlit Dashboard      Federated Server
                                         │
                                         ▼
                                    Flower ServerApp
                                         │
                                       FedAvg
                                         │
                          ┌──────────────┼──────────────┐
                          │              │              │
                          ▼              ▼              ▼
                     Hospital-1      Hospital-2      Hospital-3
                          │              │              │
                          ▼              ▼              ▼
                     Local Data      Local Data      Local Data
                          │              │              │
                          ▼              ▼              ▼
                     Local Model     Local Model     Local Model
                          │              │              │
                          └──────────────┼──────────────┘
                                         ▼
                              Differential Privacy
                                         │
                                         ▼
                                   Global Model

---

# 🧭 Target Future Architecture

The long-term architecture is intended to evolve toward:

                         FEDMED
                           │
                 ┌─────────┴─────────┐
                 │ Federated Server  │
                 │                   │
                 │ Flower + FedAvg   │
                 └─────────┬─────────┘
                           │
                   Secure Communication
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         Hospital-1    Hospital-2    Hospital-3
              │            │            │
              ▼            ▼            ▼
          Local MRI     Local MRI     Local MRI
              │            │            │
              ▼            ▼            ▼
          3D U-Net      3D U-Net      3D U-Net
              │            │            │
              ▼            ▼            ▼
          Model Update  Model Update  Model Update
              │            │            │
              └────────────┼────────────┘
                           ▼
              Privacy / Security Layer
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
       Differential Privacy      Homomorphic Encryption
              │                         │
              └────────────┬────────────┘
                           ▼
                    Secure Aggregation
                           │
                           ▼
                      Global Model
                           │
                           ▼
                 Monitoring Dashboard

---

# 🛡️ Privacy-First Design

FedMed follows the principle that sensitive medical data should remain within the local hospital environment whenever possible.

The privacy architecture is built around multiple layers.

## Current Privacy Layer

    Local Medical Data
            ↓
       Local Training
            ↓
        Client Update
            ↓
       Fixed Clipping
            ↓
    Federated Aggregation
            ↓
    Central Gaussian Noise
            ↓
        Global Model

## Future Privacy Layer

    Local Medical Data
            ↓
       Local Training
            ↓
        Client Update
            ↓
    Homomorphic Encryption
            +
    Differential Privacy
            ↓
     Secure Aggregation
            ↓
        Global Model

---

# 🧪 Testing Strategy

FedMed uses automated testing to validate core components.

## Model Testing

- Model initialization
- Forward-pass behavior
- Tensor handling
- Output shape validation

## Federated Training Testing

- Client initialization
- Server configuration
- Multi-client training
- Model aggregation
- Message handling
- Training execution

## Privacy Testing

- DP configuration
- Fixed clipping integration
- Privacy-accounting calculations
- Privacy parameter handling

## Evaluation Testing

- Dice calculation
- IoU calculation
- Precision calculation
- Recall calculation

The available automated test suite has been successfully executed after the current native Flower DP integration.

---

# 🧹 Code Quality

The project uses pre-commit hooks for repository hygiene and code-quality checks.

Installation:

    pip install pre-commit

Run all checks:

    python -m pre_commit run --all-files

Pre-commit helps maintain:

- Clean whitespace
- Proper end-of-file formatting
- Consistent repository formatting
- Basic code-quality hygiene

---

# 📁 Project Structure

The current repository contains the following major areas:

    FedMed/
    │
    ├── clients/
    │   ├── __init__.py
    │   ├── client.py
    │   └── client_app.py
    │
    ├── dashboard/
    │   └── ...
    │
    ├── data/
    │   └── ...
    │
    ├── evaluation/
    │   ├── __init__.py
    │   └── dice.py
    │
    ├── model/
    │   └── ...
    │
    ├── models/
    │   └── ...
    │
    ├── outputs/
    │   └── ...
    │
    ├── security/
    │   ├── __init__.py
    │   └── differential_privacy.py
    │
    ├── server/
    │   ├── __init__.py
    │   └── server_app.py
    │
    ├── training/
    │   └── ...
    │
    ├── tests/
    │   └── ...
    │
    ├── pyproject.toml
    ├── README.md
    └── .gitignore

---

# ⚙️ Configuration

The project uses `pyproject.toml` for major configuration values.

Current key parameters include:

    num-server-rounds = 5
    local-epochs = 1
    learning-rate = 0.001
    dp-noise-multiplier = 0.5
    dp-clipping-norm = 2.5
    num-sampled-clients = 3

This configuration allows experiments to be reproduced and adjusted without modifying the core training logic.

---

# 🐍 Python Requirements

The project is configured for modern Python environments.

The current project metadata specifies:

    requires-python = ">=3.11,<4.0"

A virtual environment is strongly recommended.

---

# 🔧 Installation

## 1. Clone the Repository

    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd FedMed

## 2. Create Virtual Environment

### Windows

    python -m venv .venv

### Linux / macOS

    python3 -m venv .venv

## 3. Activate Environment

### Windows PowerShell

    .venv\Scripts\Activate.ps1

### Windows Command Prompt

    .venv\Scripts\activate

### Linux / macOS

    source .venv/bin/activate

## 4. Install the Project

    pip install -e .

The project includes Flower 1.35.0 with Differential Privacy support.

The dependency specification uses:

    flwr[dp]==1.35.0

---

# ▶️ Running Federated Training

The current project uses Flower's application architecture.

From the repository root, use the configured Flower project command:

    flwr run .

The exact runtime behavior depends on the Flower configuration in the repository.

The expected workflow is:

    Start Federated Application
            ↓
      Initialize ServerApp
            ↓
      Initialize ClientApps
            ↓
        Load Global Model
            ↓
      Start Federated Rounds
            ↓
     Local Hospital Training
            ↓
     Apply Privacy Mechanism
            ↓
       FedAvg Aggregation
            ↓
         Global Model

---

# 📊 Running Tests

Run the automated tests with:

    pytest -q

The test suite should be executed before final integration and release.

For development validation, also run:

    python -m pre_commit run --all-files

---

# 🖥️ Running the Streamlit Dashboard

The current repository contains a Streamlit-based monitoring interface.

Start the dashboard using the Streamlit application entry point present in the repository.

For example:

    streamlit run dashboard/app.py

The exact entry point should match the dashboard file currently present in the repository.

---

# 🌿 Git Workflow

FedMed uses `main` together with feature branches for development.

The current workflow is:

                       main
                         │
                         ▼
                 Feature Branch
                         │
                    Development
                         │
                      Testing
                         │
                      Commit
                         │
                         ▼
                    Pull Request
                         │
                         ▼
                     Code Review
                         │
                         ▼
                        main

Example:

    git checkout main
    git pull origin main

    git checkout -b feature/<feature-name>

    git add .
    git commit -m "feat: <description>"

    git push -u origin feature/<feature-name>

After validation and review, the feature can be integrated into `main`.

---

# 🧪 Reproducibility

For reproducible federated experiments, record:

    Python Version
    Flower Version
    PyTorch Version
    MONAI Version
    Number of Clients
    Number of Rounds
    Local Epochs
    Learning Rate
    DP Noise Multiplier
    DP Clipping Norm
    Dataset Configuration
    Evaluation Metrics

The primary experiment configuration is maintained through the project configuration files.

---

# 📊 Experiment Tracking

Current experiments have demonstrated:

## Federated Baseline

    3 Clients
    5 Rounds
    FedAvg
    No Failures
    Average Dice ≈ 0.2086

## DP Noise 0.5

    3 Clients
    5 Rounds
    Clipping Norm = 2.5
    Noise Multiplier = 0.5
    Average Dice ≈ 0.0297

## DP Noise 1.0

    3 Clients
    5 Rounds
    Clipping Norm = 2.5
    Noise Multiplier = 1.0
    Average Dice ≈ 0.0302

These experiments are intended as engineering and research validation rather than clinical performance claims.

---

# ⚠️ Current Limitations

FedMed is currently a research and engineering prototype.

## 1. Simulated Hospitals

The current hospital nodes are simulated clients rather than independently deployed real hospital institutions.

## 2. Limited Training Configuration

The current experiments use a relatively small number of clients and federated rounds.

## 3. Segmentation Quality

Current segmentation scores are relatively low, particularly in the DP experiments.

Improving the model requires additional work in:

- Data preprocessing
- Dataset scale
- Augmentation
- Model optimization
- Loss-function selection
- Hyperparameter tuning
- Training duration

## 4. Advanced Encryption Not Yet Implemented

Homomorphic Encryption and Secure Aggregation remain future enhancements.

## 5. Production Networking Not Yet Implemented

The repository does not currently represent a complete production hospital deployment using gRPC + TLS infrastructure.

## 6. Dashboard Evolution

The current dashboard is Streamlit-based. A production React monitoring interface remains a future enhancement.

## 7. No Clinical Validation

FedMed is a research prototype and should not be interpreted as a clinically validated diagnostic or treatment system.

---

# 🗺️ Development Roadmap

## Phase 1 — Core Federated Learning

### Completed

- Repository architecture
- PyTorch setup
- MONAI setup
- 3D U-Net
- Hospital clients
- Flower ServerApp
- Flower ClientApp
- FedAvg
- Multi-round training
- Global evaluation

### Status

✅ Completed

---

# Phase 2 — Differential Privacy

### Completed

- Native Flower DP integration
- Client-side fixed clipping
- Central Gaussian noise
- Configurable clipping norm
- Configurable noise multiplier
- DP experiment execution
- Privacy accounting
- Utility evaluation

### Status

✅ Completed

---

# Phase 3 — Experimental Evaluation

### Planned

- Centralized baseline
- Federated-vs-centralized comparison
- Larger experiment matrix
- Longer federated training
- Additional noise levels
- Hyperparameter experiments
- Experiment-result visualization

### Status

🚧 In Progress / Planned

---

# Phase 4 — Dashboard Enhancement

### Current

- Streamlit monitoring foundation

### Planned

- Training curves
- Round-by-round metrics
- Hospital comparison
- Privacy panel
- Epsilon visualization
- Experiment comparison
- Segmentation mask visualization
- Improved UI

### Status

🚧 Planned

---

# Phase 5 — Advanced Privacy

### Planned

- TenSEAL integration
- Homomorphic Encryption
- Encrypted model updates
- Secure Aggregation
- Privacy-preserving aggregation experiments

### Status

🚧 Planned

---

# Phase 6 — Distributed Engineering

### Planned

- gRPC service layer
- TLS certificates
- Secure server-node communication
- WebSocket metric streaming
- Retry mechanisms
- Timeout management
- Client availability monitoring
- Failure recovery

### Status

🚧 Planned

---

# Phase 7 — DevOps

### Planned

- Dockerfile
- Docker Compose
- GitHub Actions
- CI/CD
- Automated integration tests
- Deployment configuration
- Release workflow

### Status

🚧 Planned

---

# 🔬 Research Direction

FedMed is intended to provide a foundation for research into:

**Privacy-Preserving Federated Medical AI**

The main research dimensions are:

    Federated Learning
            +
    Medical Image Segmentation
            +
    Differential Privacy
            +
    Future Encryption
            =
    Privacy-Preserving Healthcare AI

The project investigates whether collaborative model training can be achieved without requiring direct centralized access to raw medical data.

---

# ⚖️ Privacy-Utility Trade-Off

One of the most important research questions in FedMed is the relationship between privacy protection and model utility.

Conceptually:

    More Privacy
         │
         ▼
       More Noise
         │
         ▼
    Potential Utility Reduction

while:

    Less Noise
         │
         ▼
    Higher Potential Utility
         │
         ▼
    Weaker Privacy Protection

The current experiments provide initial evidence of this trade-off under the implemented Differential Privacy configuration.

---

# 🏥 Healthcare AI Considerations

FedMed is designed around several healthcare-AI principles:

- Minimize unnecessary movement of sensitive data.
- Separate local data ownership from model collaboration.
- Measure privacy and model quality independently.
- Make the federated training workflow reproducible.
- Avoid presenting experimental results as clinical validation.
- Clearly distinguish implemented functionality from future architecture.

---

# 🛡️ Security Philosophy

The long-term security model of FedMed is based on multiple layers.

              Security Architecture

                    Medical Data
                         │
                         ▼
                  Local Hospital
                         │
                         ▼
                   Local Training
                         │
                         ▼
                    Model Update
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Differential Privacy   Future Encryption
             │                       │
             └───────────┬───────────┘
                         ▼
                  Secure Transport
                         │
                         ▼
                    Aggregation
                         │
                         ▼
                    Global Model

The current system implements the Differential Privacy portion of this architecture.

The remaining security components are future work.

---

# 📈 Future Centralized vs Federated Comparison

A future experiment will compare three major configurations:

    1. Centralized Training
              ↓
    2. Federated Training
              ↓
    3. Federated + Differential Privacy

The comparison will consider:

| Metric | Centralized | Federated | Federated + DP |
|---|---|---|---|
| Dice | Future Experiment | Current Baseline | Current Experiment |
| IoU | Future Experiment | Current Baseline | Current Experiment |
| Precision | Future Experiment | Current Baseline | Current Experiment |
| Recall | Future Experiment | Current Baseline | Current Experiment |
| Privacy | Low | Improved | Stronger |
| Raw Data Centralization | Yes | No | No |

The centralized values are intentionally left as future experiments rather than invented results.

---

# 📦 Future Deployment Model

The long-term deployment goal is to represent three independent healthcare institutions.

               Hospital Network

      Hospital 1       Hospital 2       Hospital 3
           │                 │                 │
           │                 │                 │
           ▼                 ▼                 ▼
     Local Compute      Local Compute      Local Compute
           │                 │                 │
           └─────────────┬───┴───────────────┘
                         │
                         ▼
                 Federated Server
                         │
                         ▼
                 Secure Aggregation
                         │
                         ▼
                    Global Model

The production deployment architecture will require additional networking, authentication, security, monitoring, and compliance engineering.

---

# 🧪 Future Failure-Handling

Hospital availability is an important practical concern in cross-silo federated learning.

Future versions may introduce:

    Hospital Available
           ↓
    Participate in Round

and:

    Hospital Unavailable
           ↓
     Timeout / Detection
           ↓
    Continue with Available Clients
           ↓
          Aggregate
           ↓
        Next Round

This functionality is part of the planned distributed-systems roadmap.

---

# 🐳 Future Docker Deployment

Containerized deployment is planned for future versions.

The target architecture may include:

    Docker Network
          │
          ├── Federated Server
          ├── Hospital 1
          ├── Hospital 2
          ├── Hospital 3
          └── Monitoring Dashboard

Future Docker Compose configuration may be used to reproduce the complete multi-node development environment.

Docker is not currently treated as a completed production feature.

---

# 🔄 Future CI/CD

The project may introduce GitHub Actions for:

- Automated testing
- Formatting checks
- Static analysis
- Build validation
- Regression testing
- Release verification

Example future workflow:

    Git Push
       ↓
    GitHub Actions
       ↓
    Install Dependencies
       ↓
    Run Tests
       ↓
    Run Quality Checks
       ↓
    Build
       ↓
    Deployment / Release

---

# 👤 Project Ownership

FedMed is currently maintained as a solo project.

| Role | Owner |
|---|---|
| Project Lead / Developer | D. Tejdeep |

---

# 📋 Current Validation Summary

| Component | Status |
|---|---|
| Python | ✅ |
| PyTorch | ✅ |
| MONAI | ✅ |
| 3D U-Net | ✅ |
| Multiple Hospital Clients | ✅ |
| Flower 1.35.0 | ✅ |
| ServerApp | ✅ |
| ClientApp | ✅ |
| FedAvg | ✅ |
| Local Training | ✅ |
| Global Aggregation | ✅ |
| Multi-Round Training | ✅ |
| Dice | ✅ |
| IoU | ✅ |
| Precision | ✅ |
| Recall | ✅ |
| Native Differential Privacy | ✅ |
| Fixed Clipping | ✅ |
| Central Gaussian Noise | ✅ |
| Privacy Accounting | ✅ |
| Streamlit Dashboard | ✅ |
| Automated Tests | ✅ |
| Homomorphic Encryption | 🚧 Planned |
| Secure Aggregation | 🚧 Planned |
| gRPC/TLS | 🚧 Planned |
| WebSocket Monitoring | 🚧 Planned |
| React Dashboard | 🚧 Planned |
| Docker | 🚧 Planned |
| GitHub Actions | 🚧 Planned |
| Centralized Baseline | 🚧 Planned |
| Failure Recovery | 🚧 Planned |

---

# ✅ Success Criteria

The current implementation has successfully established the core federated-learning and Differential Privacy pipeline.

The next stages will extend the system toward the broader target architecture.

## Current Success Criteria

    ✓ Multiple simulated hospital nodes
    ✓ Local dataset processing
    ✓ Local model training
    ✓ Native Flower ServerApp
    ✓ Native Flower ClientApp
    ✓ FedAvg aggregation
    ✓ Multi-round federated training
    ✓ Global model evaluation
    ✓ Segmentation metrics
    ✓ Native Differential Privacy
    ✓ Fixed clipping
    ✓ Central Gaussian noise
    ✓ Privacy accounting
    ✓ Automated testing
    ✓ Streamlit monitoring foundation

## Future Success Criteria

    □ Homomorphic Encryption
    □ Secure Aggregation
    □ gRPC + TLS deployment
    □ WebSocket streaming
    □ React dashboard
    □ Centralized comparison
    □ Node failure recovery
    □ Docker deployment
    □ CI/CD
    □ Production security

---

# 📅 20-Day Development Roadmap

## Days 1–5 — Foundation

### Machine Learning

- Finalize 3D U-Net architecture.
- Validate dataset loading.
- Validate medical-image preprocessing.
- Establish baseline model training.
- Implement segmentation metrics.

### Federated Learning

- Initialize Flower.
- Build ServerApp.
- Build ClientApp.
- Configure three hospital clients.
- Implement FedAvg.

### Status

✅ Core foundation completed.

---

# Days 6–10 — Federated Training

### Tasks

- Partition data across hospital nodes.
- Run local training.
- Implement multi-round training.
- Evaluate global model.
- Record training metrics.
- Validate client participation.

### Status

✅ Core federated training completed.

---

# Days 11–14 — Differential Privacy

### Tasks

- Integrate native Flower DP.
- Configure fixed clipping.
- Configure Gaussian noise.
- Run privacy experiments.
- Perform privacy accounting.
- Compare utility across DP configurations.

### Status

✅ Current DP pipeline completed and validated.

---

# Days 15–17 — Evaluation & Dashboard

### Tasks

- Validate Dice.
- Validate IoU.
- Validate Precision.
- Validate Recall.
- Improve Streamlit monitoring.
- Organize experiment outputs.
- Prepare comparison tables.

### Status

🚧 Ongoing enhancement.

---

# Days 18–20 — Engineering & Final Review

### Tasks

- Repository cleanup.
- Documentation update.
- Re-run regression tests.
- Verify configuration.
- Review privacy methodology.
- Review limitations.
- Finalize README.
- Prepare final demonstration.

### Status

🚧 Finalization stage.

---

# 🎓 Academic / Research Value

FedMed demonstrates the integration of several important concepts:

    Deep Learning
          +
    Medical Imaging
          +
    Federated Learning
          +
    Differential Privacy
          +
    Privacy Accounting
          +
    Distributed AI

The project is useful as a research-oriented demonstration of how collaborative medical AI can be approached without requiring direct centralization of the raw training data.

---

# 🚨 Important Disclaimer

FedMed is a research and educational prototype.

It is **not a clinically validated medical device**, diagnostic system, or treatment system.

The current experiments are intended to demonstrate:

- Federated learning architecture
- Privacy-preserving training concepts
- Medical-image segmentation workflows
- Privacy-utility analysis
- Engineering feasibility

The model results should not be interpreted as clinical accuracy or clinical effectiveness.

---

# 📌 Project Status

> **FedMed is currently in active development.**

The core federated-learning and Differential Privacy pipeline is functional and experimentally validated.

The broader production-oriented architecture remains under development.

The project follows a staged strategy:

    Core Federated Learning
            ↓
    Differential Privacy
            ↓
    Experimental Evaluation
            ↓
    Dashboard Enhancement
            ↓
    Advanced Privacy
            ↓
    Distributed Deployment
            ↓
    Production Engineering

---

# 🏁 Final Vision

FedMed aims to evolve into a complete privacy-preserving collaborative healthcare AI platform.

The long-term vision is:

                        FEDMED
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        Local Hospital Data       Privacy Controls
              │                         │
              ▼                         ▼
           Local AI                DP / Encryption
              │                         │
              └────────────┬────────────┘
                           ▼
                    Federated Learning
                           │
                           ▼
                     Secure Aggregation
                           │
                           ▼
                      Global Model
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
            Hospital Nodes      Monitoring
                                    │
                                    ▼
                               Dashboard

The objective is to demonstrate how hospitals can collaboratively improve medical AI models while maintaining a privacy-first architecture.

---

# 🌟 Core Philosophy

> **Train together. Keep sensitive data local. Protect model updates. Measure privacy. Improve healthcare AI.**

---

# 🙌 Acknowledgements

FedMed builds upon open-source technologies and research ecosystems including:

- PyTorch
- MONAI
- Flower
- NumPy
- Matplotlib
- Streamlit
- Pytest

These technologies provide the foundation for deep learning, medical imaging, federated learning, visualization, and software testing used in the project.

---

# ⭐ FedMed Summary

FedMed is a privacy-first cross-silo federated learning project for collaborative medical-image segmentation.

The current implementation successfully demonstrates:

    PyTorch
       +
    MONAI
       +
    3D U-Net
       +
    Flower 1.35.0
       +
    FedAvg
       +
    Differential Privacy
       +
    RDP Privacy Accounting
       +
    Streamlit
       +
    Automated Testing

The project keeps the federated workflow focused on local training and model-update exchange while providing a foundation for more advanced privacy technologies.

Future versions are planned to investigate:

    Homomorphic Encryption
            +
    Secure Aggregation
            +
    gRPC / TLS
            +
    WebSocket Monitoring
            +
    React Dashboard
            +
    Docker
            +
    CI/CD

The current system establishes the foundation for a broader privacy-preserving healthcare AI platform while clearly separating implemented functionality from future development.

---

# 🏥 FedMed

### Privacy-Preserving Federated Learning for Collaborative Healthcare AI

**Train together. Keep data local. Protect privacy.**

---

# 📜 License

This project is currently intended for educational and research purposes.

License information will be finalized before the final public release.