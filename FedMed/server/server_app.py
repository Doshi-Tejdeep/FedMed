import os

import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import (
    FedAvg,
    DifferentialPrivacyClientSideFixedClipping,
)

from model.unet3d import create_model


MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "global_model.pth",
)


def create_initial_arrays() -> ArrayRecord:
    """Create the initial global model parameters."""

    model = create_model()

    return ArrayRecord(
        torch_state_dict=model.state_dict()
    )


def save_global_model(array_record: ArrayRecord) -> None:
    """Save the final global model as a PyTorch state dict."""

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model = create_model()

    state_dict = array_record.to_torch_state_dict()

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print()
    print("===== GLOBAL MODEL SAVED =====")
    print(f"Path: {MODEL_PATH}")
    print("==============================")
    print()


app = ServerApp()


@app.main()
def main(grid, context: Context):

    num_rounds = int(
        context.run_config["num-server-rounds"]
    )

    noise_multiplier = float(
        context.run_config["dp-noise-multiplier"]
    )

    clipping_norm = float(
        context.run_config["dp-clipping-norm"]
    )

    num_sampled_clients = int(
        context.run_config["num-sampled-clients"]
    )

    local_epochs = int(
        context.run_config.get(
            "local-epochs",
            1,
        )
    )

    learning_rate = float(
        context.run_config.get(
            "learning-rate",
            0.001,
        )
    )

    print("====================================")
    print("FedMed ServerApp")
    print("====================================")
    print(f"Federated rounds    : {num_rounds}")
    print(f"Sampled clients     : {num_sampled_clients}")
    print(
        f"DP noise multiplier : "
        f"{noise_multiplier}"
    )
    print(
        f"DP clipping norm    : "
        f"{clipping_norm}"
    )
    print("====================================")

    # ---------------------------------------------------------
    # Base FedAvg strategy
    # ---------------------------------------------------------
    base_strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=num_sampled_clients,
        min_evaluate_nodes=num_sampled_clients,
        min_available_nodes=num_sampled_clients,
    )

    # ---------------------------------------------------------
    # Native Flower client-side clipping + central DP noise
    # ---------------------------------------------------------
    strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy=base_strategy,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
        num_sampled_clients=num_sampled_clients,
    )

    # ---------------------------------------------------------
    # Initial global model
    # ---------------------------------------------------------
    initial_arrays = create_initial_arrays()

    # ---------------------------------------------------------
    # Configuration sent to every client
    # ---------------------------------------------------------
    train_config = ConfigRecord(
        {
            "local-epochs": local_epochs,
            "learning-rate": learning_rate,
        }
    )

    print()
    print("Starting federated training...")
    print()

    # ---------------------------------------------------------
    # START FEDERATED TRAINING
    # ---------------------------------------------------------
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=train_config,
    )

    # ---------------------------------------------------------
    # Save final global model
    # ---------------------------------------------------------
    if result.arrays is not None:
        save_global_model(
            result.arrays
        )

    print()
    print("====================================")
    print("FedMed federated training complete")
    print("====================================")