import os

import torch

from flwr.common import Context, ndarrays_to_parameters
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


def save_global_model(parameters):
    """Save aggregated model parameters as a PyTorch state dict."""

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model = create_model()

    state_dict = model.state_dict()

    new_state_dict = {}

    for (key, old_value), new_value in zip(
        state_dict.items(),
        parameters,
    ):
        new_state_dict[key] = torch.tensor(
            new_value,
            dtype=old_value.dtype,
        )

    model.load_state_dict(
        new_state_dict,
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


def aggregate_fit_metrics(metrics):
    """Aggregate and display client training metrics."""

    print("\n===== FEDAVG AGGREGATION =====")
    print(f"Clients participating: {len(metrics)}")

    for _, client_metrics in metrics:
        print(
            "Hospital: "
            f"{client_metrics.get('hospital_id', 'Unknown')}"
        )

    print("==============================\n")

    return {}


def aggregate_evaluate_metrics(metrics):
    """Display client evaluation participation."""

    print("\n===== FEDERATED EVALUATION =====")
    print(f"Clients participating: {len(metrics)}")

    for _, client_metrics in metrics:
        print(
            "Hospital: "
            f"{client_metrics.get('hospital_id', 'Unknown')}"
        )

    print("================================\n")

    return {}


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

    print("Starting FedMed ServerApp...")
    print(f"Federated rounds: {num_rounds}")
    print(f"Sampled clients: {num_sampled_clients}")
    print(f"DP noise multiplier: {noise_multiplier}")
    print(f"DP clipping norm: {clipping_norm}")

    base_strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_sampled_clients,
        min_evaluate_clients=num_sampled_clients,
        min_available_clients=num_sampled_clients,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy=base_strategy,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
        num_sampled_clients=num_sampled_clients,
    )

    # This is intentionally kept as a placeholder until we verify
    # the exact ServerApp grid/strategy integration in Flower 1.35.
    #
    # We do not start training here yet.

    print("FedMed ServerApp configured successfully.")
    print("Waiting for Flower App integration test...")