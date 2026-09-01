import os

import flwr as fl
import torch

from model.unet3d import create_model


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pth")


def save_global_model(parameters):
    """Save aggregated Flower parameters as a PyTorch model."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = create_model()

    state_dict = model.state_dict()

    new_state_dict = {}

    for (key, old_value), new_value in zip(
        state_dict.items(), parameters
    ):
        new_state_dict[key] = torch.tensor(
            new_value,
            dtype=old_value.dtype,
        )

    model.load_state_dict(new_state_dict, strict=True)

    torch.save(model.state_dict(), MODEL_PATH)

    print()
    print("===== GLOBAL MODEL SAVED =====")
    print(f"Path: {MODEL_PATH}")
    print("==============================")
    print()


def aggregate_fit_metrics(metrics):
    print("\n===== FEDAVG AGGREGATION =====")
    print(f"Clients participating: {len(metrics)}")

    for _, client_metrics in metrics:
        print(
            f"Hospital: "
            f"{client_metrics.get('hospital_id', 'Unknown')}"
        )

    print("==============================\n")

    return {}


def aggregate_evaluate_metrics(metrics):
    print("\n===== FEDERATED EVALUATION =====")
    print(f"Clients participating: {len(metrics)}")

    for _, client_metrics in metrics:
        print(
            f"Hospital: "
            f"{client_metrics.get('hospital_id', 'Unknown')}"
        )

    print("================================\n")

    return {}


class FedAvgWithModelSaving(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:

            print(
                f"Saving global model after round "
                f"{server_round}..."
            )

            parameters = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            save_global_model(parameters)

        return aggregated_parameters, metrics


def main():

    print("Starting FedMed Federated Server...")
    print("Waiting for 3 hospitals...")
    print()

    strategy = FedAvgWithModelSaving(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=3,

        min_evaluate_clients=3,

        min_available_clients=3,

        fit_metrics_aggregation_fn=aggregate_fit_metrics,

        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    fl.server.start_server(

        server_address="127.0.0.1:8080",

        config=fl.server.ServerConfig(
            num_rounds=5
        ),

        strategy=strategy,
    )


if __name__ == "__main__":
    main()