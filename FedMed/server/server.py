import flwr as fl


def main():
    print("Starting FedMed Federated Server...")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()