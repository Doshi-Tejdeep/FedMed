from clients.client import FedMedClient


if __name__ == "__main__":
    client = FedMedClient("Hospital-3")

    client_instance = client.to_client()

    import flwr as fl

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client_instance,
    )