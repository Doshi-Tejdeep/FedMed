import flwr as fl

from flwr.client import ClientApp
from flwr.client.mod import fixedclipping_mod

from clients.client import FedMedClient


def client_fn(context):
    client = FedMedClient("Hospital-2")
    return client.to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[fixedclipping_mod],
)


if __name__ == "__main__":
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client_fn=client_fn,
    )