from flwr.client import ClientApp
from flwr.client.mod import fixedclipping_mod

from clients.client import FedMedClient


def client_fn(context):
    """Create a FedMed client for the current Flower node."""

    # The hospital identity will be supplied through the Flower run
    # configuration/environment during deployment.
    hospital_id = context.node_config.get(
        "hospital-id",
        "Hospital-1",
    )

    client = FedMedClient(hospital_id)

    return client.to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[fixedclipping_mod],
)