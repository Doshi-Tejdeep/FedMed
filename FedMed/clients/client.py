import flwr as fl


class FedMedClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

    def get_parameters(self, config):
        print(f"{self.hospital_id}: Sending initial parameters")
        return []

    def fit(self, parameters, config):
        print(f"{self.hospital_id}: Training locally")
        return parameters, 1, {"hospital_id": self.hospital_id}

    def evaluate(self, parameters, config):
        print(f"{self.hospital_id}: Evaluating locally")
        return 0.0, 1, {"hospital_id": self.hospital_id}


if __name__ == "__main__":
    client = FedMedClient("Hospital-1")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )