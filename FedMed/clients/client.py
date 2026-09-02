import flwr as fl
import torch
import numpy as np

from model.unet3d import create_model
from training.local_train import get_dataset
from torch.utils.data import DataLoader
from evaluation.dice import dice_score
from security.differential_privacy import add_gaussian_noise

class FedMedClient(fl.client.NumPyClient):

    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_model().to(self.device)

        # Differential Privacy configuration
        self.dp_enabled = True
        self.dp_max_norm = 2.5
        self.dp_noise_multiplier = 0.0

    def get_parameters(self, config):
        print(f"{self.hospital_id}: Sending model parameters")

        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()

        new_state_dict = {}

        for (key, old_value), new_value in zip(
            state_dict.items(), parameters
        ):
            new_state_dict[key] = torch.tensor(
                new_value,
                dtype=old_value.dtype,
            )

        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):

        print(f"{self.hospital_id}: Starting local training")

        self.set_parameters(parameters)

        dataset = get_dataset(
            self.hospital_id.lower().replace("-", "")
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

        loss_function = torch.nn.CrossEntropyLoss()

        self.model.train()

        total_loss = 0.0

        for batch in loader:

            images = batch["image"].to(self.device)

            labels = batch["label"].to(self.device)

            labels = labels.squeeze(1).long()

            optimizer.zero_grad()

            outputs = self.model(images)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"{self.hospital_id}: "
            f"Local training complete - Loss: {average_loss:.4f}"
        )

        # ------------------------------------------------------------
        # Differential Privacy
        # Protect the local model update before sending it
        # to the federated server.
        # ------------------------------------------------------------

        local_state = self.model.state_dict()

        local_parameters = [
            value.cpu().numpy()
            for _, value in local_state.items()
        ]

        if self.dp_enabled:

            update = {}

            for index, (global_value, local_value) in enumerate(
                zip(parameters, local_parameters)
            ):
                # Apply DP only to floating-point model parameters.
                if np.issubdtype(local_value.dtype, np.floating):
                    update[str(index)] = (
                        local_value.astype(np.float64)
                        - global_value.astype(np.float64)
                    )

            protected_update = add_gaussian_noise(
                update,
                noise_multiplier=self.dp_noise_multiplier,
                max_norm=self.dp_max_norm,
            )

            protected_parameters = []

            for index, (global_value, local_value) in enumerate(
                zip(parameters, local_parameters)
            ):
                key = str(index)

                if np.issubdtype(local_value.dtype, np.floating):

                    protected_value = (
                        global_value.astype(np.float64)
                        + protected_update[key]
                    ).astype(local_value.dtype)

                else:
                    # Preserve integer/non-floating-point buffers.
                    protected_value = local_value.copy()

                protected_parameters.append(protected_value)

            returned_parameters = protected_parameters

            print(
                f"{self.hospital_id}: "
                f"Differential Privacy applied "
                f"(clip={self.dp_max_norm}, "
                f"noise={self.dp_noise_multiplier})"
            )

        else:

            returned_parameters = local_parameters

        return (
            returned_parameters,
            len(dataset),
            {
                "hospital_id": self.hospital_id,
                "loss": float(average_loss),
                "dp_enabled": self.dp_enabled,
                "dp_noise_multiplier": self.dp_noise_multiplier,
                "dp_max_norm": self.dp_max_norm,
            },
        )

    def evaluate(self, parameters, config):

        print(f"{self.hospital_id}: Evaluating model")

        self.set_parameters(parameters)

        dataset = get_dataset(
            self.hospital_id.lower().replace("-", "")
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        self.model.eval()

        total_dice = 0.0
        total_samples = 0

        with torch.no_grad():

            for batch in loader:

                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                labels = labels.squeeze(1).long()

                # Convert segmentation to binary foreground masks
                predictions = (predictions > 0).float()
                labels = (labels > 0).float()

                dice = dice_score(
                    predictions,
                    labels,
                )

                total_dice += dice
                total_samples += 1

        average_dice = total_dice / total_samples

        print(
            f"{self.hospital_id}: "
            f"Dice Score: {average_dice:.4f}"
        )

        return (
            float(average_dice),
            total_samples,
            {
                "hospital_id": self.hospital_id,
                "dice_score": float(average_dice),
            },
        )


if __name__ == "__main__":

    client = FedMedClient("Hospital-1")

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )