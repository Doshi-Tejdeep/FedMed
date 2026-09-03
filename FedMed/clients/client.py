import flwr as fl
import torch

from model.unet3d import create_model
from training.local_train import get_dataset
from torch.utils.data import DataLoader
from evaluation.dice import dice_score


class FedMedClient(fl.client.NumPyClient):

    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_model().to(self.device)

    # ---------------------------------------------------------
    # Get the dataset for this hospital
    # ---------------------------------------------------------
    def _get_dataset(self):
        return get_dataset(
            self.hospital_id.lower().replace("-", "")
        )

    # ---------------------------------------------------------
    # Get model parameters
    # ---------------------------------------------------------
    def get_parameters(self, config):
        print(
            f"{self.hospital_id}: "
            f"Sending model parameters"
        )

        return [
            value.detach().cpu().numpy()
            for _, value in self.model.state_dict().items()
        ]

    # ---------------------------------------------------------
    # Set model parameters
    # ---------------------------------------------------------
    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()

        new_state_dict = {}

        for (key, old_value), new_value in zip(
            state_dict.items(),
            parameters,
        ):
            new_state_dict[key] = torch.tensor(
                new_value,
                dtype=old_value.dtype,
            )

        self.model.load_state_dict(
            new_state_dict,
            strict=True,
        )

    # ---------------------------------------------------------
    # Legacy NumPyClient training method
    # ---------------------------------------------------------
    def fit(self, parameters, config):

        print(
            f"{self.hospital_id}: "
            f"Starting local training"
        )

        self.set_parameters(parameters)

        dataset = self._get_dataset()

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

            loss = loss_function(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"{self.hospital_id}: "
            f"Local training complete - "
            f"Loss: {average_loss:.4f}"
        )

        local_state = self.model.state_dict()

        local_parameters = [
            value.detach().cpu().numpy()
            for _, value in local_state.items()
        ]

        return (
            local_parameters,
            len(dataset),
            {
                "hospital_id": self.hospital_id,
                "loss": float(average_loss),
            },
        )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    def evaluate(self, parameters, config):

        print(
            f"{self.hospital_id}: "
            f"Evaluating model"
        )

        self.set_parameters(parameters)

        dataset = self._get_dataset()

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

                images = batch["image"].to(
                    self.device
                )

                labels = batch["label"].to(
                    self.device
                )

                outputs = self.model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                labels = labels.squeeze(1).long()

                predictions = (
                    predictions > 0
                ).float()

                labels = (
                    labels > 0
                ).float()

                dice = dice_score(
                    predictions,
                    labels,
                )

                total_dice += dice
                total_samples += 1

        average_dice = (
            total_dice / total_samples
        )

        print(
            f"{self.hospital_id}: "
            f"Dice Score: "
            f"{average_dice:.4f}"
        )

        return (
            float(average_dice),
            total_samples,
            {
                "hospital_id": self.hospital_id,
                "dice_score": float(
                    average_dice
                ),
            },
        )