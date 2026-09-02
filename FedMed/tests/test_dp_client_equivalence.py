import numpy as np
import torch
from torch.utils.data import DataLoader

from clients.client import FedMedClient
from training.local_train import get_dataset


def test_zero_noise_dp_matches_standard_fedavg():

    print("\n===== DP CLIENT EQUIVALENCE TEST =====")

    hospital_id = "Hospital-1"
    hospital_key = hospital_id.lower().replace("-", "")

    # ------------------------------------------------------------
    # Create identical starting models
    # ------------------------------------------------------------

    normal_client = FedMedClient(hospital_id)
    dp_client = FedMedClient(hospital_id)

    initial_parameters = normal_client.get_parameters({})

    normal_client.set_parameters(initial_parameters)
    dp_client.set_parameters(initial_parameters)

    # ------------------------------------------------------------
    # Load dataset once
    # ------------------------------------------------------------

    dataset = get_dataset(hospital_key)

    print(f"Hospital: {hospital_id}")
    print(f"Dataset size: {len(dataset)}")

    # IMPORTANT:
    # shuffle=False guarantees identical batch ordering.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # Train normal client
    # ------------------------------------------------------------

    normal_client.model.train()

    optimizer_normal = torch.optim.Adam(
        normal_client.model.parameters(),
        lr=1e-3,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    normal_total_loss = 0.0

    for batch in loader:

        images = batch["image"].to(
            normal_client.device
        )

        labels = batch["label"].to(
            normal_client.device
        )

        labels = labels.squeeze(1).long()

        optimizer_normal.zero_grad()

        outputs = normal_client.model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer_normal.step()

        normal_total_loss += loss.item()

    normal_loss = normal_total_loss / len(loader)

    normal_parameters = [
        value.cpu().numpy().copy()
        for _, value in normal_client.model.state_dict().items()
    ]

    # ------------------------------------------------------------
    # Train DP client
    # ------------------------------------------------------------

    dp_client.model.train()

    optimizer_dp = torch.optim.Adam(
        dp_client.model.parameters(),
        lr=1e-3,
    )

    dp_total_loss = 0.0

    for batch in loader:

        images = batch["image"].to(
            dp_client.device
        )

        labels = batch["label"].to(
            dp_client.device
        )

        labels = labels.squeeze(1).long()

        optimizer_dp.zero_grad()

        outputs = dp_client.model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer_dp.step()

        dp_total_loss += loss.item()

    dp_loss = dp_total_loss / len(loader)

    # ------------------------------------------------------------
    # Apply the SAME DP transformation as client.py
    # ------------------------------------------------------------

    local_parameters = [
        value.cpu().numpy()
        for _, value in dp_client.model.state_dict().items()
    ]

    protected_update = {}

    for index, (
        global_value,
        local_value
    ) in enumerate(
        zip(initial_parameters, local_parameters)
    ):

        protected_update[str(index)] = (
            local_value.astype(np.float64)
            -
            global_value.astype(np.float64)
        )

    # Import actual DP function
    from security.differential_privacy import add_gaussian_noise

    protected_update = add_gaussian_noise(
        protected_update,
        noise_multiplier=0.0,
        max_norm=2.5,
    )

    reconstructed_parameters = []

    for index, (
        global_value,
        local_value
    ) in enumerate(
        zip(initial_parameters, local_parameters)
    ):

        key = str(index)

        reconstructed_value = (
            global_value.astype(np.float64)
            +
            protected_update[key]
        ).astype(local_value.dtype)

        reconstructed_parameters.append(
            reconstructed_value
        )

    # ------------------------------------------------------------
    # Compare normal local model vs DP local model
    # ------------------------------------------------------------

    max_difference = 0.0
    total_difference = 0.0

    for normal_value, dp_value in zip(
        normal_parameters,
        reconstructed_parameters,
    ):

        difference = np.max(
            np.abs(
                normal_value.astype(np.float64)
                -
                dp_value.astype(np.float64)
            )
        )

        total_difference += np.sum(
            np.abs(
                normal_value.astype(np.float64)
                -
                dp_value.astype(np.float64)
            )
        )

        max_difference = max(
            max_difference,
            float(difference)
        )

    print("\n===== RESULTS =====")

    print(f"Normal local loss: {normal_loss:.10f}")
    print(f"DP local loss:     {dp_loss:.10f}")

    print(
        f"Maximum parameter difference: "
        f"{max_difference:.10f}"
    )

    print(
        f"Total parameter difference: "
        f"{total_difference:.10f}"
    )

    print("===================")

    assert abs(normal_loss - dp_loss) < 1e-6
    assert max_difference < 1e-5