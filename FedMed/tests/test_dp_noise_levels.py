import numpy as np
import torch
from torch.utils.data import DataLoader

from clients.client import FedMedClient
from training.local_train import get_dataset
from security.differential_privacy import add_gaussian_noise


def test_dp_noise_levels():

    print("\n===== DP NOISE LEVEL TEST =====")

    hospital_id = "Hospital-1"
    hospital_key = hospital_id.lower().replace("-", "")

    # ---------------------------------------------------------
    # Create a client
    # ---------------------------------------------------------

    client = FedMedClient(hospital_id)

    # ---------------------------------------------------------
    # IMPORTANT:
    # Save a COPY of the initial global model parameters
    # BEFORE local training.
    # ---------------------------------------------------------

    initial_parameters = [
        value.cpu().numpy().copy()
        for _, value in client.model.state_dict().items()
    ]

    print(f"Hospital: {hospital_id}")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    dataset = get_dataset(hospital_key)

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Local training
    # ---------------------------------------------------------

    client.model.train()

    optimizer = torch.optim.Adam(
        client.model.parameters(),
        lr=1e-3,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    total_loss = 0.0

    for batch in loader:

        images = batch["image"].to(client.device)
        labels = batch["label"].to(client.device)

        labels = labels.squeeze(1).long()

        optimizer.zero_grad()

        outputs = client.model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(f"Local training loss: {average_loss:.10f}")

    # ---------------------------------------------------------
    # Get locally trained parameters
    # ---------------------------------------------------------

    local_parameters = [
        value.cpu().numpy().copy()
        for _, value in client.model.state_dict().items()
    ]

    # ---------------------------------------------------------
    # Calculate LOCAL MODEL UPDATE
    # ---------------------------------------------------------

    update = {}

    for index, (global_value, local_value) in enumerate(
        zip(initial_parameters, local_parameters)
    ):

        if np.issubdtype(local_value.dtype, np.floating):

            update[str(index)] = (
                local_value.astype(np.float64)
                -
                global_value.astype(np.float64)
            )

    # ---------------------------------------------------------
    # Calculate original update norm
    # ---------------------------------------------------------

    original_update_norm = float(
        np.sqrt(
            sum(
                np.sum(value ** 2)
                for value in update.values()
            )
        )
    )

    print(
        f"\nOriginal update norm: "
        f"{original_update_norm:.6f}"
    )

    # ---------------------------------------------------------
    # Test different DP noise levels
    # ---------------------------------------------------------

    noise_levels = [
        0.0001,
        0.001,
        0.01,
        0.1,
    ]

    results = []

    for noise_multiplier in noise_levels:

        protected_update = add_gaussian_noise(
            update,
            noise_multiplier=noise_multiplier,
            max_norm=2.5,
            seed=42,
        )

        # -----------------------------------------------------
        # Calculate noise
        # -----------------------------------------------------

        noise_squared = 0.0

        for key in update:

            noise = (
                protected_update[key].astype(np.float64)
                -
                update[key].astype(np.float64)
            )

            noise_squared += np.sum(noise ** 2)

        noise_norm = float(np.sqrt(noise_squared))

        # -----------------------------------------------------
        # Calculate protected update norm
        # -----------------------------------------------------

        protected_norm = float(
            np.sqrt(
                sum(
                    np.sum(
                        value.astype(np.float64) ** 2
                    )
                    for value in protected_update.values()
                )
            )
        )

        # -----------------------------------------------------
        # Noise / update ratio
        # -----------------------------------------------------

        ratio = (
            noise_norm / original_update_norm
            if original_update_norm > 0
            else float("inf")
        )

        results.append(
            (
                noise_multiplier,
                noise_norm,
                protected_norm,
                ratio,
            )
        )

        print("\n----------------------------------------")
        print(
            f"Noise multiplier : "
            f"{noise_multiplier}"
        )
        print(
            f"Original norm    : "
            f"{original_update_norm:.6f}"
        )
        print(
            f"Protected norm   : "
            f"{protected_norm:.6f}"
        )
        print(
            f"Noise norm       : "
            f"{noise_norm:.6f}"
        )
        print(
            f"Noise/update     : "
            f"{ratio:.6f}"
        )
        print("----------------------------------------")

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    assert original_update_norm > 0

    # Noise should increase as noise_multiplier increases.
    assert results[0][1] < results[1][1]
    assert results[1][1] < results[2][1]
    assert results[2][1] < results[3][1]

    print("\n===== TEST COMPLETE =====")