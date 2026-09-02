import numpy as np
import torch
from torch.utils.data import DataLoader

from clients.client import FedMedClient
from training.local_train import get_dataset
from security.differential_privacy import clip_update


def test_dp_clipping_across_hospitals():

    print("\n===== DP CLIPPING TEST =====")

    hospitals = [
        "Hospital-1",
        "Hospital-2",
        "Hospital-3",
    ]

    max_norm = 2.5

    update_norms = []

    for hospital_id in hospitals:

        print(f"\n===== {hospital_id} =====")

        hospital_key = hospital_id.lower().replace("-", "")

        client = FedMedClient(hospital_id)

        # Save initial model parameters
        initial_parameters = [
            value.cpu().numpy().copy()
            for _, value in client.model.state_dict().items()
        ]

        dataset = get_dataset(hospital_key)

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        client.model.train()

        optimizer = torch.optim.Adam(
            client.model.parameters(),
            lr=1e-3,
        )

        loss_function = torch.nn.CrossEntropyLoss()

        for batch in loader:

            images = batch["image"].to(client.device)
            labels = batch["label"].to(client.device)

            labels = labels.squeeze(1).long()

            optimizer.zero_grad()

            outputs = client.model(images)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

        local_parameters = [
            value.cpu().numpy().copy()
            for _, value in client.model.state_dict().items()
        ]

        # Calculate local update
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

        # Original norm
        original_norm = float(
            np.sqrt(
                sum(
                    np.sum(value ** 2)
                    for value in update.values()
                )
            )
        )

        # Apply clipping
        clipped_update = clip_update(
            update,
            max_norm=max_norm,
        )

        clipped_norm = float(
            np.sqrt(
                sum(
                    np.sum(
                        value.astype(np.float64) ** 2
                    )
                    for value in clipped_update.values()
                )
            )
        )

        was_clipped = original_norm > max_norm

        print(f"Original update norm : {original_norm:.6f}")
        print(f"Clipped update norm  : {clipped_norm:.6f}")
        print(f"Max norm             : {max_norm:.6f}")
        print(f"Was clipped          : {was_clipped}")

        update_norms.append(original_norm)

        # Validate clipping
        assert clipped_norm <= max_norm + 1e-5

    print("\n===== SUMMARY =====")

    print(
        f"Minimum update norm : "
        f"{min(update_norms):.6f}"
    )

    print(
        f"Maximum update norm : "
        f"{max(update_norms):.6f}"
    )

    print(
        f"Average update norm : "
        f"{np.mean(update_norms):.6f}"
    )

    print("===================")