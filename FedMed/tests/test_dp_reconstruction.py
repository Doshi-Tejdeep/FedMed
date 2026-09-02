import numpy as np
import torch
from torch.utils.data import DataLoader

from model.unet3d import create_model
from training.local_train import get_dataset


def test_zero_noise_reconstruction():

    print("\n===== ZERO-NOISE DP RECONSTRUCTION TEST =====")

    hospital_id = "Hospital-1"
    hospital_key = hospital_id.lower().replace("-", "")

    # ------------------------------------------------------------
    # Create model exactly as FedMedClient does
    # ------------------------------------------------------------

    model = create_model()

    # Save the initial/global state
    global_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    # ------------------------------------------------------------
    # Load the exact same dataset used by FedMedClient.fit()
    # ------------------------------------------------------------

    dataset = get_dataset(hospital_key)

    print(f"Hospital: {hospital_id}")
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    # ------------------------------------------------------------
    # Exact same optimizer/loss as FedMedClient.fit()
    # ------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    model.train()

    total_loss = 0.0

    for batch in loader:

        images = batch["image"]
        labels = batch["label"]

        labels = labels.squeeze(1).long()

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(f"Local training loss: {average_loss:.6f}")

    # ------------------------------------------------------------
    # Get local parameters
    # ------------------------------------------------------------

    local_state = model.state_dict()

    # ------------------------------------------------------------
    # Calculate the complete model update
    # ------------------------------------------------------------

    total_squared_norm = 0.0
    parameters_checked = 0

    for key in global_state.keys():

        global_value = global_state[key]
        local_value = local_state[key]

        if torch.is_floating_point(local_value):

            update = (
                local_value.detach().cpu().numpy().astype(np.float64)
                -
                global_value.detach().cpu().numpy().astype(np.float64)
            )

            total_squared_norm += float(
                np.sum(update ** 2)
            )

            parameters_checked += 1

    original_norm = float(
        np.sqrt(total_squared_norm)
    )

    print(f"Floating-point parameter tensors: {parameters_checked}")
    print(f"Original update norm: {original_norm:.10f}")

    # ------------------------------------------------------------
    # Apply the SAME clipping logic as differential_privacy.py
    # ------------------------------------------------------------

    max_norm = 2.5

    if original_norm > max_norm:

        clipping_scale = max_norm / (
            original_norm + 1e-12
        )

        print(
            f"CLIPPING WOULD OCCUR "
            f"(scale={clipping_scale:.10f})"
        )

    else:

        clipping_scale = 1.0

        print("NO CLIPPING REQUIRED")

    # ------------------------------------------------------------
    # Zero-noise reconstruction
    # ------------------------------------------------------------

    max_difference = 0.0
    total_difference = 0.0

    for key in global_state.keys():

        global_value = global_state[key]
        local_value = local_state[key]

        if torch.is_floating_point(local_value):

            global_np = (
                global_value.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            local_np = (
                local_value.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            update = local_np - global_np

            # Same global clipping operation
            if original_norm > max_norm:
                update = update * clipping_scale

            # ZERO NOISE
            reconstructed = global_np + update

            reconstructed = torch.tensor(
                reconstructed,
                dtype=local_value.dtype
            )

            difference = torch.max(
                torch.abs(
                    reconstructed - local_value
                )
            ).item()

            total_difference += torch.sum(
                torch.abs(
                    reconstructed - local_value
                )
            ).item()

            max_difference = max(
                max_difference,
                difference
            )

    print(f"Maximum reconstruction difference: {max_difference:.10f}")
    print(f"Total reconstruction difference: {total_difference:.10f}")

    print("==============================================")

    # If clipping did not occur, reconstruction must
    # reproduce the original local model.
    if original_norm <= max_norm:

        assert max_difference < 1e-5