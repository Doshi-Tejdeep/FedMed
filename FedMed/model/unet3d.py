import torch
import torch.nn as nn
from monai.networks.nets import UNet


def create_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )

    return model


if __name__ == "__main__":
    model = create_model()

    print("FedMed 3D U-Net")
    print(model)

    x = torch.randn(1, 1, 64, 64, 64)

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
