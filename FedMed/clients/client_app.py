import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.client import ClientApp

from clients.client import FedMedClient


app = ClientApp()


def get_hospital_id(context):
    return context.node_config.get(
        "hospital-id",
        "Hospital-1",
    )


@app.train()
def train(msg: Message, context) -> Message:
    hospital_id = get_hospital_id(context)

    client = FedMedClient(hospital_id)

    # Load global parameters
    arrays = msg.content["arrays"]
    client.model.load_state_dict(
        arrays.to_torch_state_dict(),
        strict=True,
    )

    print(f"{hospital_id}: Starting local training")

    dataset = client._get_dataset()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        client.model.parameters(),
        lr=0.001,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    client.model.train()

    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(client.device)
        labels = batch["label"].to(client.device)
        labels = labels.squeeze(1).long()

        optimizer.zero_grad()

        outputs = client.model(images)

        loss = loss_function(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"{hospital_id}: "
        f"Local training complete - "
        f"Loss: {average_loss:.4f}"
    )

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(
                    torch_state_dict=client.model.state_dict()
                ),
                "metrics": MetricRecord(
                    {
                        "num-examples": len(dataset),
                        "loss": float(average_loss),
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context) -> Message:
    hospital_id = get_hospital_id(context)

    client = FedMedClient(hospital_id)

    arrays = msg.content["arrays"]

    client.model.load_state_dict(
        arrays.to_torch_state_dict(),
        strict=True,
    )

    print(f"{hospital_id}: Evaluating model")

    dataset = client._get_dataset()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    client.model.eval()

    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(client.device)

            labels = batch["label"].to(client.device)

            outputs = client.model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            labels = labels.squeeze(1).long()

            predictions = (predictions > 0).float()
            labels = (labels > 0).float()

            from evaluation.dice import dice_score

            dice = dice_score(
                predictions,
                labels,
            )

            total_dice += dice
            total_samples += 1

    average_dice = total_dice / total_samples

    print(
        f"{hospital_id}: "
        f"Dice Score: "
        f"{average_dice:.4f}"
    )

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "num-examples": total_samples,
                        "dice": float(average_dice),
                    }
                )
            }
        ),
        reply_to=msg,
    )