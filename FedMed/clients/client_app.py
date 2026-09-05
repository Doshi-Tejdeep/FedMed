import torch

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from flwr.clientapp.mod import fixedclipping_mod
from flwr.common import MessageType


from clients.client import FedMedClient
from evaluation.dice import dice_score


def train_only_fixedclipping_mod(msg, ctxt, call_next):
    if msg.metadata.message_type != MessageType.TRAIN:
        return call_next(msg, ctxt)

    return fixedclipping_mod(msg, ctxt, call_next)

app = ClientApp(
    mods=[train_only_fixedclipping_mod],
)


def get_hospital_id(context: Context) -> str:
    return context.node_config.get(
        "hospital-id",
        "Hospital-1",
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    hospital_id = get_hospital_id(context)

    client = FedMedClient(hospital_id)

    # Load the global model sent by the ServerApp
    client.model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict(),
        strict=True,
    )

    dataset = client._get_dataset()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    learning_rate = float(
        msg.content["config"].get(
            "learning-rate",
            0.001,
        )
    )

    optimizer = torch.optim.Adam(
        client.model.parameters(),
        lr=learning_rate,
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
                    client.model.state_dict()
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
def evaluate(msg: Message, context: Context) -> Message:
    hospital_id = get_hospital_id(context)

    client = FedMedClient(hospital_id)

    client.model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict(),
        strict=True,
    )

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
            labels = labels.squeeze(1).long()

            outputs = client.model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

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
        f"{hospital_id}: "
        f"Dice Score: {average_dice:.4f}"
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