import argparse
import os

from torch import nn, optim

from architectures import ARCHITECTURES, CLASSIFIERS
from functions import generate_data, initialize_model, load_checkpoint, train_model

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--save_dir", default="./checkpoints")
parser.add_argument("--arch", default="densenet121", choices=list(ARCHITECTURES.keys()))
parser.add_argument(
    "--classifier", default="fc_3_dropout", choices=list(CLASSIFIERS.keys())
)
parser.add_argument("--learning_rate", default=0.002)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()


checkpoint_dir = os.path.join(args.save_dir, args.arch, args.classifier)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = os.path.join(checkpoint_dir, "checkpoint.pth")
architecture = ARCHITECTURES[args.arch]
data = dict(generate_data(data_dir=args.data_dir, architecture=architecture))
model = initialize_model(args.arch, args.classifier, data["train"]["class_to_idx"])
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
if os.path.isfile(checkpoint):
    model, optimizer, epoch = load_checkpoint(
        model=model, optimizer=optimizer, filepath=checkpoint
    )
else:
    epoch = 0

train_model(
    model=model,
    criterion=nn.NLLLoss(),
    optimizer=optimizer,
    data=data,
    checkpoint=checkpoint,
    start_epoch=epoch,
    epochs=args.epochs,
)
