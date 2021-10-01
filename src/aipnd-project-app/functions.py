import json
import time

import torch
from torchvision import datasets, models, transforms

from architectures import CLASSIFIERS


def generate_data_directories(data_dir):
    for phase in ["train", "valid", "test"]:
        yield phase, data_dir + f"/{phase}"


def get_category_to_names(path):
    with open(path, "r") as f:
        cat_to_name = json.load(f)
    return cat_to_name


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def train_transform(cropsize, mean, std):
    return transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(cropsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def test_valid_transform(resize, cropsize, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def get_transforms(architecture):
    return {
        "train": train_transform(
            architecture["cropsize"], architecture["mean"], architecture["std"]
        ),
        "valid": test_valid_transform(
            architecture["resize"],
            architecture["cropsize"],
            architecture["mean"],
            architecture["std"],
        ),
        "test": test_valid_transform(
            architecture["resize"],
            architecture["cropsize"],
            architecture["mean"],
            architecture["std"],
        ),
    }


def generate_datasets(data_dir, architecture):
    data_transforms = get_transforms(architecture=architecture)
    for phase, path in generate_data_directories(data_dir=data_dir):
        yield phase, datasets.ImageFolder(path, transform=data_transforms[phase])


def generate_data(data_dir, architecture):
    for phase, dataset in generate_datasets(data_dir, architecture):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=phase == "train"
        )
        phase_data = {
            "loader": loader,
            "size": len(dataset),
            "class_to_idx": dataset.class_to_idx,
        }
        yield phase, phase_data


def initialize_model(architecture, classifier, class_to_idx):
    model = getattr(models, architecture)
    model = model(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = CLASSIFIERS[classifier]
    model.class_to_idx = class_to_idx
    model.idx_to_class = {v: k for k, v in class_to_idx.items()}
    model.best_accuracy = 0.0
    return model


def save_checkpoint(model, optimizer, filepath, epoch):
    checkpoint = {
        "epoch": epoch,
        "class_to_idx": model.class_to_idx,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": model.best_accuracy,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.to(get_device())
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.best_accuracy = checkpoint["best_accuracy"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1
    return model, optimizer, epoch


def model_phase(phase, model, criterion, optimizer, data, device):
    if phase == "train":
        model.train()  # Set model to training mode (activate dropout a.o.)
    else:
        model.eval()  # Set model to evaluate mode (ignore dropout a.o.)
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in data[phase]["loader"]:

        # Move input and label tensors to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history only in train phase
        with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            # backward + optimize only if in training phase
            if phase == "train":
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(equals.type(torch.FloatTensor))

    phase_loss = running_loss / data[phase]["size"]
    phase_accuracy = running_corrects.double() / data[phase]["size"]

    print(f"{phase.capitalize()} Loss: {phase_loss:.4f} Accuracy: {phase_accuracy:.4f}")

    return phase_accuracy


def train_model(
    model, criterion, optimizer, data, checkpoint, epochs=10, start_epoch=0
):

    start = time.time()
    phases = ["train", "valid"]
    device = get_device()
    model.to(device)

    for epoch in range(start_epoch, start_epoch + epochs):
        print("Epoch {}/{}".format(epoch + 1, start_epoch + epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            phase_accuracy = model_phase(
                phase=phase,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                data=data,
                device=device,
            )

            # save the checkpoint if accuracy improved
            if phase == "valid" and phase_accuracy > model.best_accuracy:
                model.best_accuracy = phase_accuracy
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    filepath=checkpoint.replace(".pth", "_best.pth"),
                    epoch=epoch,
                )
        # save the checkpoint
        save_checkpoint(
            model=model, optimizer=optimizer, filepath=checkpoint, epoch=epoch,
        )
        training_time = time.time() - start
        print(f"Training time: {training_time // 60:.0f}m {training_time % 60:.0f}s")
        print("Best val Acc: {:4f}\n".format(model.best_accuracy))

    # load best model weights
    return model, optimizer
