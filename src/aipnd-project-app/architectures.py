from collections import OrderedDict

from torch import nn

ARCHITECTURES = {
    "densenet121": {
        "resize": 256,
        "cropsize": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "layers": [1024, 512, 256, 102]
    },
    "vgg13": {
        "resize": 256,
        "cropsize": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "vgg16": {
        "resize": 256,
        "cropsize": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "layers": [25088, 4096, 40, 102]
    },
    "alexnet": {
        "resize": 256,
        "cropsize": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def fc_3_dropout():
    return nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(1024, 512)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(0.2)),
                ("fc2", nn.Linear(512, 256)),
                ("dropout", nn.Dropout(0.2)),
                ("relu", nn.ReLU()),
                ("fc3", nn.Linear(256, 102)),
                ("dropout", nn.Dropout(0.2)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )


CLASSIFIERS = {
    "fc_3_dropout": nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(1024, 512)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(0.2)),
                ("fc2", nn.Linear(512, 256)),
                ("dropout", nn.Dropout(0.2)),
                ("relu", nn.ReLU()),
                ("fc3", nn.Linear(256, 102)),
                ("dropout", nn.Dropout(0.2)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
}

