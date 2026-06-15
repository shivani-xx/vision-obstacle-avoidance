# src/models.py

import torch
import torch.nn as nn
import torchvision.models as models


NUM_CLASSES = 5


def get_model(model_name, num_classes=NUM_CLASSES):

    model_name = model_name.lower()

    if model_name == 'mobilenetv3':

        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes
        )

    elif model_name == 'efficientnet':

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes
        )

    elif model_name == 'squeezenet':

        model = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.DEFAULT
        )

        model.classifier[1] = nn.Conv2d(
            512,
            num_classes,
            kernel_size=1
        )

    else:

        raise ValueError(
            f'Unknown model: {model_name}'
        )

    return model


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == '__main__':

    for name in [
        'mobilenetv3',
        'efficientnet',
        'squeezenet'
    ]:

        model = get_model(name)

        dummy = torch.randn(
            1,
            3,
            224,
            224
        )

        out = model(dummy)

        print(
            f'{name}: '
            f'output={out.shape}, '
            f'params={count_parameters(model):,}'
        )