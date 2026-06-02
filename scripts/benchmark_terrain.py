import torch
import torch.nn as nn
import torchvision.models as models
import time
import numpy as np
import json
import os

torch.set_num_threads(4)

device = torch.device('cpu')


def get_model(name, num_classes=4):

    if name == 'squeezenet':

        model = models.squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(
            512,
            num_classes,
            kernel_size=1
        )

    elif name == 'mobilenetv3':

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes
        )

    elif name == 'efficientnet':

        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes
        )

    else:
        raise ValueError(name)

    return model


def benchmark_model(name, input_size,
                    warmup=20,
                    num_runs=100):

    model = get_model(name).to(device)

    model.eval()

    dummy = torch.randn(
        1,
        3,
        input_size,
        input_size
    ).to(device)

    with torch.no_grad():

        for _ in range(warmup):

            _ = model(dummy)

    latencies = []

    with torch.no_grad():

        for _ in range(num_runs):

            start = time.perf_counter()

            _ = model(dummy)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

    latencies = np.array(latencies)

    return {

        'model': name,

        'resolution': input_size,

        'mean_ms': round(
            float(latencies.mean()),
            2
        ),

        'std_ms': round(
            float(latencies.std()),
            2
        ),

        'p99_ms': round(
            float(np.percentile(
                latencies,
                99
            )),
            2
        ),

        'fps': round(
            1000 / float(latencies.mean()),
            1
        )
    }


chosen_models = [
    'mobilenetv3',
    'efficientnet',
    'squeezenet'
]

chosen_resolutions = [
    128,
    224
]

results = []

print(
    f'{"Model":<15}'
    f'{"Res":>8}'
    f'{"Mean":>10}'
    f'{"P99":>10}'
    f'{"FPS":>10}'
)

print('-' * 55)

for model_name in chosen_models:

    for size in chosen_resolutions:

        r = benchmark_model(
            model_name,
            size
        )

        results.append(r)

        print(
            f'{model_name:<15}'
            f'{size:>8}'
            f'{r["mean_ms"]:>10}'
            f'{r["p99_ms"]:>10}'
            f'{r["fps"]:>10}'
        )

os.makedirs(
    'results',
    exist_ok=True
)

with open(
    'results/capstone_latency.json',
    'w'
) as f:

    json.dump(
        results,
        f,
        indent=2
    )

print('\nSaved to results/capstone_latency.json')