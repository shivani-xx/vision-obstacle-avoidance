# src/train.py

import os
import json
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from sklearn.metrics import accuracy_score

from src.models import (
    get_model,
    count_parameters
)

from src.warehouse_dataset import (
    get_dataloaders
)

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0
    preds = []
    targets = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predicted = outputs.argmax(1)

        preds.extend(
            predicted.cpu().numpy()
        )

        targets.extend(
            labels.cpu().numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    epoch_acc = accuracy_score(
        targets,
        preds
    )

    return epoch_loss, epoch_acc


def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predicted = outputs.argmax(1)

            preds.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    acc = accuracy_score(
        targets,
        preds
    )

    return loss, acc


def train_experiment(
    model_name,
    resolution,
    epochs=1
):

    print()
    print('=' * 70)
    print(
        f'{model_name} @ {resolution}'
    )
    print('=' * 70)

    train_loader, val_loader, test_loader, _ = (
        get_dataloaders(
            data_dir='data/warehouse_dataset',
            resolution=resolution,
            batch_size=32
        )
    )

    model = get_model(
        model_name,
        num_classes=5
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=1e-3
    )

    best_val = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start = time.time()

    for epoch in range(epochs):

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_loss, val_acc = (
            evaluate(
                model,
                val_loader,
                criterion
            )
        )

        history['train_loss'].append(
            train_loss
        )

        history['train_acc'].append(
            train_acc
        )

        history['val_loss'].append(
            val_loss
        )

        history['val_acc'].append(
            val_acc
        )

        print(
            f'Epoch {epoch+1:02d}/{epochs}'
            f' | Train Acc={train_acc:.4f}'
            f' | Val Acc={val_acc:.4f}'
        )

        if val_acc > best_val:

            best_val = val_acc

            os.makedirs(
                'models',
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                f'models/{model_name}_{resolution}.pth'
            )

    training_time = (
        time.time()
        - start
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    result = {

        'model': model_name,

        'resolution': resolution,

        'parameters': count_parameters(
            model
        ),

        'best_val_acc': round(
            best_val,
            4
        ),

        'test_acc': round(
            test_acc,
            4
        ),

        'training_time_sec': round(
            training_time,
            1
        )
    }

    return result, history


if __name__ == '__main__':

    experiments = [

        ('mobilenetv3', 128),
        ('mobilenetv3', 224),

        ('efficientnet', 128),
        ('efficientnet', 224),

        ('squeezenet', 128),
        ('squeezenet', 224),
    ]

    all_results = []

    for model_name, resolution in experiments:

        result, history = train_experiment(
            model_name,
            resolution,
            epochs=10
        )

        all_results.append(
            result
        )

    os.makedirs(
        'results',
        exist_ok=True
    )

    with open(
        'results/training_results.json',
        'w'
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    print()
    print('=' * 70)
    print('ALL EXPERIMENTS COMPLETE')
    print('=' * 70)

    for r in all_results:

        print(r)