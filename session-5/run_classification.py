import torch
import itertools

import numpy as np
import torch.nn as nn
from time import time
from typing import Tuple

from tensorboard_TODO import TensorboardLogger
from wandb_TODO import WandbLogger
from argparse import Namespace

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import DatasetType, LoggerType

criterion = nn.CrossEntropyLoss()


def forward_image(
    model: nn.Module,
    data: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of a batch of images through the model

    Parameters
    ----------
    model : nn.Module
        Model being trained
    image_batch : torch.Tensor
        Image batch to forward pass

    """
    images, labels = data
    predictions = model(images)
    return predictions, labels


def forward_step(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    dataset_type: DatasetType, 
    optimizer: torch.optim.Optimizer,
):
    """Forward pass of a batch of images through the model"""
    fig = None
    loss_list = []
    acc_list = []
    for i, data in enumerate(loader):
        if dataset_type == DatasetType.TRAIN:
            preds, labels = forward_image(model, data)
            loss = criterion(preds, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds, labels = forward_image(model, data)
                loss = criterion(preds, labels.long())
                if i == 0:
                    fig = log_confusion_matrix(preds, labels)
        acc = compute_accuracy(preds, labels)
        loss_list.append(loss.item())
        acc_list.append(acc)

    return loss_list, acc_list, fig



def run_classification(
    args: Namespace,
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader,
):

    if args.log_framework.upper() == LoggerType.TENSORBOARD.name:
        logger = TensorboardLogger(args.task)
    elif args.log_framework.upper() == LoggerType.WANDB.name:
        logger = WandbLogger(args.task, model)
    else:
        raise ValueError(f"Log framework {args.log_framework} not supported")

    logger.log_model_graph(model, train_loader)

    ini = time()
    for epoch in range(args.n_epochs):

        """
        We are first running the evaluation step before a single training.
        Can you think of a reason on why we are doing this?
        ...
        In the validation step we log the confusion matrix. We want to verify 
        that at the beginning, without any tuning of the parameters, our network
        classifies the sampels randomly. In our logger we'll be able to see how
        the classifications get better over time.
        """
        model.eval()
        val_loss, val_acc, fig = forward_step(model, val_loader, DatasetType.VALIDATION, optimizer)
        val_loss_avg = np.mean(val_loss)
        val_acc_avg = np.mean(val_acc)

        model.train()
        train_loss, train_acc, _ = forward_step(model, train_loader, DatasetType.TRAIN, optimizer)
        train_loss_avg = np.mean(train_loss)
        train_acc_avg = np.mean(train_acc)

        logger.log_classification_training(
            epoch, train_loss_avg, val_loss_avg, train_acc_avg, val_acc_avg, fig
        )

        print(
            f"Epoch [{epoch} / {args.n_epochs}] average classification error: {train_loss_avg}")

    print(f"Training took {round(time() - ini, 2)} seconds")


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    pred_labels = preds.argmax(dim=1, keepdim=True)
    acc = pred_labels.eq(labels.view_as(pred_labels)).sum().item() / len(
        pred_labels)
    return acc


def log_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor) -> plt.Figure:
    predictions = preds.argmax(dim=1, keepdim=True)

    cm = confusion_matrix(labels.cpu(), predictions.cpu())
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='none', cmap=plt.cm.Blues)

    plt.colorbar()
    tick_marks = np.arange(10)

    plt.xticks(tick_marks, np.arange(0, 10))
    plt.yticks(tick_marks, np.arange(0, 10))

    plt.tight_layout()
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion matrix")
    return fig