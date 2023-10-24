import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from time import time
from typing import Tuple
from utils import DatasetType, LoggerType
from argparse import Namespace

from tensorboard_TODO import TensorboardLogger
from wandb_TODO import WandbLogger


def forward_image(
    model: nn.Module, 
    image_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of a batch of images through the model

    Parameters
    ----------
    model : nn.Module
        Model being trained
    image_batch : torch.Tensor
        Image batch to forward pass

    """
    image_batch_recon = model(image_batch)
    return F.mse_loss(image_batch_recon, image_batch), image_batch_recon


def forward_step(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    dataset_type: DatasetType, 
    optimizer: torch.optim.Optimizer,
):
    reconstruction_grid = None
    loss_list = []
    for i, (image_batch, _) in enumerate(loader):
        if dataset_type == DatasetType.TRAIN:
            loss, recon = forward_image(model, image_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, recon = forward_image(model, image_batch)
                if i == 0:
                    # make_grid returns an image tensor from a batch of data (https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid)
                    reconstruction_grid = make_grid(recon)
        loss_list.append(loss.item())
    return loss_list, reconstruction_grid


def run_reconstruction(
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
        In the validation step we log a batch of reconstructed images at the output
        of our AutoEncoder. We want to verify that at the beginning, without any
        tuning of the parameters, our network returns just noise. In our logger
        we'll be able to see how these reconstructions get better over time.
        """
        model.eval()
        val_loss, reconstruction_grid = forward_step(model, val_loader, DatasetType.VALIDATION, optimizer)
        val_loss_avg = np.mean(val_loss)

        model.train()
        train_loss, _ = forward_step(model, train_loader, DatasetType.TRAIN, optimizer)
        train_loss_avg = np.mean(train_loss)

        logger.log_reconstruction_training(
            model, epoch, train_loss_avg, val_loss_avg, reconstruction_grid
        )

        print(
            f"Epoch [{epoch} / {args.n_epochs}] average reconstruction error: {train_loss_avg}")

    print(f"Training took {round(time() - ini, 2)} seconds")

    logger.log_embeddings(model, train_loader)