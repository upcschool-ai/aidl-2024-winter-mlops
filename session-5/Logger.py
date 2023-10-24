import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt

class Logger:

    def log_reconstruction_training(
        self,
        model: nn.Module,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        reconstruction_grid: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def log_classification_training(
        self,
        epoch: int,
        train_loss_avg: np.ndarray,
        val_loss_avg: np.ndarray,
        train_acc_avg: np.ndarray,
        val_acc_avg: np.ndarray,
        fig: plt.Figure,
    ):
        raise NotImplementedError

    def log_model_graph(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
    ):
        raise NotImplementedError

    def log_embeddings(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
    ):
        raise NotImplementedError
