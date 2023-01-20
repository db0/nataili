from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(pl.LightningModule):
    def __init__(
        self, input_size, xcol="emb", ycol="avg_rating", optimizer: Literal["adam", "adamw"] = "adam", lr: float = 1e-3
    ):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        return self.optimizer
