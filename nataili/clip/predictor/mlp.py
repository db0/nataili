from typing import Literal

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_size, xcol="emb", ycol="avg_rating", optimizer: Literal["adam", "adamw"] = "adam", lr: float = 1e-3
    ):
        super().__init__()
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        return self.layers(x)
