"""
This file is part of nataili ("Homepage" = "https://github.com/Sygil-Dev/nataili").

Copyright 2022 hlky and Sygil-Dev
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
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
