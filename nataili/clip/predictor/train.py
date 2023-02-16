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
import os
import random
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nataili.clip.predictor.mlp import H14_MLP, H14_MLP_2, MLP
from nataili.util.logger import logger


class TrainPredictor:
    def __init__(
        self,
        x: str,
        y: str,
        batch_size: int = 256,
        vit_model: Literal["ViT-L/14", "ViT-H-14"] = "ViT-L/14",
        output_directory: str = "output",
        save_name: str = "model.pth",
        lr: float = 1e-3,
        optimizer: Literal["adam", "adamw"] = "adam",
        validation_percentage: float = 0.05,
        number_of_train_epochs: int = 100,
        number_of_validation_samples: int = 10,
        device: Literal["cpu", "cuda"] = "cuda",
        gpu_id: int = 0,
        project_name: str = "a e s t h e t i c s",
        wandb_entity: str = "hlky",
        mlp_model: Union[MLP, H14_MLP, H14_MLP_2] = MLP,
    ):
        self.x = None
        self.y = None
        self.val_x = None
        self.val_y = None
        self.batch_size = batch_size
        self.output_directory = output_directory
        self.save_name = f"{output_directory}/{save_name}"
        self.lr = lr
        self.optimizer = optimizer
        self.validation_percentage = validation_percentage
        self.number_of_train_epochs = number_of_train_epochs
        self.number_of_validation_samples = number_of_validation_samples
        self.mlp_model = mlp_model
        if device == "cuda":
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        if vit_model == "ViT-L/14":
            self.model = self.mlp_model(input_size=768).to(self.device)
        elif vit_model == "ViT-H-14":
            self.model = self.mlp_model(input_size=1024).to(self.device)
        else:
            raise NotImplementedError
        self.best_loss = 999
        self.best_acc = 0
        os.makedirs(self.output_directory, exist_ok=True)
        if os.path.exists(self.save_name):
            raise FileExistsError(f"{self.save_name} already exists")
        self.load_dataset(x, y)
        assert len(self.x) == len(self.y)
        self.load_val_dataset()
        assert len(self.val_x) == len(self.val_y)
        self.train_x_tensor = torch.Tensor(self.x).to(self.device)
        self.train_y_tensor = torch.Tensor(self.y).to(self.device)
        self.val_x_tensor = torch.Tensor(self.val_x).to(self.device)
        self.val_y_tensor = torch.Tensor(self.val_y).to(self.device)
        self.train_dataset = TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.val_dataset = TensorDataset(self.val_x_tensor, self.val_y_tensor)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.criterion = nn.MSELoss()
        self.current_epoch = 0
        self.config = {
            "vit": vit_model,
            "images_count": len(self.x),
            "epochs": self.number_of_train_epochs,
            "validation_samples": self.number_of_validation_samples,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "save_name": self.save_name,
        }
        wandb.init(project=project_name, entity=wandb_entity, config=self.config, reinit=True)

    def load_dataset(self, x, y):
        try:
            self.x = np.load(x)
        except Exception as e:
            logger.error(f"Could not load {self.x} - {e}")
            exit(1)
        try:
            self.y = np.load(y)
        except Exception as e:
            logger.error(f"Could not load {self.y} - {e}")
            exit(1)

    def load_val_dataset(self):
        border = int(len(self.x) * self.validation_percentage)
        self.x = self.x[border:]
        self.y = self.y[border:]
        self.val_x = self.x[:border]
        self.val_y = self.y[:border]

    def train_one_epoch(self):
        self.model.train()
        for x, y in self.train_loader:
            self.model.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.model.optimizer.step()
        return loss.item()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
        return loss.item()

    def val_accuracy(self, tolerance=0.5, low=2, high=6):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                y_pred = self.model(x)
                for i in range(len(y_pred)):
                    total += 1
                    if (
                        abs(y_pred[i].item() - y[i].item()) <= tolerance  # predicted is within tolerance of actual
                        or (
                            y[i].item() >= high and y_pred[i].item() >= y[i].item()
                        )  # predicted is higher than actual and actual is higher than high
                        or (
                            y[i].item() <= low and y_pred[i].item() <= y[i].item()
                        )  # predicted is lower than actual and actual is lower than low
                    ):
                        correct += 1
        return correct / total

    def __call__(self):
        for epoch in range(self.number_of_train_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch: {self.current_epoch}")
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            val_acc = self.val_accuracy()
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )
            logger.info(f"Val loss: {val_loss} - Val accuracy: {val_acc}")
            logger.info(f"Train loss: {train_loss}")
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_name)
                logger.info(f"Saved model with val_acc {round(val_acc, 4)} at epoch {epoch}")
        self.model.load_state_dict(torch.load(self.save_name))
        return True
