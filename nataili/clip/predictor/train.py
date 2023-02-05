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
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nataili.clip.predictor.mlp import MLP
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
        validate_set_directory: str = "validate",
        num_heads: int = 8,
        drop: float = 0.1,
    ):
        self.x = None
        self.y = None
        self.test_x_low = None
        self.test_y_low = None
        self.test_x_high = None
        self.test_y_high = None
        self.test_high_images = None
        self.test_low_images = None
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
        self.validate_set_directory = validate_set_directory
        if device == "cuda":
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        if vit_model == "ViT-L/14":
            self.model = MLP(
                input_size=768, optimizer=self.optimizer, lr=self.lr, number_of_heads=num_heads, drop=drop
            ).to(self.device)
        elif vit_model == "ViT-H-14":
            self.model = MLP(
                input_size=1024, optimizer=self.optimizer, lr=self.lr, number_of_heads=num_heads, drop=drop
            ).to(self.device)
        else:
            raise NotImplementedError
        self.best_loss = 999
        self.best_acc = 0
        os.makedirs(self.output_directory, exist_ok=True)
        if os.path.exists(self.save_name):
            raise FileExistsError(f"{self.save_name} already exists")
        self.load_dataset(x, y)
        assert len(self.x) == len(self.y)
        self.load_test_dataset()
        self.load_val_dataset()
        assert len(self.val_x) == len(self.val_y)
        self.train_x_tensor = torch.Tensor(self.x).to(self.device)
        self.train_y_tensor = torch.Tensor(self.y).to(self.device)
        self.val_x_tensor = torch.Tensor(self.val_x).to(self.device)
        self.val_y_tensor = torch.Tensor(self.val_y).to(self.device)
        self.test_x_tensor_high = torch.Tensor(self.test_x_high).to(self.device)
        self.test_y_tensor_high = torch.Tensor(self.test_y_high).to(self.device)
        self.test_x_tensor_low = torch.Tensor(self.test_x_low).to(self.device)
        self.test_y_tensor_low = torch.Tensor(self.test_y_low).to(self.device)
        self.train_dataset = TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.val_dataset = TensorDataset(self.val_x_tensor, self.val_y_tensor)
        self.test_dataset_high = TensorDataset(self.test_x_tensor_high, self.test_y_tensor_high)
        self.test_dataset_high = TensorDataset(self.test_x_tensor_high, self.test_y_tensor_high)
        self.test_dataset_low = TensorDataset(self.test_x_tensor_low, self.test_y_tensor_low)
        self.test_dataset_low = TensorDataset(self.test_x_tensor_low, self.test_y_tensor_low)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size * 2)
        self.test_loader_high = DataLoader(self.test_dataset_high, batch_size=self.batch_size * 2)
        self.test_loader_low = DataLoader(self.test_dataset_low, batch_size=self.batch_size * 2)
        self.criterion = nn.MSELoss()
        self.current_epoch = 0
        self.config = {
            "vit": vit_model,
            "images_count": len(self.x),
            "epochs": self.number_of_train_epochs,
            "validation_samples": self.number_of_validation_samples,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "num_heads": num_heads,
            "dropout": drop,
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

    def load_test_dataset(self):
        high = f"{self.validate_set_directory}/high"
        low = f"{self.validate_set_directory}/low"
        high_x = np.load(os.path.join(high, "x.npy"))
        high_y = np.load(os.path.join(high, "y.npy"))
        high_images = []
        for file in os.listdir(high):
            if file.endswith(".webp"):
                high_images.append(file)
        low_x = np.load(os.path.join(low, "x.npy"))
        low_y = np.load(os.path.join(low, "y.npy"))
        low_images = []
        for file in os.listdir(low):
            if file.endswith(".webp"):
                low_images.append(file)
        self.test_x_high = high_x
        self.test_y_high = high_y
        self.test_x_low = low_x
        self.test_y_low = low_y
        self.test_low_images = low_images
        self.test_high_images = high_images
        logger.info(f"Loaded test dataset with {len(self.test_x_high)} high and {len(self.test_x_low)} low images")
        logger.info(f"{len(self.test_high_images)} high and {len(self.test_low_images)} low images")

    def train_one_epoch(self):
        self.model.train()
        for x, y in self.train_loader:
            self.model.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.model.optimizer.step()
        return loss.item()

    def test(self):
        test_results = []
        self.model.eval()
        with torch.no_grad():
            table = wandb.Table(columns=["image", "predicted", "actual"])
            for x, y in self.test_loader_high:
                y_pred = self.model(x)
                for i in range(len(y_pred)):
                    test_results.append(
                        {"image": self.test_high_images[i], "predicted": y_pred[i].item(), "actual": y[i].item()}
                    )
                    table.add_data(
                        wandb.Image(f"{self.validate_set_directory}/high/{self.test_high_images[i]}"),
                        y_pred[i].item(),
                        y[i].item(),
                    )
            for x, y in self.test_loader_low:
                y_pred = self.model(x)
                for i in range(len(y_pred)):
                    test_results.append(
                        {"image": self.test_low_images[i], "predicted": y_pred[i].item(), "actual": y[i].item()}
                    )
                    table.add_data(
                        wandb.Image(f"{self.validate_set_directory}/low/{self.test_low_images[i]}"),
                        y_pred[i].item(),
                        y[i].item(),
                    )
            wandb.log({"test_results": table, "epoch": self.current_epoch})
        return test_results

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
        return loss.item()

    def test_accuracy(self, tolerance=0.5):
        correct = 0
        total = 0
        total_high = 0
        total_low = 0
        correct_high = 0
        correct_low = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in self.test_loader_high:
                y_pred = self.model(x)
                for i in range(len(y_pred)):
                    total += 1
                    if abs(y_pred[i].item() - y[i].item()) < tolerance or y_pred[i].item() > y[i].item():
                        correct_high += 1
                        correct += 1
                    total_high += 1
            for x, y in self.test_loader_low:
                y_pred = self.model(x)
                for i in range(len(y_pred)):
                    total += 1
                    if abs(y_pred[i].item() - y[i].item()) < tolerance or y_pred[i].item() < 3:
                        correct_low += 1
                        correct += 1
                    total_low += 1
        total_accuracy = correct / total
        high_accuracy = correct_high / total_high
        low_accuracy = correct_low / total_low
        average_accuracy = (high_accuracy + low_accuracy) / 2
        return total_accuracy, high_accuracy, low_accuracy, average_accuracy

    def val_accuracy(self, tolerance=0.5):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                y_pred = self.model(x)
                for i in range(len(y_pred)):
                    total += 1
                    if (
                        abs(y_pred[i].item() - y[i].item()) < tolerance
                        or (y[i].item() > 6 and y_pred[i].item() > y[i].item())
                        or (y[i].item() < 2 and y_pred[i].item() < y[i].item())
                    ):
                        correct += 1
        return correct / total

    def __call__(self):
        for epoch in range(0, self.number_of_train_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch: {self.current_epoch}")
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            total_accuracy, high_accuracy, low_accuracy, average_accuracy = self.test_accuracy()
            val_acc = self.val_accuracy()
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "test_total_accuracy": total_accuracy,
                    "test_high_accuracy": high_accuracy,
                    "test_low_accuracy": low_accuracy,
                    "test_average_accuracy": average_accuracy,
                    "val_accuracy": val_acc,
                }
            )
            logger.info(
                f"Test accuracy: {total_accuracy} - Test high accuracy: {high_accuracy} - Test low accuracy: {low_accuracy} - Test average accuracy: {average_accuracy}"
            )
            logger.info(f"Val loss: {val_loss} - Val accuracy: {val_acc}")
            logger.info(f"Train loss: {train_loss}")
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_name)
                logger.info(f"Saved model with val_acc {round(val_acc, 4)} at epoch {epoch}")
        self.model.load_state_dict(torch.load(self.save_name))
        test_results = self.test()
        return test_results
