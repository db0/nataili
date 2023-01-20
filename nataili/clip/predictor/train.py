import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from nataili.clip.predictor.mlp import MLP


class TrainPredictor:
    def __init__(
        self,
        x: str,
        y: str,
        vit_model: Literal["ViT-L/14", "ViT-H-14"] = "ViT-L/14",
        output_directory: str = "output",
        save_name: str = "model.pth",
        batch_size: int = 2048,
        lr: float = 1e-3,
        optimizer: Literal["adam", "adamw"] = "adam",
        validation_percentage: float = 0.05,
        number_of_train_epochs: int = 100,
        number_of_validation_samples: int = 10,
        device: Literal["cpu", "cuda"] = "cuda",
        gpu_id: int = 0,
    ):
        self.x = x
        self.y = y
        self.output_directory = output_directory
        self.save_name = f"{output_directory}/{save_name}"
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.validation_percentage = validation_percentage
        self.number_of_train_epochs = number_of_train_epochs
        self.number_of_validation_samples = number_of_validation_samples
        if device == "cuda":
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        if vit_model == "ViT-L/14":
            self.model = MLP(input_size=768, optimizer=self.optimizer, lr=self.lr).to(self.device)
        elif vit_model == "ViT-H-14":
            self.model = MLP(input_size=1024, optimizer=self.optimizer, lr=self.lr).to(self.device)
        else:
            raise NotImplementedError
        self.best_loss = 999
        os.makedirs(self.output_directory, exist_ok=True)
        if os.path.exists(self.save_name):
            raise FileExistsError(f"{self.save_name} already exists")

    def __call__(self):
        try:
            x = np.load(self.x)
        except Exception:
            raise Exception(f"Could not load {self.x}")
        try:
            y = np.load(self.y)
        except Exception:
            raise Exception(f"Could not load {self.y}")
        train_border = int(len(x) * (1 - self.validation_percentage))
        try:
            train_tensor_x = torch.Tensor(x[:train_border])
            train_tensor_y = torch.Tensor(y[:train_border])
            train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            validation_tensor_x = torch.Tensor(x[train_border:])
            validation_tensor_y = torch.Tensor(y[train_border:])
            validation_dataset = TensorDataset(validation_tensor_x, validation_tensor_y)
            validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size)
        except Exception:
            raise Exception("Could not create train and validation data")
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        self.model.train()
        for epoch in range(self.number_of_train_epochs):
            losses = []
            losses2 = []
            for batch_number, input_data in enumerate(train_loader):
                self.model.optimizer.zero_grad()
                x, y = input_data
                x = x.to(self.device).float()
                y = y.to(self.device)
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                losses.append(loss.item())
                self.model.optimizer.step()
                if batch_number % 1000 == 0:
                    logger.info("\tEpoch %d | Batch %d | Loss %6.2f" % (epoch, batch_number, loss.item()))
        logger.info("Epoch %d | Loss %6.2f" % (epoch, sum(losses) / len(losses)))
        losses = []
        losses2 = []
        for batch_number, input_data in enumerate(validation_loader):
            x, y = input_data
            x = x.to(self.device).float()
            y = y.to(self.device)
            output = self.model(x)
            loss = criterion(output, y)
            lossMAE = criterion2(output, y)
            losses.append(loss.item())
            losses2.append(lossMAE.item())
            if batch_number % 1000 == 0:
                logger.info("\tEpoch %d | Batch %d | Loss %6.2f" % (epoch, batch_number, loss.item()))
                logger.info(
                    "\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f" % (epoch, batch_number, lossMAE.item())
                )
        logger.info("Validation - Epoch %d | MSE Loss %6.2f" % (epoch, sum(losses) / len(losses)))
        logger.info("Validation - Epoch %d | MAE Loss %6.2f" % (epoch, sum(losses2) / len(losses2)))
        if sum(losses) / len(losses) < self.best_loss:
            logger.info("New best MAE loss %6.2f. Saving model." % (sum(losses) / len(losses)))
            self.best_loss = sum(losses) / len(losses)
            try:
                torch.save(self.model.state_dict(), self.save_name)
            except Exception:
                raise RuntimeError("Could not save model.")
        try:
            torch.save(self.model.state_dict(), self.save_name)
        except Exception:
            raise RuntimeError("Could not save model.")
        logger.info(f"Model saved to {self.save_name}")
        logger.info(f"Best MAE loss {self.best_loss}")
        logger.info("Done training.")
        logger.info("Validating model.")
        self.model.eval()
        output = self.model(x[: self.number_of_validation_samples].to(self.device))
        logger.info(f"Output size: {output.size()}")
        logger.info(f"Output: {output}")
