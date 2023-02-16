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
from typing import List, Literal, Union

import numpy as np
import torch
from PIL import Image

from nataili.cache import Cache
from nataili.clip import ImageEmbed, TextEmbed
from nataili.clip.predictor.mlp import MLP
from nataili.model_manager.clip import ClipModelManager
from nataili.util.logger import logger
from nataili.util.normalized import normalized


class PredictorInference:
    def __init__(
        self,
        vit_model: Literal["ViT-L/14", "ViT-H-14"] = "ViT-L/14",
        predictor_model_path="model.pt",
        device: Literal["cpu", "cuda"] = "cuda",
        gpu_id: int = 0,
    ):
        if device == "cuda":
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.model_manager = ClipModelManager()
        self.predictor_model_path = predictor_model_path
        if not os.path.exists(self.predictor_model_path):
            raise FileNotFoundError(f"{self.predictor_model_path} does not exist")
        if not os.path.isfile(self.predictor_model_path):
            raise FileNotFoundError(f"{self.predictor_model_path} is not a file")
        if vit_model == "ViT-L/14":
            self.model_manager.load("ViT-L/14")
            self.vit = self.model_manager.loaded_models["ViT-L/14"]
            self.predictor = MLP(input_size=768)
        elif vit_model == "ViT-H-14":
            self.model_manager.load("ViT-H-14")
            self.vit = self.model_manager.loaded_models["ViT-H-14"]
            self.predictor = MLP(input_size=1024)
        else:
            raise NotImplementedError
        state_dict = torch.load(self.predictor_model_path, map_location=self.device)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)
        self.cache_text = Cache(vit_model, cache_parentname="embeds", cache_subname="text")
        self.cache_image = Cache(vit_model, cache_parentname="embeds", cache_subname="image")
        self.text_embed = TextEmbed(self.vit, self.cache_text)
        self.image_embed = ImageEmbed(self.vit, self.cache_image)

    def _image_features(self, image: str):
        """
        Cache image features and return them as a numpy array
        :param image: Image to embed, either path or PIL.Image.Image
        :return: image embed as numpy array
        """
        if os.path.exists(image):
            cached = self.cache_image.get(file=os.path.basename(image))
            if cached is not None:
                return np.load(cached)
            try:
                image_hash = self.image_embed(filename=os.path.basename(image), directory=os.path.dirname(image))
                image_embed_array = np.load(f"{self.cache_image.cache_dir}/{image_hash}.npy")
                return image_embed_array
            except Exception as e:
                logger.error(f"Could not open image {image} with error {e}")
                return None
        else:
            logger.error(f"Could not find image {image} in cache")
            return None

    def __call__(self, images: Union[str, List[str], Image.Image, List[Image.Image]]):
        if isinstance(images, str):
            images = [images]
        elif isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, list):
            if isinstance(images[0], str):
                pass
            elif isinstance(images[0], Image.Image):
                pass
            else:
                logger.error(f"Invalid type {type(images[0])} in images list")
                exit(1)
        else:
            logger.error(f"Invalid type {type(images)} for images")
            exit(1)
        predictions = []
        for idx, image in enumerate(images):
            image_embed_array = self._image_features(image)
            if image_embed_array is None:
                logger.error(f"Could not get image features for {image}")
                continue
            image_embed_array = normalized(image_embed_array)
            image_embed = torch.from_numpy(image_embed_array).to(self.device)
            with torch.no_grad():
                prediction = self.predictor(image_embed)
            prediction = prediction[0][0].detach().cpu().numpy()
            logger.info(f"Prediction: {prediction}")
            if isinstance(image, str):
                predictions.append({"image": image, "prediction": prediction})
            else:
                predictions.append({"image": f"image_{idx}", "prediction": prediction})
        return predictions
