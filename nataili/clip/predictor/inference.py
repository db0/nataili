import os
from typing import List, Literal, Union

import numpy as np
import torch
from PIL import Image

from nataili.cache import Cache
from nataili.clip import ImageEmbed, TextEmbed
from nataili.clip.predictor import MLP
from nataili.model_manager import ClipModelManager
from nataili.util import logger, normalized


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

    def _image_features(self, image: Union[str, Image.Image]):
        """
        Cache image features and return them as a numpy array
        :param image: Image to embed, either path or PIL.Image.Image
        :return: image embed as numpy array
        """
        if isinstance(image, str):
            if os.path.exists(image):
                try:
                    image = Image.open(image)
                except Exception:
                    raise Exception(f"Could not open image {image}")
            else:
                raise Exception(f"Could not find image {image}")
        image = image.convert("RGB")
        image_hash = self.image_embed(image)
        self.cache_image.flush()
        image_embed_array = np.load(f"{self.cache_image.cache_dir}/{self.cache_image.kv[image_hash]}.npy")
        return image_embed_array

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
                raise Exception(f"Invalid type {type(images[0])} in images list")
        else:
            raise Exception(f"Invalid type {type(images)} for images")
        predictions = []
        for idx, image in enumerate(images):
            image_embed_array = self._image_features(image)
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
