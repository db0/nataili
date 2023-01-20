import glob
import os
from typing import List, Literal, Union

import numpy as np
import torch
from PIL import Image

from nataili.cache import Cache
from nataili.clip import ImageEmbed, TextEmbed
from nataili.model_manager import ClipModelManager
from nataili.util import logger, normalized


class PredictorPrepare:
    def __init__(self):
        self.model_manager = None
        self.model_name = None
        self.image_embed = None
        self.text_embed = None
        self.cache_text = None
        self.cache_image = None
        self.x = []
        self.y = []

    def __call__(
        self,
        model_name: Literal["ViT-L/14", "ViT-H-14"] = "ViT-L/14",
        input_directory: str = "input",
        output_directory: str = "output",
        rating_source: str = "directory",
        rating_type: Literal["float", "string"] = "float",
    ):
        """
        :param model_name: Name of model to use
        :param input_directory: Directory to read input from
        :param output_directory: Directory to write output to
        :param rating_source: Source of ratings:
            "directory": Ratings are in directory names, e.g. "input/1.0/1.jpg"
            "filename": Ratings are in a file with the same name as the image, e.g. "input/1.jpg" with "1.jpg" containing "1.0"
        """
        self.model_name = model_name
        self.model_manager = ClipModelManager()
        self.model_manager.load(self.model_name)
        self.cache_text = Cache(self.model_name, cache_parentname="embeds", cache_subname="text")
        self.cache_image = Cache(self.model_name, cache_parentname="embeds", cache_subname="image")
        self.text_embed = TextEmbed(self.model_manager.loaded_models[self.model_name], self.cache_text)
        self.image_embed = ImageEmbed(self.model_manager.loaded_models[self.model_name], self.cache_image)

        if rating_source == "directory":
            self._prepare_from_directory(input_directory=input_directory, rating_type=rating_type)
        elif rating_source == "filename":
            self._prepare_from_filename(input_directory=input_directory, rating_type=rating_type)
        else:
            raise NotImplementedError
        try:
            self.x = np.vstack(self.x)
        except Exception:
            raise Exception("Could not stack x")
        try:
            self.y = np.vstack(self.y)
        except Exception:
            raise Exception("Could not stack y")
        logger.info(f"Shape of x: {self.x.shape}")
        logger.info(f"Shape of y: {self.y.shape}")
        logger.info(f"Saving x to {output_directory}/x.npy")
        try:
            np.save(f"{output_directory}/x.npy", self.x)
        except Exception:
            raise Exception(f"Could not save x to {output_directory}/x.npy")
        logger.info(f"Saving y to {output_directory}/y.npy")
        try:
            np.save(f"{output_directory}/y.npy", self.y)
        except Exception:
            raise Exception(f"Could not save y to {output_directory}/y.npy")

    def _prepare_from_filename(self, input_directory: str, rating_type: Literal["float", "string"] = "float"):
        """
        :param input_directory: Directory to read input from
        """
        for file in os.listdir(input_directory):
            if not os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            file_path = os.path.join(input_directory, file)
            if os.path.isfile(file_path):
                file_path_no_ext = os.path.splitext(file_path)[0]
                rating_file_path = f"{file_path_no_ext}.txt"
                if os.path.exists(rating_file_path):
                    with open(rating_file_path, "r") as f:
                        rating = f.read()
                        self._prepare_from_file(file_path=file_path, rating=rating)
                else:
                    logger.warning(f"Could not find rating file {rating_file_path}")

    def _prepare_from_directory(self, input_directory: str, rating_type: Literal["float", "string"] = "float"):
        """
        :param input_directory: Directory to read input from
        """
        for directory in os.listdir(input_directory):
            directory_path = os.path.join(input_directory, directory)
            if os.path.isdir(directory_path):
                logger.info(f"Processing folder: {directory}")
                for file in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path):
                        self._prepare_from_file(file_path=file_path, rating=directory, rating_type=rating_type)

    def _prepare_from_file(self, file_path: str, rating: str, rating_type: Literal["float", "string"] = "float"):
        """
        :param file_path: Path to file to prepare
        :param rating: Rating to use for file
        """
        if rating_type == "float":
            try:
                rating = float(rating)
                logger.debug(f"Converted rating {rating} to float")
            except Exception:
                raise Exception(f"Could not parse rating {rating} as float")
        logger.info(f"Processing file: {file_path} with rating {rating}")
        self.x.append(normalized(self._image_features(file_path)))
        self.y.append(self._y(rating))

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

    def _y(self, rating: Union[str, float]):
        """
        :param rating: Rating to embed
        :return: rating
        """
        if isinstance(rating, float):
            y_ = np.zeros((1, 1))
        elif isinstance(rating, str):
            y_ = np.zeros((1, 1), dtype=np.str_)
        y_[0][0] = rating
        return y_
