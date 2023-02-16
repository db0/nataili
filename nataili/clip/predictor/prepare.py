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
from typing import Literal, Union

import numpy as np
from PIL import Image

from nataili.cache import Cache
from nataili.clip.image import ImageEmbed
from nataili.clip.text import TextEmbed
from nataili.model_manager.clip import ClipModelManager
from nataili.util.logger import logger
from nataili.util.normalized import normalized


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
        rating_source: str = "none",
        rating_type: Literal["float", "string"] = "float",
        x_only: bool = False,
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
        self.cache_text = Cache(self.model_name, cache_parentname="embeds", cache_subname="text")
        self.cache_image = Cache(self.model_name, cache_parentname="embeds", cache_subname="image")
        self.model_manager.load(self.model_name)
        self.text_embed = TextEmbed(self.model_manager.loaded_models[self.model_name], self.cache_text)
        self.image_embed = ImageEmbed(self.model_manager.loaded_models[self.model_name], self.cache_image)
        if rating_source == "directory":
            self._prepare_from_directory(input_directory=input_directory, rating_type=rating_type, x_only=x_only)
        elif rating_source == "filename":
            self._prepare_from_filename(input_directory=input_directory, rating_type=rating_type, x_only=x_only)
        elif rating_source == "none":
            self._x_only(input_directory=input_directory)
        else:
            raise NotImplementedError
        try:
            self.x = np.vstack(self.x)
        except Exception as e:
            logger.error(f"Could not stack x: {e}")
            logger.error("Could not stack x")
            exit(1)
        logger.info(f"Shape of x: {self.x.shape}")
        if not x_only:
            try:
                self.y = np.vstack(self.y)
            except Exception:
                logger.error("Could not stack y")
                exit(1)
            logger.info(f"Shape of y: {self.y.shape}")
        if not os.path.exists(output_directory):
            logger.info(f"Creating output directory: {output_directory}")
            os.makedirs(output_directory)
        logger.info(f"Saving x to {output_directory}/x.npy")
        try:
            np.save(f"{output_directory}/x.npy", self.x)
        except Exception:
            logger.error(f"Could not save x to {output_directory}/x.npy")
            exit(1)
        if not x_only:
            logger.info(f"Saving y to {output_directory}/y.npy")
            try:
                np.save(f"{output_directory}/y.npy", self.y)
            except Exception:
                logger.error(f"Could not save y to {output_directory}/y.npy")
                exit(1)

    def save(self, output_directory: str, x: np.ndarray, y: np.ndarray):
        """
        :param output_directory: Directory to write output to
        """
        if not os.path.exists(output_directory):
            logger.info(f"Creating output directory: {output_directory}")
            os.makedirs(output_directory)
        logger.info(f"Saving x to {output_directory}/x.npy")
        try:
            np.save(f"{output_directory}/x.npy", x)
        except Exception:
            logger.error(f"Could not save x to {output_directory}/x.npy")
            exit(1)
        logger.info(f"Saving y to {output_directory}/y.npy")
        try:
            np.save(f"{output_directory}/y.npy", y)
        except Exception:
            logger.error(f"Could not save y to {output_directory}/y.npy")
            exit(1)

    def load(self, input_directory: str) -> np.ndarray:
        """
        :param input_directory: Directory to read input from
        :return: x
        """
        x = np.load(f"{input_directory}/x.npy")
        y = np.load(f"{input_directory}/y.npy")
        return x, y

    def _x_only(self, input_directory: str):
        """
        :param input_directory: Directory to read input from
        """
        for file in os.listdir(input_directory):
            if not os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            file_path = os.path.join(input_directory, file)
            if os.path.isfile(file_path):
                image_features = self._image_features(file_path)
                if image_features is None:
                    return
                self.x.append(normalized(image_features))

    def _prepare_from_filename(
        self, input_directory: str, rating_type: Literal["float", "string"] = "float", x_only: bool = False
    ):
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
                        self._prepare_from_file(
                            file_path=file_path, rating=rating, rating_type=rating_type, x_only=x_only
                        )
                else:
                    logger.warning(f"Could not find rating file {rating_file_path}")

    def _prepare_from_directory(
        self, input_directory: str, rating_type: Literal["float", "string"] = "float", x_only: bool = False
    ):
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
                        self._prepare_from_file(
                            file_path=file_path, rating=directory, rating_type=rating_type, x_only=x_only
                        )

    def _prepare_from_file(
        self, file_path: str, rating: str, rating_type: Literal["float", "string"] = "float", x_only: bool = False
    ):
        """
        :param file_path: Path to file to prepare
        :param rating: Rating to use for file
        """
        image_features = self._image_features(file_path)
        if image_features is None:
            return
        self.x.append(normalized(image_features))
        if not x_only:
            if rating_type == "float":
                try:
                    rating = float(rating)
                    logger.debug(f"Converted rating {rating} to float")
                except Exception:
                    logger.error(f"Could not parse rating {rating} as float")
                    return
            logger.info(f"Processing file: {file_path} with rating {rating}")
            self.y.append(self._y(rating))

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
