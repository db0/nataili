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
import hashlib
import os
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from nataili.cache import Cache
from nataili.clip.image import ImageEmbed
from nataili.clip.text import TextEmbed
from nataili.util.logger import logger


class Interrogator:
    def __init__(self, model):
        """
        :param model: Loaded model from ModelManager
        For each data list in model["data_lists"], check if all items are cached.
        If not, embed all items and save to cache.
        If yes, load all text embeds from cache.
        """
        self.model = model
        self.cache = Cache(self.model["cache_name"], cache_parentname="embeds", cache_subname="text")
        self.cache_image = Cache(self.model["cache_name"], cache_parentname="embeds", cache_subname="image")
        self.embed_lists = {}

    def load(self, key: str, text_array: List[str], individual: bool = True, device: str = "cuda"):
        """
        :param key: Key to use for embed_lists
        :param text_array: List of text to embed
        :param individual: Whether to store each text embed individually or as a concatenated tensor
        If individual is True, self.embed_lists[key] will be a dict with text as key and embed as value.
        If individual is False, self.embed_lists[key] will be a tensor of all text embeds concatenated.
        """
        logger.debug(f"key: {key}, text_array: {text_array}, individual: {individual}")
        cached = True
        for text in text_array:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if self.cache.get(file_hash=text_hash) is None:
                cached = False
                logger.debug(f"{text} not cached")
                break
        if not cached:
            logger.debug(f"Embedding {key}...")
            text_embed = TextEmbed(self.model, self.cache)
            for text in text_array:
                logger.debug(f"Embedding {text}")
                text_embed(text)
        else:
            logger.debug(f"{key} embeds already cached")
        logger.debug(f"Loading {key} embeds")
        if individual:
            self.embed_lists[key] = {}
            for text in text_array:
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                filename = f"{self.cache.cache_dir}/{text_hash}.npy"
                logger.debug(f"text: {text}, text_hash: {text_hash}, filename: {filename}")
                embed = torch.from_numpy(np.load(filename)).float().to(device)
                if len(embed.shape) == 1:
                    embed = embed.view(1, -1)
                self.embed_lists[key][text] = embed
        else:
            with torch.no_grad():
                text_features = []
                for text in text_array:
                    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    filename = f"{self.cache.cache_dir}/{text_hash}.npy"
                    logger.debug(f"text: {text}, text_hash: {text_hash}, filename: {filename}")
                    embed = torch.from_numpy(np.load(filename)).float().to(device)
                    if len(embed.shape) == 1:
                        embed = embed.view(1, -1)
                    text_features.append(embed)
                text_features = torch.cat(text_features, dim=0)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.embed_lists[key] = text_features

    def _text_similarity(self, text_features, text_features_2):
        """
        This is an internal function that calculates the similarity between two texts.
        :param text_features: Text features to compare
        :param text_features_2: Text features to compare to
        :return: Similarity between text and text
        """
        return text_features @ text_features_2.T

    def text_similarity(self, text_array, text_array_2, key, key_2, device):
        """
        For each text in text_array, calculate the similarity between the text and each text in text_array_2.
        :param text_array: List of text to compare
        :param text_array_2: List of text to compare to
        :param key: Key to use for embed_lists
        :param key_2: Key to use for embed_lists
        :param device: Device to run on
        :return: dict of {text: {text: similarity}}
        """
        if key not in self.embed_lists:
            self.load(key, text_array, individual=True, device=device)
        if key_2 not in self.embed_lists:
            self.load(key_2, text_array_2, individual=True, device=device)
        similarity = {}
        for text in text_array:
            text_features = self.embed_lists[key][text].to(device)
            similarity[text] = {}
            for text_2 in text_array_2:
                text_features_2 = self.embed_lists[key_2][text_2].to(device)
                similarity[text][text_2] = round(self._text_similarity(text_features, text_features_2)[0][0].item(), 4)
        return similarity

    def _similarity(self, image_features, text_features):
        """
        This is an internal function that calculates the similarity between a single image and a single text.
        :param image_features: Image features to compare to text features
        :param text_features: Text features to compare to image features
        :return: Similarity between text and image
        """
        return text_features @ image_features.T

    def similarity(self, image_features, text_array, key, device):
        """
        For each text in text_array, calculate the similarity between the image and the text.
        :param image_features: Image features to compare to text features
        :param text_array: List of text to compare to image
        :param key: Key to use for embed_lists
        :param device: Device to run on
        :return: dict of {text: similarity}
        """
        if key not in self.embed_lists:
            logger.debug(f"Loading {key} embeds")
            self.load(key, text_array, individual=True, device=device)
        logger.debug(f"{len(text_array)} text_array: {text_array}")
        similarity = {}
        for text in text_array:
            text_features = self.embed_lists[key][text].to(device)
            similarity[text] = round(self._similarity(image_features, text_features)[0][0].item(), 4)
        return {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}

    def rank(self, image_features, text_array, key, device, top_count=2):
        """
        Ranks the text_array by similarity to the image.
        The top results are the most similar to the image out of the text_array.
        The bottom results are the least similar to the image out of the text_array.
        The results are relative to each other, not absolute.
        :param image_features: Image features to compare to text features
        :param text_array: List of text to compare to image.
            Text will be concatenated to a single tensor and loaded from cache if it is not already loaded.
        :param key: Key to use for embed_lists.
        :param device: Device to run on
        :param top_count: Number of top results to return
        :return: List of tuples of (text, similarity)
        """
        top_count = min(top_count, len(text_array))
        if key not in self.embed_lists:
            self.load(key, text_array, individual=False, device=device)
        text_features = self.embed_lists[key].to(device)

        similarity = torch.zeros((1, len(text_array))).to(device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        top = [
            {"text": text_array[top_labels[0][i].numpy()], "confidence": (top_probs[0][i].numpy() * 100)}
            for i in range(top_count)
        ]
        return top

    def __call__(
        self,
        image: Image.Image = None,
        filename: str = None,
        directory: str = None,
        text_array: Union[List[str], Dict[str, List[str]], None] = None,
        similarity=False,
        rank=False,
        top_count=2,
    ):
        """
        :param input_image: PIL image
        :param text_array: List of text to compare to image, or dict of lists of text to compare to image
        :param top_count: Number of top results to return
        :return:
            * If similarity is True, returns dict of {text: similarity}
            * If rank is True, returns list of tuples of (text, similarity)
            See rank() for more details
            See similarity() for more details
        If text_array is None, uses default text_array from model["data_lists"]
        """
        if image is None and filename is None:
            logger.error("Either image or filename must be set")
            return
        if image is not None and filename is not None:
            logger.error("Only one of image or filename must be set")
            return
        if os.path.isdir(f"{directory}/{filename}"):
            logger.error("Filename must be a file, not a directory")
            return
        if not similarity and not rank:
            logger.error("Must specify similarity or rank")
            return
        if text_array is None:
            text_array = self.model["data_lists"]
        if isinstance(text_array, list):
            text_array = {"default": text_array}
        elif isinstance(text_array, dict):
            pass
        image_embed = ImageEmbed(self.model, self.cache_image)
        image_hash = image_embed(image, filename, directory)
        image_embed_array = np.load(f"{self.cache_image.cache_dir}/{image_hash}.npy")
        image_features = torch.from_numpy(image_embed_array).float().to(self.model["device"])
        if similarity and not rank:
            results = {}
            for k in text_array.keys():
                results[k] = self.similarity(image_features, text_array[k], k, self.model["device"])
                logger.debug(f"{k}: {results[k]}")
            return results
        elif rank and not similarity:
            results = {}
            for k in text_array.keys():
                results[k] = self.rank(image_features, text_array[k], k, self.model["device"], top_count)
                logger.debug(f"{k}: {results[k]}")
            return results
        else:
            similarity = {}
            for k in text_array.keys():
                similarity[k] = self.similarity(image_features, text_array[k], k, self.model["device"])
                logger.debug(f"{k}: {similarity[k]}")
            rank = {}
            for k in text_array.keys():
                rank[k] = self.rank(image_features, text_array[k], k, self.model["device"], top_count)
                logger.debug(f"{k}: {rank[k]}")
            return {
                "similarity": similarity,
                "rank": rank,
            }
