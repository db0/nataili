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

# threading
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from nataili.cache import Cache
from nataili.util.cast import autocast_cuda
from nataili.util.logger import logger


class ImageEmbed:
    def __init__(self, model, cache: Cache):
        """
        :param model: Loaded model from ModelManager
        :param cache: Cache object
        """
        self.model = model
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=1024, thread_name_prefix="SaveThread")

    @autocast_cuda
    def _batch(self, pil_images):
        for pil_image in pil_images:
            pil_images[pil_image]["hash"] = hashlib.sha256(pil_images[pil_image]["pil_image"].tobytes()).hexdigest()
        preprocess_images = []
        with torch.no_grad():
            for pil_image in pil_images:
                preprocess_images.append(
                    self.model["preprocess"](pil_images[pil_image]["pil_image"]).unsqueeze(0).to(self.model["device"])
                )
            preprocess_images = torch.cat(preprocess_images, dim=0)
            image_features = self.model["model"].encode_image(preprocess_images)
            for image_embed_array, pil_image in zip(image_features, pil_images):
                future = self.executor.submit(self._save, image_embed_array, pil_images[pil_image])

    def _save(self, image_embed_array, pil_image):
        np.save(f"{self.cache.cache_dir}/{pil_image['hash']}", image_embed_array.float().cpu().detach().numpy())
        self.cache.add_sqlite_row(
            file=pil_image["filename"].replace(".webp", ""), pil_hash=pil_image["hash"], hash=None
        )

    @autocast_cuda
    def __call__(self, filename: str, directory: str = None, skip_cache: bool = False):
        """
        :param pil_image: PIL image to embed
        SHA256 hash of image is used as key in cache
        If image is not in cache, embed it and save it to cache
        Returns SHA256 hash of image
        """
        pil_image = Image.open(f"{directory}/{filename}").convert("RGB")
        file_hash = hashlib.sha256(open(f"{directory}/{filename}", "rb").read()).hexdigest()
        image_hash = hashlib.sha256(pil_image.tobytes()).hexdigest()
        if not skip_cache:
            cached = self.cache.get(pil_hash=image_hash)
            if cached:
                logger.debug(f"Image {image_hash} already in cache")
                return image_hash
        else:
            logger.debug(f"Skipping cache for image {image_hash}")
        logger.debug(f"Embedding image {image_hash}")
        with torch.no_grad():
            preprocess_image = self.model["preprocess"](pil_image).unsqueeze(0).to(self.model["device"])
        image_features = self.model["model"].encode_image(preprocess_image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embed_array = image_features.cpu().detach().numpy()
        np.save(f"{self.cache.cache_dir}/{image_hash}", image_embed_array)
        self.cache.add_sqlite_row(file=filename, hash=file_hash, pil_hash=image_hash)
