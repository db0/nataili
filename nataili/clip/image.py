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
from uuid import uuid4

import numpy as np
import torch

from nataili.cache import Cache
from nataili.util import autocast_cuda, logger


class ImageEmbed:
    def __init__(self, model, cache: Cache):
        """
        :param model: Loaded model from ModelManager
        :param cache: Cache object
        """
        self.model = model
        self.cache = cache

    def save_batch(self, batch):
        filenames = []
        for image_embed_array in batch["image_features"]:
            filename = str(uuid4())
            np.save(f"{self.cache.cache_dir}/{filename}", image_embed_array.float().cpu().detach().numpy())
            filenames.append(filename)
        for image_hash, filename in zip(batch["image_hashes"], filenames):
            self.cache.kv[image_hash] = filename

    @autocast_cuda
    def __call__(self, pil_image):
        """
        :param pil_image: PIL image to embed
        SHA256 hash of image is used as key in cache
        If image is not in cache, embed it and save it to cache
        Returns SHA256 hash of image
        """
        image_hash = hashlib.sha256(pil_image.tobytes()).hexdigest()
        if image_hash in self.cache.kv:
            logger.debug(f"Image {image_hash} already in cache")
            return image_hash
        logger.debug(f"Embedding image {image_hash}")
        with torch.no_grad():
            preprocess_image = self.model["preprocess"](pil_image).unsqueeze(0).to(self.model["device"])
        image_features = self.model["model"].encode_image(preprocess_image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embed_array = image_features.cpu().detach().numpy()
        filename = str(uuid4())
        np.save(f"{self.cache.cache_dir}/{filename}", image_embed_array)
        self.cache.kv[image_hash] = filename
        return image_hash

    @autocast_cuda
    def batch(self, pil_images, batch_size: int = 32):
        """
        :param pil_images: List of PIL images to embed
        SHA256 hash of image is used as key in cache
        If image is not in cache, embed it and save it to cache
        Returns SHA256 hash of image
        """
        image_hashes = []
        logger.debug("Generating hashes for images")
        for pil_image in pil_images:
            image_hashes.append(hashlib.sha256(pil_image["pil_image"].tobytes()).hexdigest())
        cached = True
        logger.debug("Checking if images are in cache")
        for image_hash in image_hashes:
            if image_hash not in self.cache.kv:
                cached = False
                break
        if cached:
            logger.debug(f"Images {image_hashes} already in cache")
            return image_hashes
        logger.debug("Embedding images")
        batches = []
        if len(pil_images) > batch_size:
            for i in range(0, len(pil_images), batch_size):
                batches.append(pil_images[i : i + batch_size])
            batches.append(pil_images[i + batch_size :])
        else:
            batches.append(pil_images)
        for batch in batches:
            with torch.no_grad():
                preprocess_images = []
                for pil_image in batch:
                    preprocess_images.append(
                        self.model["preprocess"](pil_image["pil_image"]).unsqueeze(0).to(self.model["device"])
                    )
                preprocess_images = torch.cat(preprocess_images, dim=0)
                image_features = self.model["model"].encode_image(preprocess_images)
                self.save_batch({"image_hashes": image_hashes, "image_features": image_features})
        return image_hashes
