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
from concurrent.futures import ThreadPoolExecutor

import clip
import numpy as np
import torch

from nataili.cache import Cache
from nataili.util.cast import autocast_cuda
from nataili.util.logger import logger


class TextEmbed:
    def __init__(self, model, cache: Cache):
        """
        :param model: Loaded model from ModelManager
        :param cache: Cache object
        """
        self.model = model
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=1024, thread_name_prefix="SaveThread")

    @autocast_cuda
    def _batch(self, text_list: list):
        for text in text_list:
            if isinstance(text["prompt"], bytes):
                text["prompt"] = text["prompt"].decode("utf-8")
            text["hash"] = hashlib.sha256(text["prompt"]).hexdigest()
        text_tokens = [clip.tokenize(text["prompt"], truncate=True).to(self.model["device"]) for text in text_list]
        text_tokens = torch.cat(text_tokens, dim=0)
        with torch.no_grad():
            text_features = self.model["model"].encode_text(text_tokens).float()
        for text_embed_array, text in zip(text_features, text_list):
            future = self.executor.submit(self._save, text_embed_array, text["hash"])
            self.cache.add_sqlite_row(text["filename"], text["hash"], text["hash"])

    def _save(self, text_embed_array, text_hash):
        text_embed_array /= text_embed_array.norm(dim=-1, keepdim=True)
        np.save(f"{self.cache.cache_dir}/{text_hash}", text_embed_array.float().cpu().detach().numpy())

    @autocast_cuda
    def __call__(self, text: str, check_cache: bool = False, filename: str = None):
        """
        :param text: Text to embed
        :param check_cache: Check cache for text
        :param filename: Filename to save to cache
        If text is not in cache, embed it and save it to cache
        """
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if check_cache:
            cached = self.cache.get(file_hash=text_hash)
            if cached:
                return cached
        text_tokens = clip.tokenize([text], truncate=True).to(self.model["device"])
        with torch.no_grad():
            text_features = self.model["model"].encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embed_array = text_features.cpu().detach().numpy()
        if filename:
            np.save(f"{self.cache.cache_dir}/{text_hash}", text_embed_array)
            self.cache.add_sqlite_row(filename, text_hash, text_hash)
        else:
            np.save(f"{self.cache.cache_dir}/{text_hash}", text_embed_array)
            self.cache.add_sqlite_row(text, text_hash, text_hash)
