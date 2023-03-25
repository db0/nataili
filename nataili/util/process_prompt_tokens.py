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

import requests
from tqdm import tqdm
import open_clip.tokenizer

from nataili import disable_download_progress
from nataili.cache import get_cache_directory
from nataili.model_manager.base import BaseModelManager
from nataili.train.lora.lora import LoRA
from nataili.util.load_learned_embed_in_clip import load_learned_embed_in_clip, load_learned_embed_in_clip_v2
from nataili.util.logger import logger
from nataili.util.lora import load_lora_for_models

class EmbedsManager:
    def __init__(self,
        remote_db = "https://raw.githubusercontent.com/ResidentChief/AI-Horde-image-model-reference/Updates0603/db_embeds.json",
        path = None,
        all_embeds = [],
    ):
        self.remote_db = remote_db
        self.path = "nataili/concepts-library"
        self.all_embeds = self.download_remote_db()


    def download_remote_db(self):
        r = requests.get(self.remote_db)
        all_embeds = r.json()
        return all_embeds

    def find_embedding(self, name):
        if name in self.all_embeds:
            return self.all_embeds[name]
        else:
            return None
        
    def get_embedding_info(self, name):
        EmbedType = self.all_embeds[name]["EmbedType"]
        baseline = self.all_embeds[name]["baseline"]
        return EmbedType, baseline

    def download_embedding(self, name, full_path):
        download_location = self.all_embeds[name]["DownloadPath"]
        print(f"Downloading embedding {name} from {download_location} to {full_path}")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        pbar_desc = full_path.split("/")[-1]
        r = requests.get(download_location, stream=True, allow_redirects=True)
        with open(full_path, "wb") as f:
            with tqdm(
                # all optional kwargs
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=pbar_desc,
                total=int(r.headers.get("content-length", 0)),
                disable=disable_download_progress.active,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def load_embedding(self, name):
        print(f"Loading embed for {name}")
        file_name = self.all_embeds[name]["DownloadPath"].split('/')[-1]
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            full_path = f"{self.path}/{file_name}"
            self.download_embedding(name, full_path)
            return self.load_embedding(name)
        else:
            for files in os.listdir(self.path):
                if files.startswith(name):
                    return os.path.basename(files)
                else:
                    full_path = f"{self.path}/{file_name}"
                    self.download_embedding(name, full_path)
                    return os.path.basename(full_path)


def process_prompt_tokens(prompt_tokens, model, model_baseline):
    embed_manager = EmbedsManager()
    new_tokens = None
    
    for token_name in prompt_tokens:
        print(f"Token for processing = {token_name}")
        print(f"Model baseline = {model_baseline}")
        text_encoder = (
            model.cond_stage_model.transformer
            if model_baseline == "stable diffusion 1"
            else model.cond_stage_model.model.transformer
        )
        tokenizer = (
            model.cond_stage_model.tokenizer
            if model_baseline == "stable diffusion 1"
            else open_clip.tokenizer._tokenizer
        )
        print (f"Tokenizer = {tokenizer}")

        embed_data = embed_manager.find_embedding(token_name)
        if embed_data is not None:
            embedding_path = embed_manager.load_embedding(token_name)
            embedding_type, embedding_baseline = embed_manager.get_embedding_info(token_name)

            print(f"Embedding path = {embedding_path}; Embedding type = {embedding_type}; Embedding baseline = {embedding_baseline}")

            if embedding_type == "Textual Inversion":
                if model_baseline == "stable diffusion 1":
                    new_tokens = load_learned_embed_in_clip(
                        f"{os.path.join(embed_manager.path, embedding_path)}",
                        text_encoder,
                        tokenizer,
                        token_name
                    )
                else:
                    new_tokens = load_learned_embed_in_clip_v2(
                        f"{os.path.join(embed_manager.path, embedding_path)}",
                        model,
                        text_encoder,
                        tokenizer,
                        token_name
                    )
            elif embedding_type == "LoRA":
                model, tokenizer = load_lora_for_models(
                    model, 
                    model.cond_stage_model, 
                    f"{os.path.join(embed_manager.path, embedding_path)}",
                    1,
                    1
                )
            else:
                logger.info(f"Embedding for {token_name} is for {embedding_baseline}; Model loaded is based on {model_baseline}")
        else:
            logger.info(f"No embedding for {token_name} found")
    
    del embed_manager
    return new_tokens
