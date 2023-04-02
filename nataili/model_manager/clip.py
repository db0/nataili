"""
This file is part of nataili ("Homepage" = "https://github.com/db0/nataili").

Copyright 2022-2023 hlky. Copyright 2023 hlky and AI Horde Community
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
import time
from pathlib import Path

import clip
import open_clip
import torch

from nataili.cache import get_cache_directory
from nataili.model_manager.base import BaseModelManager
from nataili.util.load_list import load_list
from nataili.util.logger import logger


class ClipModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/clip"
        self.models_db_name = "clip"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = (
            f"https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/{self.models_db_name}.json"
        )
        self.init(list_models=True)

    def load_data_lists(self):
        data_lists = {}
        data_lists["artist"] = load_list(self.pkg / "artists.txt")
        data_lists["flavors"] = load_list(self.pkg / "flavors.txt")
        data_lists["medium"] = load_list(self.pkg / "mediums.txt")
        data_lists["movement"] = load_list(self.pkg / "movements.txt")
        data_lists["trending"] = load_list(self.pkg / "sites.txt")
        data_lists["techniques"] = load_list(self.pkg / "techniques.txt")
        data_lists["tags"] = load_list(self.pkg / "tags.txt")
        return data_lists

    def load_coca(self, model_name, half_precision=True, gpu_id=0, cpu_only=False):
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        model, _, transform = open_clip.create_model_and_transforms(
            "coca_ViT-L-14",
            pretrained=model_path,
            device=device,
            precision="fp16" if half_precision else "fp32",
        )
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        return {
            "model": model,
            "device": device,
            "transform": transform,
            "half_precision": half_precision,
            "cache_name": model_name.replace("/", "_"),
        }

    def load_open_clip(self, model_name, half_precision=True, gpu_id=0, cpu_only=False):
        pretrained = self.get_model(model_name)["pretrained_name"]
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=self.path,
            device=device,
            precision="fp16" if half_precision else "fp32",
        )
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        data_lists = self.load_data_lists()
        return {
            "model": model,
            "device": device,
            "preprocess": preprocess,
            "data_lists": data_lists,
            "half_precision": half_precision,
            "cache_name": model_name.replace("/", "_"),
        }

    def load_clip(self, model_name, half_precision=True, gpu_id=0, cpu_only=False):
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        model, preprocess = clip.load(model_name, device=device, download_root=self.path)
        model = model.eval()
        if half_precision:
            model = model.half()
        data_lists = self.load_data_lists()
        return {
            "model": model,
            "device": device,
            "preprocess": preprocess,
            "data_lists": data_lists,
            "half_precision": half_precision,
            "cache_name": model_name.replace("/", "_"),
        }

    def load(self, model_name: str, half_precision=True, gpu_id=0, cpu_only=False):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        """
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            logger.init_ok(f"Downloading {model_name}", status="Downloading")
            self.download_model(model_name)
            logger.init_ok(f"{model_name} downloaded", status="Downloading")
        if model_name not in self.loaded_models:
            if not self.cuda_available:
                cpu_only = True
            tic = time.time()
            logger.init(f"{model_name}", status="Loading")
            if self.models[model_name]["type"] == "open_clip":
                self.loaded_models[model_name] = self.load_open_clip(model_name, half_precision, gpu_id, cpu_only)
            elif self.models[model_name]["type"] == "clip":
                self.loaded_models[model_name] = self.load_clip(model_name, half_precision, gpu_id, cpu_only)
            elif self.models[model_name]["type"] == "coca":
                self.loaded_models[model_name] = self.load_coca(model_name, half_precision, gpu_id, cpu_only)
            else:
                logger.error(f"Unknown model type: {self.models[model_name]['type']}")
                return
            logger.init_ok(f"Loading {model_name}", status="Success")
            toc = time.time()
            logger.init_ok(f"Loading {model_name}: Took {toc-tic} seconds", status="Success")
            return True
