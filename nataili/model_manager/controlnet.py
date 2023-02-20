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
from pathlib import Path

import safetensors.torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from nataili.cache import get_cache_directory
from nataili.model_manager.base import BaseModelManager
from nataili.util.logger import logger


class ControlNetModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/controlnet"
        self.models_db_name = "controlnet"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = (
            f"https://raw.githubusercontent.com/Sygil-Dev/nataili-model-reference/main/{self.models_db_name}.json"
        )
        self.control_nets = {}
        self.init()

    def load_control_ldm(
        self,
        model_name,
        target_name,
        target_state_dict,
    ):
        if model_name not in self.control_nets:
            logger.error(f"{model_name} not loaded")
            return False
        config_path = self.get_model_files(model_name)[1]["path"]
        config_path = f"{self.pkg}/{config_path}"
        logger.info(f"Loading controlLDM {model_name} for {target_name}")
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model).cpu()
        full_name = f"{model_name}_{target_name}"
        logger.info(f"Loaded {full_name} ControlLDM")
        scratch_dict = self.control_nets[model_name]["state_dict"]
        modified = 0
        new = 0
        copied = 0
        new_dict = {}
        for k in target_state_dict.keys():
            if k not in new_dict.keys():
                new_dict[k] = target_state_dict[k].clone()
                copied += 1
        logger.info(f"Copied {copied} parameters")
        logger.debug("Merging control net state dict into target state dict")
        for k in scratch_dict.keys():
            copy_key = f"model.diffusion_model.{k}"
            control_key = f"control_model.{k}"
            if copy_key in target_state_dict.keys():
                new_dict[copy_key] = scratch_dict[k].clone() + target_state_dict[copy_key].clone()
                new_dict[control_key] = scratch_dict[k].clone() + target_state_dict[copy_key].clone()
                modified += 1
            else:
                new_dict[control_key] = scratch_dict[k].clone()
                new += 1
        logger.info(f"Modified {modified} parameters, added {new} parameters")
        logger.info(f"Loaded {full_name} ControlLDM")
        model.load_state_dict(new_dict, strict=True)
        self.loaded_models[full_name] = {
            "model": model,
        }

    def load_controlnet(
        self,
        model_name,
    ):
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            logger.init_ok(f"Downloading {model_name}", status="Downloading")
            self.download_model(model_name)
            logger.init_ok(f"{model_name} downloaded", status="Downloading")
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        logger.info(f"Loading controlnet {model_name}")
        logger.info(f"Model path: {model_path}")
        state_dict = safetensors.torch.load_file(model_path)

        self.control_nets[model_name] = {"state_dict": state_dict}
        return True
