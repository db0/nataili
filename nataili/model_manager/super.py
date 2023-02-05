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
import torch

from nataili.model_manager.aitemplate import AITemplateModelManager
from nataili.model_manager.blip import BlipModelManager
from nataili.model_manager.clip import ClipModelManager
from nataili.model_manager.codeformer import CodeFormerModelManager
from nataili.model_manager.compvis import CompVisModelManager
from nataili.model_manager.esrgan import EsrganModelManager
from nataili.model_manager.gfpgan import GfpganModelManager
from nataili.util.logger import logger


class ModelManager:
    def __init__(
        self,
        aitemplate: AITemplateModelManager = AITemplateModelManager(),
        blip: BlipModelManager = BlipModelManager(),
        clip: ClipModelManager = ClipModelManager(),
        compvis: CompVisModelManager = CompVisModelManager(),
        esrgan: EsrganModelManager = EsrganModelManager(),
        gfpgan: GfpganModelManager = GfpganModelManager(),
        codeformer: bool = False,
    ):
        self.aitemplate = aitemplate
        self.blip = blip
        self.clip = clip
        self.compvis = compvis
        self.esrgan = esrgan
        self.gfpgan = gfpgan
        if codeformer and self.esrgan is not None and self.gfpgan is not None:
            self.codeformer = CodeFormerModelManager(gfpgan=self.gfpgan, esrgan=self.esrgan)
        self.cuda_available = torch.cuda.is_available()

    def load(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        voodoo=False,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        voodoo: bool. (compvis only) Voodoo ray.
        """
        if not self.cuda_available:
            cpu_only = True
        if model_name in self.aitemplate.models:
            return self.aitemplate.load(model_name, gpu_id)
        if model_name in self.blip.models:
            return self.blip.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
        if model_name in self.clip.models:
            return self.clip.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
        if model_name in self.compvis.models:
            return self.compvis.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only, voodoo=voodoo
            )
        if model_name in self.esrgan.models:
            return self.esrgan.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
        if model_name in self.gfpgan.models:
            return self.gfpgan.load(model_name=model_name, gpu_id=gpu_id, cpu_only=cpu_only)
        if model_name in self.codeformer.models:
            return self.codeformer.load(
                model_name=model_name, half_precision=half_precision, gpu_id=gpu_id, cpu_only=cpu_only
            )
        logger.error(f"{model_name} not found")
        return
