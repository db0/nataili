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
import re
from contextlib import nullcontext
from typing import Literal

import einops
import k_diffusion as K
import numpy as np
import skimage
import torch
from einops import rearrange
from PIL import Image, ImageOps
from slugify import slugify
from torch import nn
from transformers import CLIPFeatureExtractor

from annotator.canny import CannyDetector
from annotator.util import HWC3
from annotator.util import resize_image as control_resize_image
from cldm.cldm import ControlLDM
from ldm2.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.kdiffusion import CFGMaskedDenoiser, KDiffusionSampler
from ldm.models.diffusion.plms import PLMSSampler

# from nataili import disable_progress
from nataili.model_manager.controlnet import ControlNetModelManager
from nataili.stable_diffusion.annotation import (
    HED,
    Canny,
    Depth,
    FakeScribbles,
    Hough,
    Normal,
    Openpose,
    Scribble,
    Seg,
)
from nataili.stable_diffusion.prompt_weights import (
    fix_mismatched_tensors,
    get_learned_conditioning_with_prompt_weights,
)
from nataili.util.cache import torch_gc
from nataili.util.cast import autocast_cpu, autocast_cuda
from nataili.util.create_random_tensors import create_random_tensors
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.img2img import find_noise_for_image, get_matched_noise, process_init_mask, resize_image
from nataili.util.logger import logger
from nataili.util.process_prompt_tokens import process_prompt_tokens
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int

try:
    from nataili.util.voodoo import load_from_plasma
except ModuleNotFoundError as e:
    from nataili import disable_voodoo

    if not disable_voodoo.active:
        raise e


def offload_model(model, cpu=None, gpu=None):
    if cpu is None and gpu is None:
        raise ValueError("Must specify either cpu or gpu")
    if cpu and gpu:
        raise ValueError("Cannot offload model to both cpu and gpu")
    if cpu:
        model = model.cpu()
    if gpu:
        model = model.to(gpu)
    torch_gc()
    return model


def low_vram_mode():
    """
    enabled by default
    """
    return os.environ.get("LOW_VRAM_MODE", "1") == "1"


"""
has no effect if low_vram_mode is disabled or if disable_low_vram_mode is enabled
NOTE:
"cpu" and "cuda" are the supported device types
"cpu" could be replaced with "offload", the check would need to account for that
the check is "cuda" in device, so "cuda:0" would also be supported
other device types can be added as needed "mps" etc -
    * add a check for the device type
    * pass that device type in the list of models

list of models is a list of tuples (model, device)
example:
    models = [(model, "cpu"), (model.cond_stage_model, "cuda:0")]
"""


def low_vram(models: list):
    if not low_vram_mode():
        return
    torch_gc()
    for m in models:
        model, device = m
        if device == "cpu":
            model = offload_model(model, cpu=device)
        elif "cuda" in device:
            model = offload_model(model, gpu=device)
        else:
            raise ValueError("Unknown device")
    torch_gc()


class CompVis:
    def __init__(
        self,
        model,
        output_dir,
        model_name=None,
        model_baseline=None,
        save_extension="jpg",
        output_file_path=False,
        load_concepts=False,
        concepts_dir=None,
        verify_input=True,
        auto_cast=True,
        filter_nsfw=False,
        safety_checker=None,
        disable_voodoo=False,
    ):
        self.model = model
        self.model_name = model_name
        self.model_baseline = model_baseline
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
        self.auto_cast = auto_cast
        self.comments = []
        self.output_images = []
        self.info = ""
        self.stats = ""
        self.images = []
        self.filter_nsfw = filter_nsfw
        self.safety_checker = safety_checker
        self.feature_extractor = CLIPFeatureExtractor()
        self.disable_voodoo = disable_voodoo
        self.apply_control = None
        self.control_net_manager = ControlNetModelManager()
        self.control_net_model = None

    @autocast_cuda
    def generate(
        self,
        prompt: str,
        init_img=None,
        init_mask=None,
        mask_mode="mask",
        resize_mode="resize",
        noise_mode="seed",
        find_noise_steps=50,
        denoising_strength: float = 0.8,
        ddim_steps=50,
        sampler_name="k_lms",
        n_iter=1,
        batch_size=1,
        cfg_scale=7.5,
        seed=None,
        height=512,
        width=512,
        save_individual_images: bool = True,
        save_grid: bool = True,
        ddim_eta: float = 0.0,
        sigma_override: dict = None,
        tiling: bool = False,
        clip_skip=1,
        hires_fix: bool = False,
        control_type: Literal[
            "canny", "hed", "depth", "normal", "openpose", "seg", "scribble", "fakescribbles", "hough"
        ] = None,
    ):
        model_context = (
            load_from_plasma(self.model["model"], self.model["device"]) if not self.disable_voodoo else nullcontext()
        )
        with model_context as model:
            if self.disable_voodoo:
                model = self.model["model"]
            if control_type is not None and init_img is not None and "stable diffusion 2" not in self.model_baseline:
                sampler_name = "DDIM"
                if control_type == "canny":
                    control_name = "control_canny"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    canny = Canny()
                    control_result = canny(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del canny
                elif control_type == "hed":
                    control_name = "control_hed"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]

                    hed = HED()
                    control_result = hed(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del hed
                elif control_type == "depth":
                    control_name = "control_depth"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    depth = Depth()
                    control_result = depth(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del depth
                elif control_type == "scribble":
                    control_name = "control_scribble"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    scribble = Scribble()
                    control_result = scribble(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del scribble
                elif control_type == "fakescribbles":
                    control_name = "control_scribble"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    fake_scribbles = FakeScribbles()
                    control_result = fake_scribbles(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del fake_scribbles
                elif control_type == "hough":
                    control_name = "control_mlsd"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    hough = Hough()
                    control_result = hough(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del hough
                elif control_type == "openpose":
                    control_name = "control_openpose"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    openpose = Openpose()
                    control_result = openpose(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del openpose
                elif control_type == "seg":
                    control_name = "control_seg"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    seg = Seg()
                    control_result = seg(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del seg
                elif control_type == "normal":
                    control_name = "control_normal"
                    low_vram(
                        [(model, "cpu"), (model.cond_stage_model, "cpu"), (model.first_stage_model, "cpu")],
                    )
                    self.control_net_manager.load_controlnet(control_name)
                    self.control_net_manager.load_control_ldm(
                        control_name, self.model_name, model.state_dict(), _device=self.model["device"]
                    )
                    loaded_control_ldm = f"{control_name}_{self.model_name}"
                    self.model_name = loaded_control_ldm
                    self.control_net_model = self.control_net_manager.loaded_models[loaded_control_ldm]["model"]
                    normal = Normal()
                    control_result = normal(init_img)
                    control = control_result["control"]
                    H, W, C = control_result["shape"]
                    del normal
                else:
                    raise ValueError(f"Invalid control_type: {control_type}")
            elif init_img is not None:
                init_img = resize_image(resize_mode, init_img, width, height)
                hires_fix = False
            else:
                if hires_fix and width > 512 and height > 512:
                    logger.debug("HiRes Fix Requested")
                    final_width = width
                    final_height = height
                    if self.model_baseline == "stable diffusion 2":
                        first_pass_ratio = min(final_height / 768, final_width / 768)
                    else:
                        first_pass_ratio = min(final_height / 512, final_width / 512)
                    width = (int(final_width / first_pass_ratio) // 64) * 64
                    height = (int(final_height / first_pass_ratio) // 64) * 64
                    logger.debug(f"First pass image will be processed at width={width}; height={height}")
                else:
                    hires_fix = False
            if mask_mode == "mask":
                if init_mask:
                    init_mask = process_init_mask(init_mask)
            elif mask_mode == "invert":
                if init_mask:
                    init_mask = process_init_mask(init_mask)
                    init_mask = ImageOps.invert(init_mask)
            elif mask_mode == "alpha":
                init_img_transparency = init_img.split()[-1].convert(
                    "L"
                )  # .point(lambda x: 255 if x > 0 else 0, mode='1')
                init_mask = init_img_transparency
                init_mask = init_mask.convert("RGB")
                init_mask = resize_image(resize_mode, init_mask, width, height)
                init_mask = init_mask.convert("RGB")

            """
            vram
            control net doesn't need first stage model on gpu
            regular does, for img2img and hires_fix, if txt2img keep it on cpu
            """
            if control_type is not None:
                low_vram(
                    [
                        (self.control_net_model, "cpu"),
                        (self.control_net_model.control_model, "cpu"),
                        (self.control_net_model.cond_stage_model, "cuda"),
                        (self.control_net_model.first_stage_model, "cpu"),
                    ],
                )
            else:
                """
                k-diffusion needs the model to be on gpu to create the model wrap
                so we move it to gpu here
                """
                low_vram(
                    [
                        (model, "cuda"),
                        (model.cond_stage_model, "cuda"),
                        (model.first_stage_model, "cuda"),
                    ],
                )

            assert 0.0 <= denoising_strength <= 1.0, "can only work with strength in [0.0, 1.0]"
            t_enc = int(denoising_strength * ddim_steps)

            if (
                init_mask is not None
                and (noise_mode == "matched" or noise_mode == "find_and_matched")
                and init_img is not None
            ):
                noise_q = 0.99
                color_variation = 0.0
                mask_blend_factor = 1.0

                np_init = (np.asarray(init_img.convert("RGB")) / 255.0).astype(
                    np.float64
                )  # annoyingly complex mask fixing
                np_mask_rgb = 1.0 - (np.asarray(ImageOps.invert(init_mask).convert("RGB")) / 255.0).astype(np.float64)
                np_mask_rgb -= np.min(np_mask_rgb)
                np_mask_rgb /= np.max(np_mask_rgb)
                np_mask_rgb = 1.0 - np_mask_rgb
                np_mask_rgb_hardened = 1.0 - (np_mask_rgb < 0.99).astype(np.float64)
                blurred = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16.0, channel_axis=2, truncate=32.0)
                blurred2 = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16.0, channel_axis=2, truncate=32.0)
                # np_mask_rgb_dilated = np_mask_rgb + blurred  # fixup mask todo: derive magic constants
                # np_mask_rgb = np_mask_rgb + blurred
                np_mask_rgb_dilated = np.clip((np_mask_rgb + blurred2) * 0.7071, 0.0, 1.0)
                np_mask_rgb = np.clip((np_mask_rgb + blurred) * 0.7071, 0.0, 1.0)

                noise_rgb = get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
                blend_mask_rgb = np.clip(np_mask_rgb_dilated, 0.0, 1.0) ** (mask_blend_factor)
                noised = noise_rgb[:]
                blend_mask_rgb **= 2.0
                noised = np_init[:] * (1.0 - blend_mask_rgb) + noised * blend_mask_rgb

                np_mask_grey = np.sum(np_mask_rgb, axis=2) / 3.0
                ref_mask = np_mask_grey < 1e-3

                all_mask = np.ones((height, width), dtype=bool)
                noised[all_mask, :] = skimage.exposure.match_histograms(
                    noised[all_mask, :] ** 1.0, noised[ref_mask, :], channel_axis=1
                )

                init_img = Image.fromarray(np.clip(noised * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")

            def init(model, init_img):
                image = init_img.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = image[None].transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)

                mask_channel = None
                if init_mask:
                    alpha = resize_image(resize_mode, init_mask, width // 8, height // 8)
                    mask_channel = alpha.split()[-1]

                mask = None
                if mask_channel is not None:
                    mask = np.array(mask_channel).astype(np.float32) / 255.0
                    mask = 1 - mask
                    mask = np.tile(mask, (4, 1, 1))
                    mask = mask[None].transpose(0, 1, 2, 3)
                    mask = torch.from_numpy(mask).to(model.first_stage_model.device)

                init_image = 2.0 * image - 1.0
                init_image = init_image.to(model.first_stage_model.device)
                init_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(init_image)
                )  # move to latent space

                return (
                    init_latent,
                    mask,
                )

            def sample_img2img(
                init_data,
                ddim_steps,
                x,
                conditioning,
                unconditional_conditioning,
                sampler_name,
                batch_size=1,
                shape=None,
                karras=False,
                sigma_override: dict = None,
            ):
                nonlocal sampler
                t_enc_steps = t_enc
                if hires_fix:
                    ddim_steps = 20
                    t_enc_steps = int(0.2 * ddim_steps)
                else:
                    t_enc_steps = t_enc
                obliterate = False
                if ddim_steps == t_enc_steps:
                    t_enc_steps = t_enc_steps - 1
                    obliterate = True

                if sampler_name != "DDIM":
                    samples_ddim, _ = sampler.sample_img2img(
                        init_data=init_data,
                        S=ddim_steps,
                        t_enc=t_enc_steps,
                        obliterate=obliterate,
                        conditioning=conditioning,
                        unconditional_conditioning=unconditional_conditioning,
                        unconditional_guidance_scale=cfg_scale,
                        x_T=x,
                    )
                else:
                    x0, z_mask = init_data

                    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
                    z_enc = sampler.stochastic_encode(
                        x0,
                        torch.tensor([t_enc_steps] * batch_size).to(self.model["device"]),
                    )

                    # Obliterate masked image
                    if z_mask is not None and obliterate:
                        random = torch.randn(z_mask.shape, device=z_enc.device)
                        z_enc = (z_mask * random) + ((1 - z_mask) * z_enc)

                        # decode it
                    samples_ddim = sampler.decode(
                        z_enc,
                        conditioning,
                        t_enc_steps,
                        unconditional_guidance_scale=cfg_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        z_mask=z_mask,
                        x0=x0,
                    )
                return samples_ddim

            def sample(
                init_data,
                x,
                conditioning,
                unconditional_conditioning,
                sampler_name,
                batch_size=1,
                shape=None,
                karras=False,
                sigma_override: dict = None,
            ):
                if sampler_name == "dpmsolver":
                    samples_ddim, _ = sampler.sample(
                        ddim_steps,
                        batch_size,
                        shape,
                        conditioning=conditioning,
                        unconditional_guidance_scale=cfg_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        x_T=x,
                        karras=karras,
                        sigma_override=sigma_override,
                    )
                else:
                    samples_ddim, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=conditioning,
                        unconditional_guidance_scale=cfg_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        x_T=x,
                        karras=karras,
                        sigma_override=sigma_override,
                    )
                return samples_ddim

            def create_sampler_by_sampler_name(model):
                nonlocal sampler_name
                if self.model_baseline == "stable diffusion 2":
                    v = True
                    # model_baseline = "stable diffusion 2" covers sd2.x 768 models
                else:
                    # 1.x models and 2.x 512 models do not use v-prediction
                    v = False
                if (
                    sampler_name == "PLMS" and "stable diffusion 2" not in self.model_baseline
                ):  # TODO: check support for sd2.x
                    sampler = PLMSSampler(model)
                elif (
                    sampler_name == "DDIM" and "stable diffusion 2" not in self.model_baseline
                ):  # TODO: check support for sd2.x
                    sampler = DDIMSampler(model)
                elif sampler_name == "k_dpm_2_a":
                    sampler = KDiffusionSampler(model, "dpm_2_ancestral", v=v)
                elif sampler_name == "k_dpm_2":
                    sampler = KDiffusionSampler(model, "dpm_2", v=v)
                elif sampler_name == "k_euler_a":
                    sampler = KDiffusionSampler(model, "euler_ancestral", v=v)
                elif sampler_name == "k_euler":
                    sampler = KDiffusionSampler(model, "euler", v=v)
                elif sampler_name == "k_heun":
                    sampler = KDiffusionSampler(model, "heun", v=v)
                elif sampler_name == "k_lms":
                    sampler = KDiffusionSampler(model, "lms", v=v)
                elif sampler_name == "k_dpm_fast":
                    sampler = KDiffusionSampler(model, "dpm_fast", v=v)
                elif sampler_name == "k_dpm_adaptive":
                    sampler = KDiffusionSampler(model, "dpm_adaptive", v=v)
                elif sampler_name == "k_dpmpp_2s_a":
                    sampler = KDiffusionSampler(model, "dpmpp_2s_ancestral", v=v)
                elif sampler_name == "k_dpmpp_2m":
                    sampler = KDiffusionSampler(model, "dpmpp_2m", v=v)
                elif sampler_name == "k_dpmpp_sde":
                    sampler = KDiffusionSampler(model, "dpmpp_sde", v=v)
                elif sampler_name == "dpmsolver" and "stable diffusion 2" in self.model_baseline:  # only for sd2.x
                    sampler = DPMSolverSampler(model)
                else:
                    logger.error(f"Sampler '{sampler_name}' unknown or does not match model.")
                    # Ensure we don't hit an UnboundLocalError because sampler is not set
                    raise Exception(f"Sampler name not found of does not match model baseline {self.model_baseline}")
                return sampler

            seed = seed_to_int(seed)

            image_dict = {"seed": seed}
            negprompt = ""
            if "###" in prompt:
                prompt, negprompt = prompt.split("###", 1)
                prompt = prompt.strip()
                negprompt = negprompt.strip()

            os.makedirs(self.output_dir, exist_ok=True)

            sample_path = os.path.join(self.output_dir, "samples")
            os.makedirs(sample_path, exist_ok=True)

            karras = False
            if "karras" in sampler_name:
                karras = True
                sampler_name = sampler_name.replace("_karras", "")

            if control_type is None:
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                        m.padding_mode = "circular" if tiling else m._orig_padding_mode
            sampler = create_sampler_by_sampler_name(model if control_type is None else self.control_net_model)
            if self.load_concepts and self.concepts_dir is not None:
                prompt_tokens = re.findall("<([a-zA-Z0-9-]+)>", prompt)
                if prompt_tokens:
                    process_prompt_tokens(
                        prompt_tokens, model if control_type is None else self.control_net_model, self.concepts_dir
                    )

            all_prompts = batch_size * n_iter * [prompt]
            all_seeds = [seed + x for x in range(len(all_prompts))]
            if control_type is None:
                if self.model_name != "pix2pix":
                    with torch.no_grad():
                        for n in range(n_iter):
                            logger.debug(f"Iteration: {n+1}/{n_iter}")
                            prompts = all_prompts[n * batch_size : (n + 1) * batch_size]
                            seeds = all_seeds[n * batch_size : (n + 1) * batch_size]
                            uc = get_learned_conditioning_with_prompt_weights(negprompt, model, clip_skip)

                            if isinstance(prompts, tuple):
                                prompts = list(prompts)

                            # no need to apply the fix here, it's applied internally
                            c = torch.cat(
                                [
                                    get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip)
                                    for prompt in prompts
                                ]
                            )

                            opt_C = 4
                            opt_f = 8
                            shape = [opt_C, height // opt_f, width // opt_f]
                            # find_noise_for_image also applies the fix internally
                            if noise_mode in ["find", "find_and_matched"]:
                                x = torch.cat(
                                    batch_size
                                    * [
                                        find_noise_for_image(
                                            model,
                                            self.model["device"],
                                            init_img.convert("RGB"),
                                            "",
                                            find_noise_steps,
                                            0.0,
                                            normalize=True,
                                            clip_skip=clip_skip,
                                        )
                                    ],
                                    dim=0,
                                )
                            else:
                                x = create_random_tensors(shape, seeds=seeds, device=self.model["device"])
                            init_data = init(model, init_img) if init_img else None
                            low_vram(
                                [
                                    (model, "cuda"),
                                    (model.cond_stage_model, "cpu"),
                                    (model.first_stage_model, "cpu"),
                                ],
                            )
                            samples_ddim = (
                                sample_img2img(
                                    init_data=init_data,
                                    ddim_steps=ddim_steps,
                                    x=x,
                                    conditioning=c,
                                    unconditional_conditioning=uc,
                                    sampler_name=sampler_name,
                                )
                                if init_img
                                else sample(
                                    init_data=init_data,
                                    x=x,
                                    conditioning=c,
                                    unconditional_conditioning=uc,
                                    sampler_name=sampler_name,
                                    karras=karras,
                                    batch_size=batch_size,
                                    shape=shape,
                                    sigma_override=sigma_override,
                                )
                            )
                            low_vram(
                                [
                                    (model, "cpu"),
                                    (model.cond_stage_model, "cpu"),
                                    (model.first_stage_model, "cuda"),
                                ],
                            )
                        if hires_fix:
                            # Put the image back together
                            temp_x = model.decode_first_stage(samples_ddim)
                            temp_x_samples_ddim = torch.clamp((temp_x + 1.0) / 2.0, min=0.0, max=1.0)
                            for i, x_sample in enumerate(temp_x_samples_ddim):
                                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                                x_sample = x_sample.astype(np.uint8)
                                temp_image = Image.fromarray(x_sample)

                            # Resize Image to final dimensions
                            temp_image = ImageOps.fit(
                                temp_image, (final_width, final_height), method=Image.Resampling.LANCZOS
                            )
                            shape = [opt_C, final_height // opt_f, final_width // opt_f]
                            x = create_random_tensors(shape, seeds=seeds, device=self.model["device"])

                            # Re-initialise the image
                            init_data_temp = init(model, temp_image)

                            # Send image for img2img processing
                            logger.debug("Hi-Res Fix Pass")
                            low_vram(
                                [
                                    (model, "cuda"),
                                    (model.cond_stage_model, "cpu"),
                                    (model.first_stage_model, "cpu"),
                                ],
                            )
                            samples_ddim = sample_img2img(
                                init_data=init_data_temp,
                                ddim_steps=ddim_steps,
                                x=x,
                                conditioning=c,
                                unconditional_conditioning=uc,
                                sampler_name=sampler_name,
                            )
                            low_vram(
                                [
                                    (model, "cpu"),
                                    (model.cond_stage_model, "cpu"),
                                    (model.first_stage_model, "cuda"),
                                ],
                            )
                else:
                    init_image = init_img
                    init_image = ImageOps.fit(init_image, (width, height), method=Image.Resampling.LANCZOS).convert(
                        "RGB"
                    )
                    null_token = model.get_learned_conditioning([""], 1)
                    with torch.no_grad():
                        for n in range(n_iter):
                            logger.debug(f"Iteration: {n+1}/{n_iter}")
                            prompts = all_prompts[n * batch_size : (n + 1) * batch_size]
                            seeds = all_seeds[n * batch_size : (n + 1) * batch_size]

                            cond = {}
                            c = torch.cat(
                                [
                                    get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip)
                                    for prompt in prompts
                                ]
                            )
                            cond["c_crossattn"] = [c]
                            init_image = 2 * torch.tensor(np.array(init_image)).float() / 255 - 1
                            init_image = rearrange(init_image, "h w c -> 1 c h w").to(self.model["device"])
                            cond["c_concat"] = [model.encode_first_stage(init_image).mode()]

                            uncond = {}
                            uncond["c_crossattn"] = [null_token]
                            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                            init_data = init(model, init_img) if init_img else None
                            x0, z_mask = init_data

                            extra_args = {
                                "cond": cond,
                                "uncond": uncond,
                                "text_cfg_scale": cfg_scale,
                                "image_cfg_scale": denoising_strength * 2,
                                "mask": z_mask,
                                "x0": x0,
                            }

                            torch.manual_seed(seed)
                            z = torch.randn_like(cond["c_concat"][0])
                            low_vram(
                                [
                                    (model, "cuda"),
                                    (model.cond_stage_model, "cpu"),
                                    (model.first_stage_model, "cpu"),
                                ],
                            )
                            samples_ddim, _ = sampler.sample(
                                S=ddim_steps,
                                conditioning=extra_args["cond"],
                                unconditional_guidance_scale=extra_args["text_cfg_scale"],
                                unconditional_conditioning=extra_args["uncond"],
                                x_T=z,
                                karras=karras,
                                sigma_override=sigma_override,
                                extra_args=extra_args,
                            )
                            low_vram(
                                [
                                    (model, "cpu"),
                                    (model.cond_stage_model, "cpu"),
                                    (model.first_stage_model, "cuda"),
                                ],
                            )
            else:
                with torch.no_grad():
                    for n in range(n_iter):
                        prompts = all_prompts[n * batch_size : (n + 1) * batch_size]
                        seeds = all_seeds[n * batch_size : (n + 1) * batch_size]
                        logger.debug(f"Iteration: {n+1}/{n_iter}")
                        """
                        NOTE:
                        Use `self.control_net_model` instead of `model` for the control net
                        """
                        cond = {
                            "c_concat": [control],
                            "c_crossattn": [
                                get_learned_conditioning_with_prompt_weights(prompt, self.control_net_model, clip_skip)
                            ],
                        }
                        un_cond = {
                            "c_concat": [control],
                            "c_crossattn": [
                                get_learned_conditioning_with_prompt_weights(
                                    negprompt, self.control_net_model, clip_skip
                                )
                            ],
                        }

                        if cond["c_crossattn"][0].shape[1] != un_cond["c_crossattn"][0].shape[1]:
                            cond["c_crossattn"][0], un_cond["c_crossattn"][0] = fix_mismatched_tensors(
                                cond["c_crossattn"][0], un_cond["c_crossattn"][0], self.control_net_model
                            )
                        shape = (4, H // 8, W // 8)
                        logger.info(f"shape = {shape}")
                        self.control_net_model.control_scales = [1.0] * 13
                        low_vram(
                            [
                                (self.control_net_model, "cuda"),
                                (self.control_net_model.control_model, "cuda"),
                                (self.control_net_model.cond_stage_model, "cpu"),
                                (self.control_net_model.first_stage_model, "cpu"),
                            ],
                        )
                        samples_ddim, _ = sampler.sample(
                            ddim_steps,
                            n_iter,
                            shape,
                            cond,
                            verbose=False,
                            eta=ddim_eta,
                            unconditional_guidance_scale=cfg_scale,
                            unconditional_conditioning=un_cond,
                        )
                        low_vram(
                            [
                                (self.control_net_model, "cpu"),
                                (self.control_net_model.control_model, "cpu"),
                                (self.control_net_model.cond_stage_model, "cpu"),
                                (self.control_net_model.first_stage_model, "cuda"),
                            ],
                        )

            x_samples_ddim = (
                model.decode_first_stage(samples_ddim)
                if control_type is None
                else self.control_net_model.decode_first_stage(samples_ddim)
            )
            if control_type is None:
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            else:
                x_samples_ddim = (
                    (einops.rearrange(x_samples_ddim, "b c h w -> b h w c") * 127.5 + 127.5)
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )

        for i, x_sample in enumerate(x_samples_ddim):
            sanitized_prompt = slugify(prompts[i])
            full_path = os.path.join(os.getcwd(), sample_path)
            sample_path_i = sample_path
            base_count = get_next_sequence_number(sample_path_i)
            if karras:
                sampler_name += "_karras"
            filename = f"{base_count:05}-{ddim_steps}_{sampler_name}_{seeds[i]}_{sanitized_prompt}"[
                : 200 - len(full_path)
            ]
            if control_type is None:
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)
            else:
                image = Image.fromarray(x_sample)
            if self.safety_checker is not None and self.filter_nsfw:
                image_features = self.feature_extractor(image, return_tensors="pt").to(self.safety_checker.device)
                output_images, has_nsfw_concept = self.safety_checker(
                    clip_input=image_features.pixel_values, images=x_sample
                )
                if has_nsfw_concept and True in has_nsfw_concept:
                    logger.info(f"Image {filename} has NSFW concept")
                    image = Image.new("RGB", (512, 512))
                    image_dict["censored"] = True
            image_dict["image"] = image
            self.images.append(image_dict)

            if save_individual_images:
                path = os.path.join(sample_path, filename + "." + self.save_extension)
                success = save_sample(image, filename, sample_path_i, self.save_extension)
                if success:
                    if self.output_file_path:
                        self.output_images.append(path)
                    else:
                        self.output_images.append(image)
                else:
                    return

        self.info = f"""
                {prompt}
                Steps: {ddim_steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}
                """.strip()
        self.stats = """
                """

        for comment in self.comments:
            self.info += "\n\n" + comment

        torch_gc()

        del sampler
        if control_type is not None:
            self.control_net_manager.unload_model(loaded_control_ldm)
            del self.control_net_model

            if not self.disable_voodoo:
                del model  # cleanup voodoo model
        torch.cuda.empty_cache()
        return
