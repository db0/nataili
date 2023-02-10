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
import time
from typing import Literal, Union

import k_diffusion as K
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from torch import nn
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid

from nataili.util.logger import logger
from nataili.util.save_sample import save_sample


class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
            # Shortcut for when we don't need to run both.
            if self.cond_scale == 0.0:
                c_in = self.uc
            elif self.cond_scale == 1.0:
                c_in = c
            return self.inner_model(x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in)

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(
            x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in
        ).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class StableDiffusionUpscaler:
    def __init__(self, model, vae_840k, vae_560k, tokenizer, text_encoder, device):
        self.model = model
        self.vae_840k = vae_840k
        self.vae_560k = vae_560k
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.save_location = "stable-diffusion-upscaler/%T-%I-%P.png"
        self.SD_C = 4  # Latent dimension
        self.SD_F = 8  # Latent patch size (pixels per latent)
        self.SD_Q = 0.18215  # sd_model.scale_factor; scaling for latents in first stage models

    @torch.no_grad()
    def condition_up(self, prompts):
        return self.text_encoder(self.tokenizer(prompts))

    def clean_prompt(self, prompt):
        badchars = re.compile(r"[/\\]")
        prompt = badchars.sub("_", prompt)
        if len(prompt) > 100:
            prompt = prompt[:100] + "…"
        return prompt

    def format_filename(self, timestamp, seed, index, prompt):
        string = self.save_location
        string = string.replace("%T", f"{timestamp}")
        string = string.replace("%S", f"{seed}")
        string = string.replace("%I", f"{index:02}")
        string = string.replace("%P", self.clean_prompt(prompt))
        return string

    def save_image(self, image, **kwargs):
        filename = self.format_filename(**kwargs)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image.save(filename)

    def __call__(
        self,
        prompt: str,
        input_image: Union[str, Image.Image],
        num_samples: int = 1,
        batch_size=1,
        decoder: Literal["finetuned_840k", "finetuned_560k"] = "finetuned_840k",
        guidance_scale: float = 1.0,
        noise_aug_level: float = 0.0,
        noise_aug_type: Literal["gaussian", "fake"] = "gaussian",
        sampler: Literal[
            "k_euler", "k_euler_ancestral", "k_dpm_2_ancestral", "k_dpm_fast", "k_dpm_adaptive"
        ] = "k_dpm_adaptive",
        steps: int = 50,
        tol_scale: float = 0.25,
        eta: float = 1.0,
        seed: int = 0,
    ):
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise ValueError(f"Image path {input_image} does not exist")
            try:
                input_image = Image.open(input_image)
            except Exception as e:
                raise ValueError(f"Could not open image {input_image}: {e}")

        timestamp = int(time.time())
        if seed == 0:
            logger.info(f"Using timestamp {timestamp} as seed")
            seed = timestamp
        seed_everything(seed)

        uc = self.condition_up(batch_size * [""])
        c = self.condition_up(batch_size * [prompt])

        if decoder == "finetuned_840k":
            vae = self.vae_840k
        elif decoder == "finetuned_560k":
            vae = self.vae_560k

        image = input_image
        image = TF.to_tensor(image).to(self.device) * 2 - 1
        low_res_latent = vae.encode(image.unsqueeze(0)).sample() * self.SD_Q
        low_res_decoded = vae.decode(low_res_latent / self.SD_Q)

        [_, C, H, W] = low_res_latent.shape

        sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

        model_wrap = CFGUpscaler(self.model, uc, cond_scale=guidance_scale)
        low_res_sigma = torch.full([batch_size], noise_aug_level, device=self.device)
        x_shape = [batch_size, C, 2 * H, 2 * W]

        def do_sample(noise, extra_args):
            # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
            sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps + 1).exp().to(self.device)
            if sampler == "k_euler":
                return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
            elif sampler == "k_euler_ancestral":
                return K.sampling.sample_euler_ancestral(
                    model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta
                )
            elif sampler == "k_dpm_2_ancestral":
                return K.sampling.sample_dpm_2_ancestral(
                    model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta
                )
            elif sampler == "k_dpm_fast":
                return K.sampling.sample_dpm_fast(
                    model_wrap, noise * sigma_max, sigma_min, sigma_max, steps, extra_args=extra_args, eta=eta
                )
            elif sampler == "k_dpm_adaptive":
                sampler_opts = dict(
                    s_noise=1.0, rtol=tol_scale * 0.05, atol=tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0
                )
            return K.sampling.sample_dpm_adaptive(
                model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=eta, **sampler_opts
            )

        image_id = 0
        for _ in range((num_samples - 1) // batch_size + 1):
            if noise_aug_type == "gaussian":
                latent_noised = low_res_latent + noise_aug_level * torch.randn_like(low_res_latent)
            elif noise_aug_type == "fake":
                latent_noised = low_res_latent * (noise_aug_level**2 + 1) ** 0.5
            extra_args = {"low_res": latent_noised, "low_res_sigma": low_res_sigma, "c": c}
            noise = torch.randn(x_shape, device=self.device)
            up_latents = do_sample(noise, extra_args)

            pixels = vae.decode(up_latents / self.SD_Q)  # equivalent to sd_model.decode_first_stage(up_latents)
            pixels = pixels.add(1).div(2).clamp(0, 1)
            # Display and save samples.
            grid = TF.to_pil_image(make_grid(pixels, batch_size))
            self.save_image(grid, timestamp=timestamp, index=image_id, prompt=prompt, seed=seed)
            image_id += 1
            for j in range(pixels.shape[0]):
                img = TF.to_pil_image(pixels[j])
                self.save_image(img, timestamp=timestamp, index=image_id, prompt=prompt, seed=seed)
                image_id += 1
