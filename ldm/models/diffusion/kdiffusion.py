import einops
import k_diffusion as K
import torch
import torch.nn as nn
import warnings
from nataili import disable_progress
from nataili.stable_diffusion.prompt_weights import fix_mismatched_tensors
from nataili.util.cast import autocast_cuda
from nataili.util.logger import logger

warnings.filterwarnings("ignore")


class KDiffusionSampler:
    def __init__(self, m, sampler, v: bool = False, callback=None):
        self.model = m
        self.model_wrap = K.external.CompVisVDenoiser(m) if v else K.external.CompVisDenoiser(m)
        self.schedule = sampler
        self.generation_callback = callback

    def get_sampler_name(self):
        return self.schedule

    @autocast_cuda
    def sample_img2img(
        self,
        init_data,
        S,
        t_enc,
        obliterate,
        conditioning,
        unconditional_guidance_scale,
        unconditional_conditioning,
        x_T,
        karras=False,
        sigma_override: dict = None,
        extra_args=None,
    ):
        logger.debug(f"model.device: {self.model.device}")
        x0, z_mask = init_data
        if sigma_override:
            if "min" not in sigma_override:
                raise ValueError("sigma_override must have a 'min' key")
            if "max" not in sigma_override:
                raise ValueError("sigma_override must have a 'max' key")
            if "rho" not in sigma_override:
                raise ValueError("sigma_override must have a 'rho' key")
        sigma_min = self.model_wrap.sigmas[0] if sigma_override is None else sigma_override["min"]
        sigma_max = self.model_wrap.sigmas[-1] if sigma_override is None else sigma_override["max"]
        sigmas = None
        if karras:
            if sigma_override is None:
                if S > 8:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.0292, 14.6146, 7.0, self.model.device)
                elif S == 8:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.0936, 14.6146, 7.0, self.model.device)
                elif S <= 7 and S > 5:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.1072, 14.6146, 7.0, self.model.device)
                elif S <= 5:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.1072, 7.0796, 9.0, self.model.device)
            else:
                sigmas = K.sampling.get_sigmas_karras(
                    S, sigma_override["min"], sigma_override["max"], sigma_override["rho"], self.model.device
                )
        else:
            sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[S - t_enc - 1]
        xi = x0 + x
        if z_mask is not None and obliterate:
            random = torch.randn(z_mask.shape, device=xi.device)
            xi = (z_mask * x) + ((1 - z_mask) * xi)  # NOTE: random is not used here. Check if this is correct.
        if extra_args is None:
            model_wrap_cfg = CFGMaskedDenoiser(self.model_wrap)
        else:
            model_wrap_cfg = CFGPix2PixDenoiser(self.model_wrap)
        if extra_args is None:
            if conditioning.shape[1] != unconditional_conditioning.shape[1]:
                conditioning, unconditional_conditioning = fix_mismatched_tensors(
                    conditioning, unconditional_conditioning, self.model
                )
            extra_args = {
                "cond": conditioning,
                "uncond": unconditional_conditioning,
                "cond_scale": unconditional_guidance_scale,
                "mask": z_mask,
                "x0": x0,
                "xi": xi,
            }
        extra_args["cond"] = extra_args["cond"].to(self.model.device)
        extra_args["uncond"] = extra_args["uncond"].to(self.model.device)
        if extra_args["mask"] is not None:
            extra_args["mask"] = extra_args["mask"].to(self.model.device)
        extra_args["x0"] = extra_args["x0"].to(self.model.device)
        extra_args["xi"] = extra_args["xi"].to(self.model.device)
        sigmas = sigmas[S - t_enc - 1 :]

        samples_ddim = None
        if self.schedule == "dpm_fast":
            samples_ddim = K.sampling.__dict__[f"sample_{self.schedule}"](
                model_wrap_cfg,
                xi,
                sigma_min,
                sigmas[0],
                S,
                extra_args=extra_args,
                disable=disable_progress.active,
                callback=self.generation_callback,
            )
        elif self.schedule == "dpm_adaptive":
            samples_ddim = K.sampling.__dict__[f"sample_{self.schedule}"](
                model_wrap_cfg,
                xi,
                sigma_min,
                sigmas[0],
                extra_args=extra_args,
                disable=disable_progress.active,
                callback=self.generation_callback,
            )
        else:
            samples_ddim = K.sampling.__dict__[f"sample_{self.schedule}"](
                model_wrap_cfg,
                xi,
                sigmas,
                extra_args=extra_args,
                disable=disable_progress.active,
                callback=self.generation_callback,
            )
        #
        return samples_ddim, None

    @autocast_cuda
    def sample(
        self,
        S,
        conditioning,
        unconditional_guidance_scale,
        unconditional_conditioning,
        x_T,
        karras=False,
        sigma_override: dict = None,
        extra_args=None,
    ):
        logger.debug(f"model.device: {self.model.device}")
        if sigma_override:
            if "min" not in sigma_override:
                raise ValueError("sigma_override must have a 'min' key")
            if "max" not in sigma_override:
                raise ValueError("sigma_override must have a 'max' key")
            if "rho" not in sigma_override:
                raise ValueError("sigma_override must have a 'rho' key")
        sigma_min = self.model_wrap.sigmas[0] if sigma_override is None else sigma_override["min"]
        sigma_max = self.model_wrap.sigmas[-1] if sigma_override is None else sigma_override["max"]
        sigmas = None
        if karras:
            if sigma_override is None:
                if S > 8:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.0292, 14.6146, 7.0, self.model.device)
                elif S == 8:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.0936, 14.6146, 7.0, self.model.device)
                elif S <= 7 and S > 5:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.1072, 14.6146, 7.0, self.model.device)
                elif S <= 5:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.1072, 7.0796, 9.0, self.model.device)
            else:
                sigmas = K.sampling.get_sigmas_karras(
                    S, sigma_override["min"], sigma_override["max"], sigma_override["rho"], self.model.device
                )
        else:
            sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]

        if extra_args is None:
            model_wrap_cfg = CFGDenoiser(self.model_wrap)
        else:
            model_wrap_cfg = CFGPix2PixDenoiser(self.model_wrap)
        if extra_args is None:
            if conditioning.shape[1] != unconditional_conditioning.shape[1]:
                conditioning, unconditional_conditioning = fix_mismatched_tensors(
                    conditioning, unconditional_conditioning, self.model
                )
            extra_args = {
                "cond": conditioning,
                "uncond": unconditional_conditioning,
                "cond_scale": unconditional_guidance_scale,
            }
            extra_args["cond"] = extra_args["cond"].to(self.model.device)
            extra_args["uncond"] = extra_args["uncond"].to(self.model.device)
        else:
            # pix2pix, assume it is correct
            pass
        samples_ddim = None
        if self.schedule == "dpm_fast":
            samples_ddim = K.sampling.__dict__[f"sample_{self.schedule}"](
                model_wrap_cfg,
                x,
                sigma_min,
                sigma_max,
                S,
                extra_args=extra_args,
                disable=disable_progress.active,
                callback=self.generation_callback,
            )
        elif self.schedule == "dpm_adaptive":
            samples_ddim = K.sampling.__dict__[f"sample_{self.schedule}"](
                model_wrap_cfg,
                x,
                sigma_min,
                sigma_max,
                extra_args=extra_args,
                disable=disable_progress.active,
                callback=self.generation_callback,
            )
        else:
            samples_ddim = K.sampling.__dict__[f"sample_{self.schedule}"](
                model_wrap_cfg,
                x,
                sigmas,
                extra_args=extra_args,
                disable=disable_progress.active,
                callback=self.generation_callback,
            )
        #
        return samples_ddim, None


class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        x_in = x
        x_in = torch.cat([x_in] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1.0 - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class CFGPix2PixDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, mask, x0):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        denoised = (
            out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        )

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1.0 - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised
