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
import inspect
import itertools
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from nataili import disable_progress
from nataili.train.dataset import EveryDreamBatch
from nataili.train.lora.lora import LoRA
from nataili.util.logger import logger


class DreamboothLoRA:
    def __init__(self):
        self.logger = logger
        self.accelerator = None
        self.tokenizer = None
        self.unet = None
        self.text_encoder = None
        self.vae = None
        self.LoRA = LoRA()
        self.progress_bar = None

    def train(
        self,
        base_checkpoint: str,
        data_root: str,
        project_name: str,
        output_dir: str,
        max_train_steps: int = None,
        base_vae_checkpoint: Optional[str] = None,
        conditional_dropout: Optional[float] = 0.04,
        flip_p: Optional[float] = 0.0,
        log_dir: Optional[str] = "logs",
        write_schedule: Optional[bool] = False,
        batch_size: Optional[int] = 1,
        seed: Optional[int] = 69,
        resolution: Optional[int] = 512,
        train_text_encoder: Optional[bool] = True,
        num_train_epochs: Optional[int] = 1,
        save_steps: Optional[int] = 500,
        gradient_accumulation_steps: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
        lora_rank: Optional[int] = 4,
        learning_rate: Optional[float] = 1e-4,
        learning_rate_text: Optional[float] = 5e-6,
        scale_lr: Optional[float] = False,
        lr_scheduler: Optional[str] = "constant",
        lr_warmup_steps: Optional[int] = 500,
        use_8bit_adam: Optional[bool] = True,
        adam_beta1: Optional[float] = 0.9,
        adam_beta2: Optional[float] = 0.999,
        adam_weight_decay: Optional[float] = 1e-2,
        adam_epsilon: Optional[float] = 1e-8,
        max_grad_norm: Optional[float] = 1.0,
        mixed_precision: Optional[str] = "no",
        resume_unet: Optional[str] = None,
        resume_text_encoder: Optional[str] = None,
        progress_bar=None,
    ):
        self.progress_bar = progress_bar
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        self.logger.info("Training LoRA")
        if train_text_encoder and gradient_accumulation_steps > 1 and self.accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        if seed is not None:
            set_seed(seed)

        if self.accelerator.is_main_process:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            base_checkpoint,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_checkpoint,
            subfolder="text_encoder",
        )
        self.vae = AutoencoderKL.from_pretrained(
            base_checkpoint or base_vae_checkpoint,
            subfolder=None if base_vae_checkpoint else "vae",
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            base_checkpoint,
            subfolder="unet",
        )
        self.unet.requires_grad_(False)
        unet_lora_params, _ = self.LoRA.inject_trainable_lora(self.unet, r=lora_rank, loras=resume_unet)

        for _up, _down in self.LoRA.extract_lora_ups_down(self.unet):
            print("Before training: Unet First Layer lora up", _up.weight.data)
            print("Before training: Unet First Layer lora down", _down.weight.data)
            break

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if train_text_encoder:
            text_encoder_lora_params, _ = self.LoRA.inject_trainable_lora(
                self.text_encoder,
                target_replace_module=["CLIPAttention"],
                r=lora_rank,
                loras=resume_text_encoder,
            )
            for _up, _down in self.LoRA.extract_lora_ups_down(
                self.text_encoder, target_replace_module=["CLIPAttention"]
            ):
                print("Before training: text encoder First Layer lora up", _up.weight.data)
                print("Before training: text encoder First Layer lora down", _down.weight.data)
                break

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if scale_lr:
            learning_rate = learning_rate * gradient_accumulation_steps * batch_size * self.accelerator.num_processes

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        text_lr = learning_rate if learning_rate_text is None else learning_rate_text

        params_to_optimize = (
            [
                {"params": itertools.chain(*unet_lora_params), "lr": learning_rate},
                {
                    "params": itertools.chain(*text_encoder_lora_params),
                    "lr": text_lr,
                },
            ]
            if train_text_encoder
            else itertools.chain(*unet_lora_params)
        )
        optimizer = optimizer_class(
            params_to_optimize,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        noise_scheduler = DDPMScheduler.from_config(base_checkpoint, subfolder="scheduler")

        train_batch = EveryDreamBatch(
            data_root=data_root,
            flip_p=flip_p,
            debug_level=1,
            batch_size=batch_size,
            conditional_dropout=conditional_dropout,
            resolution=resolution,
            tokenizer=self.tokenizer,
            seed=seed,
            log_folder=log_dir,
            write_schedule=write_schedule,
        )

        def collate_fn(batch):
            """
            Collates batches
            """
            images = [example["image"] for example in batch]
            captions = [example["caption"] for example in batch]
            tokens = [example["tokens"] for example in batch]
            runt_size = batch[0]["runt_size"]

            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()

            ret = {
                "tokens": torch.stack(tuple(tokens)),
                "image": images,
                "captions": captions,
                "runt_size": runt_size,
            }
            del batch
            return ret

        train_dataloader = torch.utils.data.DataLoader(
            train_batch, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )

        if train_text_encoder:
            (
                self.unet,
                self.text_encoder,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = self.accelerator.prepare(self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler)
        else:
            self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                self.unet, optimizer, train_dataloader, lr_scheduler
            )

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and self.vae to gpu.
        # For mixed precision training we cast the self.text_encoder and self.vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        if not train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = batch_size * self.accelerator.num_processes * gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(train_batch)}")
        self.logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        if self.progress_bar is not None:
            progress_bar = self.progress_bar.tqdm(
                range(max_train_steps),
                disable=not self.accelerator.is_local_main_process or disable_progress.active,
                desc="Steps",
            )
        else:
            progress_bar = tqdm(
                range(max_train_steps), disable=not self.accelerator.is_local_main_process or disable_progress.active
            )
            progress_bar.set_description("Steps")
        global_step = 0
        last_save = 0
        for epoch in range(num_train_epochs):
            self.unet.train()
            if train_text_encoder:
                self.text_encoder.train()

            for step, batch in enumerate(train_dataloader):
                with torch.no_grad():
                    # Convert images to latent space
                    pixel_values = batch["image"].to(memory_format=torch.contiguous_format).to(self.unet.device)
                    with self.accelerator.autocast():
                        latents = self.vae.encode(pixel_values, return_dict=False)
                    del pixel_values
                    latents = latents[0].sample() * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    cuda_caption = batch["tokens"].to(self.text_encoder.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(cuda_caption, output_hidden_states=True)[0]

                # Predict the noise residual
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if batch["runt_size"] > 0:
                    grad_scale = batch["runt_size"] / batch_size
                    with torch.no_grad():  # not required? just in case for now, needs more testing
                        for param in self.unet.parameters():
                            if param.grad is not None:
                                param.grad *= grad_scale
                        if self.text_encoder.training:
                            for param in self.text_encoder.parameters():
                                if param.grad is not None:
                                    param.grad *= grad_scale

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                        if train_text_encoder
                        else self.unet.parameters()
                    )
                    self.accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                optimizer.zero_grad()

                global_step += 1

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if save_steps and global_step - last_save >= save_steps:
                        if self.accelerator.is_main_process:
                            pipeline = StableDiffusionPipeline.from_pretrained(
                                base_checkpoint,
                                unet=self.accelerator.unwrap_model(self.unet),
                                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                            )

                            filename_unet = f"{output_dir}/lora_weight_e{epoch}_s{global_step}.pt"
                            filename_text_encoder = f"{output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                            self.logger.info(f"save weights {filename_unet}, {filename_text_encoder}")
                            self.LoRA.save_lora_weight(pipeline.unet, filename_unet)
                            if train_text_encoder:
                                self.LoRA.save_lora_weight(
                                    pipeline.text_encoder,
                                    filename_text_encoder,
                                    target_replace_module=["CLIPAttention"],
                                )

                            for _up, _down in self.LoRA.extract_lora_ups_down(pipeline.unet):
                                print(
                                    "First Unet Layer's Up Weight is now : ",
                                    _up.weight.data,
                                )
                                print(
                                    "First Unet Layer's Down Weight is now : ",
                                    _down.weight.data,
                                )
                                break
                            if train_text_encoder:
                                for _up, _down in self.LoRA.extract_lora_ups_down(
                                    pipeline.text_encoder,
                                    target_replace_module=["CLIPAttention"],
                                ):
                                    print(
                                        "First Text Encoder Layer's Up Weight is now : ",
                                        _up.weight.data,
                                    )
                                    print(
                                        "First Text Encoder Layer's Down Weight is now : ",
                                        _down.weight.data,
                                    )
                                    break

                            last_save = global_step

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                if self.progress_bar is None:
                    progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

        self.accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if self.accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                base_checkpoint,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            )

            self.logger.info("\n\nLora TRAINING DONE!\n\n")

            self.LoRA.save_lora_weight(pipeline.unet, output_dir + "/lora_weight.pt")
            if train_text_encoder:
                self.LoRA.save_lora_weight(
                    pipeline.text_encoder,
                    output_dir + "/lora_weight.text_encoder.pt",
                    target_replace_module=["CLIPAttention"],
                )

            self.LoRA.save_safeloras(
                {
                    "unet": (pipeline.unet, {"CrossAttention", "Attention", "GEGLU"}),
                    "text_encoder": (pipeline.text_encoder, {"CLIPAttention"}),
                },
                output_dir + "/lora_weight.safetensors",
            )

        return output_dir + "/lora_weight.safetensors"
