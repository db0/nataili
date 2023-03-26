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
import os

import torch
from slugify import slugify

from nataili.util.cache import torch_gc
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int


class AITemplate:
    def __init__(
        self,
        pipe,
        output_dir="",
        device="cuda",
        save_extension="jpg",
        output_file_path=False,
        load_concepts=False,
        concepts_dir=None,
        verify_input=True,
        auto_cast=True,
        filter_nsfw=False,
    ):
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
        self.auto_cast = auto_cast
        self.device = device
        self.comments = []
        self.output_images = []
        self.info = ""
        self.stats = ""
        self.images = []
        self.filter_nsfw = filter_nsfw
        self.pipe = pipe

    def generate(
        self,
        prompt: str,
        ddim_steps=50,
        n_iter=1,
        cfg_scale=7.5,
        seed=None,
        height=512,
        width=512,
        save_individual_images: bool = True,
    ):
        if not self.filter_nsfw:
            self.pipe.safety_checker = None
        seed = seed_to_int(seed)

        os.makedirs(self.output_dir, exist_ok=True)

        sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        x_samples = self.pipe(
            prompt=prompt,
            guidance_scale=cfg_scale,
            num_inference_steps=ddim_steps,
            generator=generator,
            num_images_per_prompt=n_iter,
            width=width,
            height=height,
        ).images

        for i, x_sample in enumerate(x_samples):
            image_dict = {"seed": seed, "image": x_sample}

            self.images.append(image_dict)

            if save_individual_images:
                sanitized_prompt = slugify(prompt)
                sample_path_i = sample_path
                base_count = get_next_sequence_number(sample_path_i)
                full_path = os.path.join(os.getcwd(), sample_path)
                filename = f"{base_count:05}-{ddim_steps}_{seed}_{sanitized_prompt}"[: 200 - len(full_path)]

                path = os.path.join(sample_path, filename + "." + self.save_extension)
                success = save_sample(x_sample, filename, sample_path_i, self.save_extension)

                if success:
                    if self.output_file_path:
                        self.output_images.append(path)
                    else:
                        self.output_images.append(x_sample)
                else:
                    return

        self.info = f"""
                {prompt}
                Steps: {ddim_steps}, CFG scale: {cfg_scale}, Seed: {seed}
                """.strip()
        self.stats = """
                """

        for comment in self.comments:
            self.info += "\n\n" + comment

        torch_gc()

        del generator

        return


if __name__ == "__main__":
    pass
