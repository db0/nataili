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
import re

import torch

from nataili.util.logger import logger

# When using prompt weights, use this to recover the original non-weighted prompt
prompt_filter_regex = r"[\(\)]|:\d+(\.\d+)?"


def fix_mismatched_tensors(conditioning, unconditional_conditioning, model):
    if conditioning.shape[1] < unconditional_conditioning.shape[1]:
        unconditional_conditioning = unconditional_conditioning.to(conditioning.get_device())
        dim_to_add: int = unconditional_conditioning.shape[1] - conditioning.shape[1]
        paddings = torch.zeros(conditioning.shape[0], dim_to_add, conditioning.shape[2]).to(model.device)
        conditioning = torch.cat((conditioning, paddings), dim=1)
        logger.debug(f"Updated Conditioning shape = {conditioning.shape}")

    elif conditioning.shape[1] > unconditional_conditioning.shape[1]:
        unconditional_conditioning = unconditional_conditioning.to(conditioning.get_device())
        dim_to_add: int = conditioning.shape[1] - unconditional_conditioning.shape[1]
        paddings = torch.zeros(
            unconditional_conditioning.shape[0], dim_to_add, unconditional_conditioning.shape[2]
        ).to(model.device)
        unconditional_conditioning = torch.cat((unconditional_conditioning, paddings), dim=1)
        logger.debug(f"Updated Unonditioning shape = {unconditional_conditioning.shape}")

    return conditioning, unconditional_conditioning


# We subtract the conditioning of the full prompt without the subprompt, from the conditioning of the full prompt
# The remainder is exactly what the subprompt 'adds' to the embedding vector in the context of the full prompt
# Then, we use this value to update the current embedding vector according to the desired weight of the subprompt
def update_conditioning(
    filtered_whole_prompt, filtered_whole_prompt_c, model, current_prompt_c, subprompt, weight, clip_skip=None
):
    prompt_wo_subprompt = filtered_whole_prompt.replace(subprompt, "")
    # workaround for sd2.x
    # clip_skip will be set to None if the model is sd2.x then we use the original get_learned_conditioning
    if clip_skip is not None:
        prompt_wo_subprompt_c = model.get_learned_conditioning(prompt_wo_subprompt, clip_skip)
    else:
        prompt_wo_subprompt_c = model.get_learned_conditioning(prompt_wo_subprompt)
    if filtered_whole_prompt_c.shape[1] != prompt_wo_subprompt_c.shape[1]:
        filtered_whole_prompt_c, prompt_wo_subprompt_c = fix_mismatched_tensors(
            filtered_whole_prompt_c, prompt_wo_subprompt_c, model
        )
    subprompt_contribution_to_c = filtered_whole_prompt_c - prompt_wo_subprompt_c
    current_prompt_c += (weight - 1.0) * subprompt_contribution_to_c
    return current_prompt_c


def get_learned_conditioning_with_prompt_weights(prompt, model, clip_skip=None):
    # Get a filtered prompt without (, ), and :number + conditioning
    filtered_whole_prompt = re.sub(prompt_filter_regex, "", prompt)

    # Get full prompt embedding vector
    # workaround for sd2.x
    # clip_skip will be set to None if the model is sd2.x then we use the original get_learned_conditioning
    if clip_skip is not None:
        filtered_whole_prompt_c = model.get_learned_conditioning(filtered_whole_prompt, clip_skip)
    else:
        filtered_whole_prompt_c = model.get_learned_conditioning(filtered_whole_prompt)
    current_prompt_c = filtered_whole_prompt_c

    # Find the first () delimited subprompt
    subprompt_open_i = prompt.find("(")
    subprompt_close_i = prompt.find(")", subprompt_open_i + 1)

    # Process the (next) subprompt
    while subprompt_open_i != -1 and subprompt_close_i != -1:
        subprompt = prompt[subprompt_open_i + 1 : subprompt_close_i]
        weight_i = subprompt.find(":")
        subprompt_wo_weight = subprompt[0:weight_i]

        # Process the weight if we have it
        if weight_i != -1:
            weight_str = subprompt[weight_i + 1 :]
            try:
                weight_val = float(weight_str)
                # Update the conditioning with this subprompt and weight
                logger.debug(f"Adjusting subprompt weight to {weight_val}")
                current_prompt_c = update_conditioning(
                    filtered_whole_prompt,
                    filtered_whole_prompt_c,
                    model,
                    current_prompt_c,
                    subprompt_wo_weight,
                    weight_val,
                    clip_skip,
                )
            except ValueError:
                pass

        # Find next () delimited subprompt
        subprompt_open_i = prompt.find("(", subprompt_open_i + 1)
        subprompt_close_i = prompt.find(")", subprompt_open_i + 1)

    return current_prompt_c
