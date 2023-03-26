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


def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    # separate token and the embeds
    if learned_embeds_path.endswith(".pt"):
        # old format
        # token = * so replace with file directory name when converting
        trained_token = os.path.basename(learned_embeds_path)
        params_dict = {trained_token: torch.tensor(list(loaded_learned_embeds["string_to_param"].items())[0][1])}
        learned_embeds_path = os.path.splitext(learned_embeds_path)[0] + ".bin"
        torch.save(params_dict, learned_embeds_path)
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]
    elif learned_embeds_path.endswith(".bin"):
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

    embeds = loaded_learned_embeds[trained_token]
    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token

    tokenizer.add_tokens(token)

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token
