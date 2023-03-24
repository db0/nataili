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
from typing import List

import os
import torch


def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    # separate token and the embeds
    if learned_embeds_path.endswith(".pt"):
        # old format
        # token = * so replace with file directory name when converting
        trained_token = os.path.basename(learned_embeds_path)
        params_dict = {trained_token: torch.tensor(List(loaded_learned_embeds["string_to_param"].items())[0][1])}
        learned_embeds_path = os.path.splitext(learned_embeds_path)[0] + ".bin"
        torch.save(params_dict, learned_embeds_path)
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        trained_token = List(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]
    elif learned_embeds_path.endswith(".bin"):
        trained_token = List(loaded_learned_embeds.keys())[0]
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

def load_learned_embed_in_clip_v2(
        learned_embeds_path, model, text_encoder, tokenizer, token=None, idempotent=False
    ):
        learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        if isinstance(token, str):
            trained_tokens = [token]
        elif isinstance(token, list):
            assert len(learned_embeds.keys()) == len(
                token
            ), "The number of tokens and the number of embeds should be the same"
            trained_tokens = token
        else:
            trained_tokens = list(learned_embeds.keys())

        for token in trained_tokens:
            trained_token = token
            if "string_to_param" in learned_embeds:
                params_dict = {trained_token: torch.tensor(list(learned_embeds["string_to_param"].items())[0][1])}
            else:
                params_dict = {trained_token: torch.tensor(learned_embeds["emb_params"])}

            learned_embeds_path = f"nataili/concepts-library/temp/{token}.bin"
            if not os.path.exists("nataili/concepts-library/temp"):
                os.makedirs("nataili/concepts-library/temp")
            torch.save(params_dict, learned_embeds_path)
            loaded_learned_embeds = torch.load(learned_embeds_path)
            
            trained_token = list(loaded_learned_embeds.keys())[0]
            embeds = loaded_learned_embeds[trained_token]

            dtype = text_encoder.get_cast_dtype()
            embeds.to(dtype)
            # num_added_tokens = tokenizer.encode(token)

            # i = 1
            # if not idempotent:
            #     while num_added_tokens == 0:
            #         print(f"The tokenizer already contains the token {token}.")
            #         token = f"{token[:-1]}-{i}>"
            #         print(f"Attempting to add the token {token}.")
            #         num_added_tokens = tokenizer.encode(token)
            #         i += 1
            # elif num_added_tokens == 0 and idempotent:
            #     print(f"The tokenizer already contains the token {token}.")
            #     print(f"Replacing {token} embedding.")

            temp_tensor = embeds.reshape(-1, model.cond_stage_model.model.token_embedding.weight.shape[1]).cuda()
            model.cond_stage_model.model.encode_text(temp_tensor)
            
            model.cond_stage_model.model.token_embedding.weight = torch.nn.Parameter(
                torch.cat((model.cond_stage_model.model.token_embedding.weight, temp_tensor), dim=0)
            )
            print (f"After = {model.cond_stage_model.model.token_embedding.weight.shape}")

        return token