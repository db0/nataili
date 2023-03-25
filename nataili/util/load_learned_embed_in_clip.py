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
    
    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0]
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    tokens_temp += [y]
                else:
                    embedding_weights += [y]
                    tokens_temp += [next_new_token]
                    next_new_token += 1
            out_tokens += [tokens_temp]

        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token, current_embeds.weight.shape[1])
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:]
            n = token_dict_size
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            self.transformer.set_input_embeddings(new_embedding)
        return out_tokens

    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    # separate token and the embeds
    if learned_embeds_path.endswith(".pt"):
        # old format
        # token = * so replace with file directory name when converting
        trained_token = os.path.basename(learned_embeds_path)
        if "string_to_param" in loaded_learned_embeds:
            params_dict = {trained_token: torch.tensor(list(loaded_learned_embeds["string_to_param"].items())[0][1])}
        else:
            params_dict = {trained_token: torch.tensor(loaded_learned_embeds["emb_params"])}
        learned_embeds_path = os.path.splitext(learned_embeds_path)[0] + ".bin"
        torch.save(params_dict, learned_embeds_path)
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]
    elif learned_embeds_path.endswith(".bin"):
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

    # convert [768] Tensor to [1,768]
    if len(embeds.shape) == 1:
        embeds = torch.stack([embeds])
    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)
    embeds.requires_grad = False

    token = token[1 if token.startswith("<") else None:-1 if token.endswith(">") else None]
    tokens = [f"<{token}>"] if embeds.shape[0] == 1 else [f"<{token}{i}>" for i in range(embeds.shape[0])]

    # add the token(s) in tokenizer
    for (token, embed) in zip(tokens, embeds):
        try:
            encoder_shape = text_encoder.get_input_embeddings().weight.data[0].shape
            if not encoder_shape == embed.shape:
                if encoder_shape[0] in [768, 1024] and embed.shape[0] in [768,1024]:
                    sd1_clip = "SD_1.x" #"CLIP-ViT-L/14, SD_1.x"
                    sd2_clip = "SD_2.x" #"OpenCLIP-ViT/H, SD_2.x"
                    raise RuntimeError(f"Text encoder: {sd1_clip if encoder_shape[0] == 768 else sd2_clip}, embed: {sd1_clip if embed.shape[0] == 768 else sd2_clip}")
                raise RuntimeError(f"Incompatible: embed shape {embed.shape} does not match text encoder shape {text_encoder.get_input_embeddings().weight.data[0].shape}")
            # num_added_tokens = tokenizer.add_tokens(token)
            # if num_added_tokens == 0:
            #     # simply attempt to add the token with a number suffix
            #     for i in range(0, 256):
            #         if num_added_tokens == 0:
            #             num_added_tokens = tokenizer.add_tokens(f"{token}{i}")
            #         else:
            #             break
            #     if num_added_tokens == 0:
            #         print(f"WARNING: Unable to add token {token} to tokenizer. Too many instances? Skipping addition!")
            #         continue
            # # resize the token embeddings
            # text_encoder.resize_token_embeddings(len(tokenizer))
            # # get the id for the token and assign the embed
            # token_id = tokenizer.convert_tokens_to_ids(token)
            # text_encoder.get_input_embeddings().weight.data[token_id] = embed
            # text_encoder.set_input_embeddings()

            current_embeds = text_encoder.get_input_embeddings()
            next_new_token = token_dict_size = current_embeds.weight.shape[0]
            tokens_temp = []
            out_tokens = []
            embedding_weights = []
            for y in token:
                if isinstance(y, int):
                    tokens_temp += [y]
                else:
                    embedding_weights += [y]
                    tokens_temp += [next_new_token]
                    next_new_token += 1
            out_tokens += [tokens_temp]
            if len(embedding_weights) > 0:
                new_embedding = torch.nn.Embedding(next_new_token, current_embeds.weight.shape[1])
                new_embedding.weight[:token_dict_size] = current_embeds.weight[:]
                n = token_dict_size
                for x in embedding_weights:
                    new_embedding.weight[n] = x
                    n += 1
                text_encoder.set_input_embeddings(new_embedding)
            return out_tokens

            
        except RuntimeError as e:
            print(f" (incompatible: {token}) {e}")
            return
            #print_exc()
    return

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