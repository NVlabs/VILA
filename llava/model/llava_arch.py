#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import warnings
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            try:
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            except:
                # Haotian: workaround
                # The shape of mm_projector's weights can be all [0]s
                pass


class LlavaMetaForCausalLM(ABC):
    """This class is originally implemented by the LLaVA team and
    modified by Haotian Tang and Jason Lu based on Ji Lin's implementation
    to support multiple images and input packing."""

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)
        
        # Note (kentang-mit@): image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids_copy = input_ids.clone()
        # kentang-mit@: Otherwise tokenizer out of bounds. Embeddings of image tokens will not be used.
        input_ids_copy[input_ids_copy == IMAGE_TOKEN_INDEX] = 0
        input_embeds = self.get_model().embed_tokens(input_ids_copy)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        input_embeds_1 = [cur_input_embeds[cur_attention_mask] for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        # print("BEFORE BATCH LOOP:", len(input_ids), input_ids[0].shape, input_ids[0].device, [(x == IMAGE_TOKEN_INDEX).sum() for x in input_ids])

        # kentang-mit@: If some part of the model is executed in the loop, the the loop length needs to be a constant.
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = input_ids[batch_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[0]
                # cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # kenang-mit@: we do not have placeholdr image for text-only data now.
                # cur_image_idx += 1
                continue
            
            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_input_embeds_no_im = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            # cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels 

    def repack_multimodal_data(
        self, input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
    ):
        # kentang-mit@: reorder and repack (reduce computation overhead)
        # requires transformers replacement.
        new_inputs_embeds = []
        new_position_ids = []
        new_labels = []
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        sorted_seqlens_in_batch, sorted_idx = torch.sort(seqlens_in_batch, descending=True)
        # print(sorted_seqlens_in_batch)
        max_seqlen = inputs_embeds.shape[1]

        cur_inputs_embeds = []
        cur_position_ids = []
        cur_labels = []
        cur_batch_len = 0
        # print(sorted_seqlens_in_batch.device, len(sorted_seqlens_in_batch), max_seqlen)
        for i in range(len(sorted_seqlens_in_batch)):
            cur_seqlen = sorted_seqlens_in_batch[i].item()
            if cur_seqlen + cur_batch_len <= max_seqlen:
                cur_batch_len += cur_seqlen
                # each item: num_tokens x num_channels
                # remove padding on-the-fly
                cur_inputs_embeds.append(inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]])
                # each item: num_tokens
                cur_position_ids.append(torch.arange(cur_inputs_embeds[-1].shape[0], device=cur_inputs_embeds[-1].device))
                # each item: num_tokens
                # remove padding on-the-fly
                cur_labels.append(labels[sorted_idx[i]][attention_mask[sorted_idx[i]]])
            else:
                new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
                new_position_ids.append(torch.cat(cur_position_ids, 0))
                new_labels.append(torch.cat(cur_labels, 0))
                # The current batch is too long. We will start a new batch.
                cur_batch_len = cur_seqlen
                cur_inputs_embeds = [inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
                cur_position_ids = [torch.arange(cur_inputs_embeds[-1].shape[0], device=cur_inputs_embeds[-1].device)]
                cur_labels = [labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]]
        
        if len(cur_inputs_embeds):
            new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
            new_position_ids.append(torch.cat(cur_position_ids, 0))
            new_labels.append(torch.cat(cur_labels, 0))
        
        # print(new_position_ids[0].device, [x.shape for x in new_inputs_embeds], [x.shape for x in new_labels], [x.shape for x in new_position_ids])
        # assert 0
        new_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            new_inputs_embeds,
            batch_first=True,
            padding_value=self.pad_token_id)
        
        new_position_ids = torch.nn.utils.rnn.pad_sequence(
            new_position_ids,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        new_labels = torch.nn.utils.rnn.pad_sequence(
            new_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        new_attention_mask = new_position_ids.ne(IGNORE_INDEX)
        # sanity check
        assert new_attention_mask.sum() == attention_mask.sum()
        # print(new_inputs_embeds.shape, (new_attention_mask.sum(1)))
        # print(sorted_seqlens_in_batch.device, sorted_seqlens_in_batch, new_attention_mask.sum(1))

        # return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
        return None, new_position_ids, new_attention_mask, past_key_values, new_inputs_embeds, new_labels, sorted_seqlens_in_batch


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
