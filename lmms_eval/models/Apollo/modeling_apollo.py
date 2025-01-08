from typing import List, Optional, Tuple, Union
import warnings, os, torch
import torch.nn as nn

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ContextManagers, no_init_weights
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from .configuration_apollo import ApolloConfig

from .vision_tower import ApolloVisionTower
from .mm_connector import MMConnector

IGNORE_INDEX = -100
X_TOKEN_INDEX = -200


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_connector_cfg"]
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path

    return_pths = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_pths.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find resume path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_pths.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_pths.append(cfg)

    return_list = []
    for pth in return_pths:
        return_list.append(AutoConfig.from_pretrained(pth, trust_remote_code=True))

    return return_list


def build_llm_and_tokenizer(
        llm_cfg: str,
        config: PretrainedConfig,
        attn_implementation=None,
        model_max_length=None,
        *args,
        **kwargs,
) -> PreTrainedModel:
    llm_arch = getattr(llm_cfg, "architectures")[0].lower()
    
    llm_path = llm_cfg._name_or_path
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path, config=llm_cfg, torch_dtype=eval(config.model_dtype), *args, **kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        model_max_length=llm_cfg.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False,
        **kwargs
    )

    #config.hidden_size = llm.config.hidden_size
    return llm, tokenizer


class ApolloForCausalLM(PreTrainedModel):
    def __init__(self, config: ApolloConfig, *args, **kwargs):
        super().__init__(config)
        llm_cfg, vision_tower_cfg, mm_connector_cfg = get_model_config(config)
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        # Initialize weights and apply final processing

        self.lm_head = nn.Linear(llm_cfg.hidden_size, config.vocab_size, bias=False)
        self.vision_tower = ApolloVisionTower(config, vision_tower_cfg)
        self.mm_connector = MMConnector.from_pretrained(mm_connector_cfg._name_or_path)
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.post_init()
        self.is_loaded = True

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            vision_input: Optional[List[torch.FloatTensor]] = None,
            data_types: Optional[List[str]] = None,
            return_dict: Optional[bool] = None,
            cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                vision_input,
                data_types
            )

        return self.get_llm().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            vision_input: Optional[List[torch.Tensor]] = None,
            data_types: Optional[List[str]] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if vision_input is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, vision_input, data_types=data_types)
        else:
            inputs_embeds = self.embed_tokens(inputs)

        return self.get_llm().generate(position_ids=position_ids, attention_mask=attention_mask,
                                       inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        vision_input = kwargs.pop("vision_input", None)
        data_types = kwargs.pop("data_types", None)
        inputs = self.get_llm().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values,
                                                              inputs_embeds=inputs_embeds, **kwargs)
        if vision_input is not None:
            inputs["vision_input"] = vision_input
        if data_types is not None:
            inputs["data_types"] = data_types
        return inputs

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            use_safetensors: bool = None,
            **kwargs,
    ):

        return cls.load_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def get_llm(self):
        return self.llm

    def get_vision_tower(self):
        return self.vision_tower

    def get_mm_connector(self):
        return self.mm_connector

    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)
        
        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config, trust_remote_code=True, **kwargs)
        elif isinstance(model_path_or_config, ApolloConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, ApolloConfig)}")

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        with ContextManagers([no_init_weights(_enable=True), ]):
            vlm = cls(config, *args, **kwargs)

        if hasattr(vlm, "llm") and hasattr(vlm, "vision_tower") and hasattr(vlm, "mm_connector"):
            if vlm.is_loaded:
                return vlm
            else:
                print('loading model failed!')
        else:
            print('loading model failed!')

    def _encode_mm(self, x):
        x = self.get_vision_tower()(x)
        x = self.mm_connector(x)
        return x

    def encode_mm_minibatch(self, x):
        split_sizes = [x_s[0].shape[0] for x_s in x]
        x = [torch.split(torch.cat([x_s[i] for x_s in x], dim=0), self.config.encode_batch_size) for i in
             range(self.get_vision_tower().num_vision_encoders)]
        swapped_x = []
        for i in range(len(x[0])):
            swapped_x.append([x_s[i] for x_s in x])

        features = []
        for xx in swapped_x:
            xx = self._encode_mm(xx)
            features.append(xx)
        x = torch.cat(features, dim=0)
        x = torch.split(x, split_sizes, dim=0)
        return [xx.contiguous().view(-1, xx.shape[2]) for xx in x]

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, vision_input, data_types
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or vision_input is None or input_ids.shape[1] == 1:
            if (
                    past_key_values is not None
                    and vision_tower is not None
                    and vision_input is not None
                    and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        '''
            vision_input is a list of tuples, and data_type is a list of strings:
            data_type = ['image', 'video', 'video'..., 'text']
            (for one video and two image encoders)
            vision_input = 
            [
                [image(1, T, C, H, W), image(1, T, C, H, W), image(1, T, C, H, W)],
                [video(Nc1, C, T, H, W), video(Nc1, T, C, H, W), video(Nc1, T, C, H, W)],
                [video(Nc2, C, T, H, W), video(Nc2, T, C, H, W), video(Nc2, T, C, H, W)],
            ]
            -> video encoders typlically expect (C,T,H,W), images expect (C,H,W).
        '''
        # ====================================================================================================
        merged_mm_features = self.encode_mm_minibatch(vision_input)

        if not getattr(self.config, "tune_language_model", True) and getattr(self.config, "use_mm_start_end", False):
            raise NotImplementedError
        # ====================================================================================================
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
        input_ids_copy[input_ids_copy == X_TOKEN_INDEX] = 0
        input_embeds = self.get_llm().model.embed_tokens(input_ids_copy)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        # input_ids, new_input_embeds = self.inputs_merger(input_ids, input_embeds_1, merged_mm_features)
        new_labels = []
        new_input_embeds = []
        # print("BEFORE BATCH LOOP:", len(input_ids), input_ids[0].shape, input_ids[0].device, [(x == X_TOKEN_INDEX).sum() for x in input_ids])
        # kentang-mit@: If some part of the model is executed in the loop, the the loop length needs to be a constant.
        for batch_idx, (cur_labels, cur_input_ids, mm_features) in enumerate(
                zip(labels, input_ids, merged_mm_features)):
            cur_input_ids = input_ids[batch_idx]
            num_mm = (cur_input_ids == X_TOKEN_INDEX).sum()
            if num_mm == 0:
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, mm_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                # kenang-mit@: we do not have placeholdr image for text-only data now.
                continue

            if mm_features.shape[0] != num_mm:
                print(data_types[batch_idx])
                assert num_mm == len(
                    mm_features), f'Error in {data_types[batch_idx]}{num_mm}=/={len(mm_features)} not the same number of vision tokens in and vision embeddings!'

            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = (
                    [-1] + torch.where(cur_input_ids == X_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_input_embeds_no_im = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1: image_token_indices[i + 1]])

            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_mm + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                # print("cur_new_input_embeds1", cur_new_input_embeds.shape[-1])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_mm:
                    cur_image_features = mm_features[i:i + 1]
                    cur_new_input_embeds.append(cur_image_features)
                    # print("cur_new_input_embeds2", cur_new_input_embeds.shape[-1])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.get_llm().config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                priny("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.get_llm().config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

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

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )
