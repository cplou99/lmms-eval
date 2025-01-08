import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
import torch
from transformers import AutoModelForCausalLM



from lmms_eval.models.Apollo.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_mm_token,
    ApolloMMLoader,
)
# from apollo.modeling_apollo import ApolloForCausalLM
from lmms_eval.models.Apollo.conversation import conv_templates, SeparatorStyle
from lmms_eval.models.Apollo.constants import X_TOKEN, X_TOKEN_INDEX
from huggingface_hub import snapshot_download

# eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

    
def custom_encode_mm_minibatch(self, x):
    # Using the custom encode_mm_minibatch function that saves memory in GPU
    split_sizes = [x_s[0].shape[0] for x_s in x]
    x = [torch.split(torch.cat([x_s[i] for x_s in x], dim=0), self.config.encode_batch_size) for i in
        range(self.get_vision_tower().num_vision_encoders)]
    swapped_x = []
    for i in range(len(x[0])):
        swapped_x.append([x_s[i] for x_s in x])

    features = []
    for xx in swapped_x:
        xx = [x_s.to(self.device) for x_s in xx]
        with torch.no_grad():
            xx = self._encode_mm(xx).cpu()
        features.append(xx)

    x = torch.cat(features, dim=0)
    x = torch.split(x, split_sizes, dim=0)
    result = [xx.contiguous().view(-1, xx.shape[2]) for xx in x]

    # Free memory
    del x, features, swapped_x, split_sizes
    torch.cuda.empty_cache()  # Clear GPU cache
    result = [xx.to(self.device) for xx in result]
    return result

@register_model("apollo")
class Apollo(lmms):
    """
    Apollo Model
    """

    def __init__(
        self,
        pretrained: str = "GoodiesHere/Apollo-LMMs-Apollo-7B-t32",
        torch_dtype: Optional[Union[str, torch.dtype]] = "torch.bfloat16",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="qwen_2",
        use_cache=True,
        clip_sampling_ratio: float = 0.5,
        frames_per_clip: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and (device_map == "auto" or device_map == "balanced_low_0"):
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = self.pretrained.split("/")[-1]

        self.attn_implementation = attn_implementation
        self.frames_per_clip = frames_per_clip
        self.torch_dtype = torch_dtype
        self.model_path = snapshot_download(self.pretrained, repo_type="model")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=self.attn_implementation,
        )
        
        self._model = self._model.to(self._device, dtype=torch.bfloat16)

        self._config = self._model.config
        self._tokenizer = self._model.tokenizer
        self.vision_processors = self._model.vision_tower.vision_processor
        self._max_length = self._config.llm_cfg["model_max_length"]
        self.num_repeat_token = self._config.mm_connector_cfg["num_output_tokens"]
        self.mm_use_im_start_end = self._config.use_mm_start_end
        self.clip_duration = getattr(self._config, "clip_duration")
        self.clip_sampling_ratio = clip_sampling_ratio

        self.mm_processor = ApolloMMLoader(
            self.vision_processors,
            self.clip_duration,
            self.frames_per_clip,
            clip_sampling_ratio=self.clip_sampling_ratio,
            model_max_length=self._max_length,
            device=self.device,
            num_repeat_token=self.num_repeat_token
        )

        self.model.eval()
        self.model.encode_mm_minibatch = custom_encode_mm_minibatch.__get__(self.model)

        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def load_image(self, image_path):
        frame_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        num_frames_to_sample = 10

        total_frames = len(frame_files)

        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # Read and store the sampled frames
        video = []
        for idx in sampled_indices:
            frame_path = frame_files[idx]
            try:
                with Image.open(frame_path) as img:
                    # Convert the PIL image to a numpy array if needed
                    # frame = np.array(img.convert('RGB'))
                    frame = img.convert("RGB")
                    video.append(frame)
            except IOError:
                print(f"Failed to read frame at path: {frame_path}")
        return video

    def load_video(self, video_path, max_frames_num, fps, force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()

        return spare_frames, frame_time, video_time

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        print("To be implemented")
        return None

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            try:
                mm_data, replace_string = self.mm_processor.load_video(visuals[0])
            except Exception as e:
                # import pdb;pdb.set_trace()
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} can not load, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} can not load, check the source")
                pbar.update(1)
                continue

            question = contexts
            message = replace_string + "\n\n" + question

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_mm_token(prompt, self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)

            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = question

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.4
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 0.7
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "return_dict_in_generate" not in gen_kwargs:
                gen_kwargs["return_dict_in_generate"] = False
            if "output_scores" not in gen_kwargs:
                gen_kwargs["output_scores"] = False
            if "output_logits" not in gen_kwargs:
                gen_kwargs["output_logits"] = False

            # import pdb;pdb.set_trace()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    vision_input=[mm_data],
                    data_types=['video'],
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    return_dict_in_generate=gen_kwargs["return_dict_in_generate"],
                    output_scores=gen_kwargs["output_scores"],
                    output_logits=gen_kwargs["output_logits"]
                )
                

            if gen_kwargs["return_dict_in_generate"]:
                scores = output_ids.scores
                scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

                scores = scores.reshape(-1, scores.shape[0], scores.shape[-1])
                scores = torch.nn.functional.log_softmax(scores, dim=1)
                scores = scores.reshape(-1, scores.shape[-1]).cpu().numpy()
                probs = np.exp(scores)

                # print("Response without skipping special tokens:", self.tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=False)[0].strip())
                # print("Number of tokens:", output_ids.sequences.shape[-1])
                tokens_dict = {}
                for i in range(output_ids.sequences.shape[-1]):
                    out_token = self.tokenizer.decode(output_ids.sequences[0, i].item())
                    tokens_dict[i] = {'token': out_token}
                    # print(f"Token [{i}]: {out_token}")
                for i in range(output_ids.sequences.shape[-1]):
                    # print(f"Top 5 tokens for token at pos {i}")
                    # print("| token | token string | log probability | probability |")
                    top5_token_list, top5_prob_list = [], []
                    for tok_id in np.argsort(scores[:, i]).tolist()[::-1][:5]:
                        tok = self.tokenizer.decode(tok_id)
                        score = scores[tok_id, i]
                        prob = np.exp(score)
                        top5_token_list.append(tok)
                        top5_prob_list.append(prob)
                        # print(f"| {tok_id:5d} | {tok:8s} | {score:.3f} | {prob:.2%}")
                    tokens_dict[i]['top5_tokens'] = top5_token_list
                    tokens_dict[i]['top5_probs'] = top5_prob_list
                    tokens_dict[i]['avg_prob'] = np.mean(probs[:, i])
                    tokens_dict[i]['std_prob'] = np.std(probs[:, i])

                response = self.tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()
                output_dict = {
                    "response": response,
                    "num_tokens": output_ids.sequences.shape[-1],
                    "tokens": tokens_dict
                }
                output_ids = output_ids.sequences
                res.append(output_dict)
            else:
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                eval_logger.debug(f"Question: {cur_prompt}")
                eval_logger.debug(f"Answer: {outputs}")
                # import pdb;pdb.set_trace()
                res.append(outputs)
            
            del mm_data, output_ids
            torch.cuda.empty_cache()

            pbar.update(1)
        return res


    def inference(self, frames, context, gen_kwargs):
        videos = []
        video = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda()
        if self.torch_dtype == "bfloat16":
            video = video.bfloat16()
        else:
            video = video.half()
        videos.append(video)
            
        qs = context
        # import pdb;pdb.set_trace()
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs

        # This is much safer for llama3, as we now have some object type in it
        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if "llama_3" in self.conv_template:
            pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
        attention_masks = input_ids.ne(pad_token_ids).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        cur_prompt = qs

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        if "return_dict_in_generate" not in gen_kwargs:
            gen_kwargs["return_dict_in_generate"] = False
        if "output_scores" not in gen_kwargs:
            gen_kwargs["output_scores"] = False
        if "output_logits" not in gen_kwargs:
            gen_kwargs["output_logits"] = False

        # import pdb;pdb.set_trace()
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=videos,
                attention_mask=attention_masks,
                modalities="video",
                use_cache=self.use_cache,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                return_dict_in_generate=gen_kwargs["return_dict_in_generate"],
                output_scores=gen_kwargs["output_scores"],
                output_logits=gen_kwargs["output_logits"]
            )
            # output_ids_2 = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=False, max_new_tokens=50,stopping_criteria=[stopping_criteria])
            # output_ids = self.model.generate(inputs=input_ids, images=videos, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=50,use_cache=True)

        if gen_kwargs["return_dict_in_generate"]:
            scores = output_ids.scores
            scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

            scores = scores.reshape(-1, scores.shape[0], scores.shape[-1])
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1]).cpu().numpy()
            probs = np.exp(scores)

            # print("Response without skipping special tokens:", self.tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=False)[0].strip())
            # print("Number of tokens:", output_ids.sequences.shape[-1])
            tokens_dict = {}
            for i in range(output_ids.sequences.shape[-1]):
                out_token = self.tokenizer.decode(output_ids.sequences[0, i].item())
                tokens_dict[i] = {'token': out_token}
                # print(f"Token [{i}]: {out_token}")
            for i in range(output_ids.sequences.shape[-1]):
                # print(f"Top 5 tokens for token at pos {i}")
                # print("| token | token string | log probability | probability |")
                top5_token_list, top5_prob_list = [], []
                for tok_id in np.argsort(scores[:, i]).tolist()[::-1][:5]:
                    tok = self.tokenizer.decode(tok_id)
                    score = scores[tok_id, i]
                    prob = np.exp(score)
                    top5_token_list.append(tok)
                    top5_prob_list.append(prob)
                    # print(f"| {tok_id:5d} | {tok:8s} | {score:.3f} | {prob:.2%}")
                tokens_dict[i]['top5_tokens'] = top5_token_list
                tokens_dict[i]['top5_probs'] = top5_prob_list
                tokens_dict[i]['avg_prob'] = np.mean(probs[:, i])
                tokens_dict[i]['std_prob'] = np.std(probs[:, i])

            response = self.tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)[0].strip()
            output_dict = {
                "response": response,
                "num_tokens": output_ids.sequences.shape[-1],
                "tokens": tokens_dict
            }
            output_ids = output_ids.sequences
            return output_dict
        else:
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            eval_logger.debug(f"Question: {cur_prompt}")
            eval_logger.debug(f"Answer: {outputs}")
            # import pdb;pdb.set_trace()
            return outputs
        
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
