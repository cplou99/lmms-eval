import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import copy
import json
import time
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.language_model.llava_qwen import LlavaQwenConfig

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
from lmms_eval.utils import handle_arg_string
from lmms_eval.models import get_model

AutoConfig.register("llava_llama", LlavaConfig)
AutoConfig.register("llava_qwen", LlavaQwenConfig)


@register_model("socratic_model")
class Socratic(lmms):
    """
    Socratic Model
    """

    def __init__(
        self,
        vlm_caption_name: str,
        vlm_caption_config: str,
        llm_vqa_name: str,
        llm_vqa_config: str,
        frames_sampling_strategy: Optional[str] = "uniform", # ffmpeg_keyframes, resnet_keyframes
        frame_rate: Optional[int] = 0.5,
        batch_size: Optional[int] = 1, 
        vlm_caption_device: Optional[str] = "cuda",
        llm_vqa_device: Optional[str] = "cuda",
        device: Optional[str] = "cuda",
        video_decode_backend: Optional[str] = "decord",
        add_time_instruction: Optional[bool] = False,
        window_span: Optional[float] = 60,
        vlm_caption_max_new_tokens: Optional[int] = 768,
        save_captions: Optional[bool] = True,
        load_captions: Optional[bool] = True,
        captions_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/captions",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.vlm_caption_name = vlm_caption_name
        self.vlm_caption_config = vlm_caption_config
        self.llm_vqa_name = llm_vqa_name
        self.llm_vqa_config = llm_vqa_config
        self.frames_sampling_strategy = frames_sampling_strategy   
        self.frame_rate = frame_rate
        self.vlm_caption_device = vlm_caption_device
        self.llm_vqa_device = llm_vqa_device
        self.video_decode_backend = video_decode_backend
        self.add_time_instruction = add_time_instruction
        self.window_span = window_span
        self.vlm_caption_max_new_tokens = vlm_caption_max_new_tokens
        self.captions_dir = captions_dir
        self.load_captions = load_captions
        self.save_captions = save_captions
        vlm_caption_ModelClass = get_model(self.vlm_caption_name)

        self.vlm_caption_config = self.vlm_caption_config[1:-1].replace(";", ",").replace("#", "=")
        self.vlm_caption = vlm_caption_ModelClass.create_from_arg_string(
           self.vlm_caption_config,
            {
                "batch_size": batch_size,
                "device": self.vlm_caption_device,
            },
        )
        self.vlm_caption_max_frames_num = self.vlm_caption.max_frames_num

        llm_vqa_ModelClass = get_model(self.llm_vqa_name)
        self.llm_vqa_config = self.llm_vqa_config[1:-1].replace(";", ",").replace("#", "=")
        self.llm_vqa = llm_vqa_ModelClass.create_from_arg_string(
           self.llm_vqa_config,
            {
                "batch_size": batch_size,
                "device": self.llm_vqa_device,
            },
        )
        self.llm_vqa_max_frames_num = self.llm_vqa.max_frames_num

        if self.save_captions or self.load_captions:
            os.makedirs(self.captions_dir, exist_ok=True)
            self.captions_filename = os.path.join(self.captions_dir, f'{vlm_caption_name}_window{window_span}s.json')
            if os.path.isfile(self.captions_filename):
                with open(self.captions_filename, "r") as f:
                    self.captions = json.load(f)
            else:
                self.captions = {}

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

    def load_video(self, video_file, max_num_frames, window_time=None):
        from decord import VideoReader

        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()

        total_valid_frames = len(vr)
        video_time = total_valid_frames / fps

        if window_time is None:
            num_frames = min(max_num_frames, int(total_valid_frames))
            frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
        else:
            start_time, end_time = window_time[0], window_time[1]
            end_time = min(end_time, video_time)
            start_frame, end_frame = int(start_time * fps), int(end_time * fps)
            total_window_frames = int((end_time - start_time) * fps) 
            num_frames = min(max_num_frames, total_window_frames)
            frame_indices = [int(total_window_frames / num_frames) * i + start_frame for i in range(num_frames)]

        frames = vr.get_batch(frame_indices)
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        else:
            frames = frames.asnumpy()
        frame_timestamps = [frame_index / fps for frame_index in frame_indices]
        frame_timestamps = ",".join([f"{i:.2f}s" for i in frame_timestamps])
        return frames, frame_timestamps, video_time

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        print("TODO: NOT IMPLEMENTED YET")
        return None

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_video_info(self, video_file):
        from decord import VideoReader

        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()

        total_valid_frames = len(vr)
        video_time = total_valid_frames / fps
        return fps, video_time


    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        num_frames = int(self.frame_rate * self.window_span)
        caption_contexts = "Provide a detailed caption of this clip."
        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            filename = visuals[0].split("/")[-1].split(".")[0]
            if self.load_captions and filename in self.captions:
                all_captions_dict = self.captions[filename]["captions"]
                num_inferences = self.captions[filename]["num_inferences"]
                caption_time = self.captions[filename]["total_time"]
            
            else:
                window_start, window_end = 0.0, self.window_span
                num_inferences = 0
                all_captions_dict = {}
                fps, video_time = self.get_video_info(visuals[0])

                gen_kwargs["max_new_tokens"] = self.vlm_caption_max_new_tokens
                t0 = time.time()
                while window_start < video_time:

                    window_time = [window_start, window_end]

                    try:
                        frames, frames_times, video_time = self.load_video(visuals[0], num_frames, window_time=window_time)
                        if self.add_time_instruction:
                            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from the clip {window_time}. These frames are located at {frames_times}.Please answer the following questions related to this video."
                            contexts = f"{time_instruciton}\n{contexts}"
            
                    except Exception as e:
                        eval_logger.info(f"{e}")
                        eval_logger.info(f"Video {visuals} can not load, check the source")
                        video_path = "\n".join(visuals)
                        res.append(f"Video {video_path} can not load, check the source")
                        pbar.update(1)
                        continue
                    
                    if frames.shape[0] == 0:
                        print(f"No more frames extracted in the temporal window {window_time} of the total length {video_time}")
                        break

                    caption = self.vlm_caption.inference(frames, caption_contexts, gen_kwargs)
                    all_captions_dict[f"[{window_start}s-{window_end}s]"] = caption

                    window_start = window_end
                    window_end = window_start + self.window_span
                    num_inferences += 1

                t1 = time.time()
                caption_time = t1 - t0
                if self.save_captions:
                    file_dict = {"captions": all_captions_dict, "num_inferences": num_inferences, "total_time": caption_time}
                    self.captions[filename] = file_dict
                    with open(self.captions_filename, "w") as f:
                        json.dump(self.captions, f)

            t1 = time.time()
            all_captions = [f"{key}: {value}" for key, value in all_captions_dict.items()]
            complete_context = f"Answer the next question from the following captions of a video. Question: {contexts}. Captions: {all_captions}"
            answer = self.llm_vqa.inference(imgs=[], contexts=complete_context, gen_kwargs=gen_kwargs)
            t2 = time.time()
            vqa_time = t2 - t1

            answer["num_inferences"] = num_inferences
            answer["caption_time"] = caption_time
            answer["vqa_time"] = vqa_time
            answer["total_time"] = caption_time + vqa_time
            res.append(answer)
                
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
