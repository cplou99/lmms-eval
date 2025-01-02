import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import copy
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


@register_model("sampling_iterative_model")
class IterativeSampling(lmms):
    """
    Sequential Model
    """

    def __init__(
        self,
        vlm_pred_name: str,
        vlm_pred_config: str,
        frames_sampling_strategy: Optional[str] = "uniform", # ffmpeg_keyframes, resnet_keyframes
        num_frames_sampled: Optional[int] = 32,
        batch_size: Optional[int] = 1, 
        vlm_pred_device: Optional[str] = "cuda",
        device: Optional[str] = "cuda",
        video_decode_backend: Optional[str] = "decord",
        add_time_instruction: Optional[bool] = False,
        window_span: Optional[float] = 60,
        conf_thres: Optional[float] = 0.9,
        min_gap_threshold: Optional[float] = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.vlm_pred_name = vlm_pred_name
        self.vlm_pred_config = vlm_pred_config
        self.frames_sampling_strategy = frames_sampling_strategy   
        self.num_frames_sampled = num_frames_sampled
        self.vlm_pred_device = vlm_pred_device
        self.video_decode_backend = video_decode_backend
        self.add_time_instruction = add_time_instruction
        self.window_span = window_span
        self.conf_thres = conf_thres
        self.min_gap_threshold = min_gap_threshold
        vlm_pred_ModelClass = get_model(self.vlm_pred_name)

        self.vlm_pred_config = self.vlm_pred_config[1:-1].replace(";", ",").replace("#", "=")
        self.vlm_pred = vlm_pred_ModelClass.create_from_arg_string(
           self.vlm_pred_config,
            {
                "batch_size": batch_size,
                "device": self.vlm_pred_device,
            },
        )
        self.vlm_pred_max_frames_num = self.vlm_pred.max_frames_num

    def get_next_frame_indices(self, total_duration, all_sampled_times,  num_frames, min_gap_threshold):
        """Returns the next set of frame times to sample based on the minimum gap threshold."""
        points = [0] + sorted(all_sampled_times) + [total_duration]
        new_points = []
        max_gaps = []
        for i in range(len(points) - 1):
            gap = points[i + 1] - points[i]
            if gap > min_gap_threshold:
                mid_point = (points[i] + points[i + 1]) // 2
                max_gaps.append((gap, mid_point))

        max_gaps = sorted(max_gaps, key=lambda x: -x[0])
        
        if len(max_gaps) == 0:
            return []
        
        for gap in max_gaps[:min(num_frames, len(max_gaps))]:
            _, mid_point = gap
            new_points.append(mid_point)
        
        while len(new_points) < num_frames:
            remaining_times = [0] + sorted(all_sampled_times + new_points) + [total_duration]
            for i in range(len(remaining_times) - 1):
                if len(new_points) >= num_frames:
                    break
                mid_point = (remaining_times[i] + remaining_times[i + 1]) / 2
                if mid_point not in all_sampled_times and mid_point not in new_points:
                    new_points.append(mid_point)

        new_timestamps = sorted(new_points[:num_frames])
        return new_timestamps

    def get_video_info(self, video_file):
        from decord import VideoReader

        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()

        total_valid_frames = len(vr)
        video_time = total_valid_frames / fps
        return fps, video_time
    

    def load_video(self, video_file, frame_indices):
        from decord import VideoReader

        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()

        total_valid_frames = len(vr)
        video_time = total_valid_frames / fps

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

    def get_confidence_from_outputs(self, outputs, choices_in_question=False):
        num_tokens = outputs["num_tokens"]
        tokens = outputs["tokens"]

        if choices_in_question:
            options_token = [tok for tok in tokens.values() if tok["token"] in ["A", "B", "C", "D"]]
            if len(options_token) == 0:
                response_probs = [float(tok["top5_probs"][0]) for tok in tokens.values()]
                conf = math.prod(response_probs) ** (1/len(response_probs))
            else:
                options_token = options_token[0]
                conf = options_token["top5_probs"][0]
        else:
            response_probs = [float(tok["top5_probs"][0]) for tok in tokens.values()]
            conf = math.prod(response_probs) ** (1/len(response_probs))
        
        return conf


    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            confidence, best_confidence = 0, 0
            all_sampled_times = []
            fps, total_duration = self.get_video_info(visuals[0])
            num_inferences = 0
            while confidence <= self.conf_thres:
                curr_timestamps = self.get_next_frame_indices(total_duration, all_sampled_times,  self.vlm_pred_max_frames_num, self.min_gap_threshold)
                
                if len(curr_timestamps) == 0:
                    print("Surpassed the minimum gap threshold")
                    break

                curr_frame_idxs = [int(t*fps) for t in curr_timestamps]

                try:
                    frames, frames_times, video_time = self.load_video(visuals[0], curr_frame_idxs)
                    if self.add_time_instruction:
                        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frames_times}.Please answer the following questions related to this video."
                        contexts = f"{time_instruciton}\n{contexts}"
        
                except Exception as e:
                    eval_logger.info(f"{e}")
                    eval_logger.info(f"Video {visuals} can not load, check the source")
                    video_path = "\n".join(visuals)
                    res.append(f"Video {video_path} can not load, check the source")
                    pbar.update(1)
                    continue

                gen_kwargs["return_dict_in_generate"] = True
                gen_kwargs["output_scores"] = True
                gen_kwargs["output_logits"] = True

                outputs = self.vlm_pred.inference(frames, contexts, gen_kwargs)
                choices_in_question = "choices" in contexts
                confidence = self.get_confidence_from_outputs(outputs, choices_in_question)
                num_inferences += 1
                all_sampled_times.extend(curr_timestamps)

                if confidence > best_confidence:
                    best_outputs = outputs
                    best_confidence = confidence
            
            best_outputs["num_inferences"] = num_inferences
            res.append(best_outputs)
                
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
