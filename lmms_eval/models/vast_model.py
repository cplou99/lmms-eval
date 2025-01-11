import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import copy
import json
import time
import subprocess

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

from pydantic import BaseModel

class TimeWindow(BaseModel):
    start: float
    end: float

class Scene(BaseModel):
    explanation: str
    cliptimeinterval: TimeWindow

class VideoReasoning(BaseModel):
    scenes: list[Scene]
        
@register_model("vast_model")
class VAST(lmms):
    """
    VAST Model
    """

    def __init__(
        self,
        vlm_name: str,
        vlm_config: str,
        llm_reasoning_name: str,
        llm_reasoning_config: str,
        frames_sampling_strategy: Optional[str] = "ffmpeg_keyframes", # ffmpeg_keyframes, resnet_keyframes
        conf_thres: Optional[int] = 0.9,
        max_num_vqa_inf: Optional[int] = 90,
        min_window_duration: Optional[int] = 5,
        add_time_instruction: Optional[bool] = False,
        window_span: Optional[float] = 60,
        vlm_caption_max_new_tokens: Optional[int] = 64,
        ffmpeg_scene_threshold: Optional[float] = 0.4,
        ffmpeg_min_segments: Optional[int] = 8,
        ffmpeg_max_segments: Optional[int] = 128,
        ffmpeg_skip_frames: Optional[int] = 1,
        ffmpeg_width_res: Optional[int] = 128,
        uniform_maxwindow2caption: Optional[int] = 60,
        uniform_minwindow2caption: Optional[int] = 5,
        batch_size: Optional[int] = 1, 
        vlm_device: Optional[str] = "cuda",
        llm_reasoning_device: Optional[str] = "cuda",
        device: Optional[str] = "cuda",
        video_decode_backend: Optional[str] = "decord",
        save_captions: Optional[bool] = True,
        load_captions: Optional[bool] = True,
        captions_dir: Optional[str] = f"{os.path.dirname(os.getcwd())}/features/captions",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.vlm_name = vlm_name
        self.vlm_config = vlm_config
        self.llm_reasoning_name = llm_reasoning_name
        self.llm_reasoning_config = llm_reasoning_config
        self.vlm_device = vlm_device
        self.llm_reasoning_device = llm_reasoning_device
        self.video_decode_backend = video_decode_backend
        self.add_time_instruction = add_time_instruction
        self.window_span = window_span
        self.captions_dir = captions_dir
        self.load_captions = load_captions
        self.save_captions = save_captions
        vlm_ModelClass = get_model(self.vlm_name)

        self.vlm_config = self.vlm_config[1:-1].replace(";", ",").replace("#", "=")
        self.vlm = vlm_ModelClass.create_from_arg_string(
           self.vlm_config,
            {
                "batch_size": batch_size,
                "device": self.vlm_device,
            },
        )
        self.vlm_max_frames_num = self.vlm.max_frames_num
        # self.vlm_caption_prompt = "Provide a detailed caption of this clip."
        self.vlm_caption_prompt = "Provide 5 unique actions or events that could distinguish this clip from the rest of the video. 15 words maximum."
        self.vlm_caption_max_new_tokens = vlm_caption_max_new_tokens

        llm_reasoning_ModelClass = get_model(self.llm_reasoning_name)
        self.llm_reasoning_config = self.llm_reasoning_config[1:-1].replace(";", ",").replace("#", "=")
        self.llm_reasoning = llm_reasoning_ModelClass.create_from_arg_string(
           self.llm_reasoning_config,
            {
                "batch_size": batch_size,
                "device": self.llm_reasoning_device,
            },
        )
        self.llm_reasoning_max_frames_num = self.llm_reasoning.max_frames_num

        self.llm_reasoning_prompt1 = "You are a helpful video question answering assistant. The user provides some captions of the video with a question to be answered."
        self.llm_reasoning_prompt2 = "Identify and return the top5 scenes from the list above that are most likely to contain the visual information needed to answer the question."
        
        self.llm_resp_format = VideoReasoning
        self.max_num_vqa_inf = max_num_vqa_inf
        self.frames_sampling_strategy = frames_sampling_strategy   
        self.conf_thres = conf_thres
        self.min_window_duration = min_window_duration

        self.ffmpeg_scene_threshold = ffmpeg_scene_threshold
        self.ffmpeg_min_segments = ffmpeg_min_segments
        self.ffmpeg_max_segments = ffmpeg_max_segments
        self.ffmpeg_skip_frames = ffmpeg_skip_frames
        self.ffmpeg_width_res = ffmpeg_width_res
        self.ffmpeg_video_windows = {}

        self.uniform_maxwindow2caption = uniform_maxwindow2caption
        self.uniform_minwindow2caption = uniform_minwindow2caption


        if self.save_captions or self.load_captions:
            os.makedirs(self.captions_dir, exist_ok=True)
            self.captions_filename = os.path.join(self.captions_dir, f'{vlm_name}_window{window_span}s_toks{self.vlm_caption_max_new_tokens}.json')
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
        video_info = {"fps": fps, "total_duration": video_time}
        return video_info

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
    

    def extract_keyframes_ffmpeg(self, video_path, video_info, start_time, end_time, scene_threshold, skip_frames, width_res):
        scale_filter = f"scale={width_res}:-1"
        command = [
                "ffmpeg", "-ss", f"{round(start_time, 2)}", "-to", f"{round(end_time,2)}", "-i", video_path,
                "-vf", f"{scale_filter},select='not(mod(n,{skip_frames}))*gt(scene,{scene_threshold})',metadata=print",
                "-vsync", "vfr",
                "-f", "null", "-"
            ]
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stderr_output = process.stderr
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
            return None, None, None

        # Extract frame indices and timestamps from FFmpeg output
        frames_info = []
        for line in stderr_output.split("\n"):
            if "pts_time:" in line and "frame:" in line:
                try:
                    # Parse filtered frame index (`n:`)
                    filtered_frame_index = int(line.split("frame:")[1].split()[0])
                    # Parse pts_time
                    pts_time = start_time + float(line.split("pts_time:")[1].split()[0])
                    # Compute global frame index
                    global_frame_index = int(pts_time * video_info["fps"])

                except (IndexError, ValueError) as e:
                    print(f"Error parsing line: {line}\n{e}")

            elif "score" in line:
                scene_score = round(float(line.split("score=")[1]), 3)

                # Append parsed data
                frames_info.append({
                    "filtered_index": filtered_frame_index,  # Index in filtered sequence
                    "global_index": global_frame_index,  # Global index in the video
                    "timestamp": pts_time,  # Timestamp in video timeline
                    "scene_score": scene_score
                })
                # print(line)
        
        return frames_info

    def extract_windows_ffmpeg(self, video_path, video_info, start, end):
        frames_info = []
        threshold = self.ffmpeg_scene_threshold

        frames_info = self.extract_keyframes_ffmpeg(video_path, video_info, start, end, threshold, self.ffmpeg_skip_frames, self.ffmpeg_width_res)
        while len(frames_info) < self.ffmpeg_min_segments:
            threshold /= 2
            if threshold < 0.001:
                print(f"Video '{video_path}' has less than {self.ffmpeg_min_segments} keyframes ({len(frames_info)}). Threshold is too low.")
                frames_info = [{
                    "filtered_index": k,  # Index in filtered sequence
                    "global_index": int(video_info["total_duration"]*video_info["fps"]*k/self.ffmpeg_min_segments),  # Global index in the video
                    "timestamp": video_info["total_duration"]*k/self.ffmpeg_min_segments,
                    "scene_score": 0.1
                } for k in range(self.ffmpeg_min_segments)]
                break
            frames_info = self.extract_keyframes_ffmpeg(video_path, video_info, start, end, threshold, self.ffmpeg_skip_frames, self.ffmpeg_width_res)

        # Extract timestamps and indices
        frame_indices = [frame["global_index"] for frame in frames_info]
        timestamps = [frame["timestamp"] for frame in frames_info]
        scores = [frame["scene_score"] for frame in frames_info]

        # Return the min_segments timestamps with the highest scores
        best_scores_idxs = np.argsort(scores)[::-1]
        key_timestamps = [timestamps[i] for i in best_scores_idxs[:self.ffmpeg_max_segments]]
        key_timestamps.extend([start, end])
        key_timestamps = sorted(key_timestamps)
        windows = [[key_timestamps[i], key_timestamps[i+1]] for i in range(len(key_timestamps) - 1)]
        return windows

    def generate_captions(self, video_path, video_info, window_start, window_end, explored_windows, times_and_inferences, gen_kwargs):
        window_duration = window_end - window_start
        filename = video_path.split("/")[-1].split(".")[0]
        gen_kwargs["return_dict_in_generate"], gen_kwargs["output_scores"], gen_kwargs["output_logits"] = False, False, False
        gen_kwargs["max_new_tokens"] = self.vlm_caption_max_new_tokens

        t_generate_candidates = time.time()
        if self.frames_sampling_strategy == "ffmpeg_keyframes":
            window = f"{[window_start, window_end]}"
            if video_path not in self.ffmpeg_video_windows:
                smallwindows = self.extract_windows_ffmpeg(video_path, video_info, window_start, window_end)
                self.ffmpeg_video_windows[video_path] = {window: smallwindows}
            elif window not in self.ffmpeg_video_windows[video_path]:
                smallwindows = self.extract_windows_ffmpeg(video_path, video_info, window_start, window_end)
                self.ffmpeg_video_windows[video_path][window] = smallwindows
            else:
                smallwindows = self.ffmpeg_video_windows[video_path][window]

        elif self.frames_sampling_strategy == "resnet_keyframes":
            print("TODO: Implement resnet_keyframes")

        elif self.frames_sampling_strategy == "uniform":
            smallwindow_span = self.uniform_maxwindow2caption if window_duration > 2*self.uniform_maxwindow2caption else self.uniform_minwindow2caption
            num_smallwindows = int(window_duration / smallwindow_span)
            smallwindows = [[window_start + k*smallwindow_span, window_start + (k+1)*smallwindow_span] for k in range(num_smallwindows)]

        times_and_inferences["windows_candidates_generation_time"] += time.time() - t_generate_candidates
        smallwindows = [w for w in smallwindows if w not in explored_windows]
        
        if self.load_captions and filename in self.captions:
            all_captions_dict = self.captions[filename]["captions"]
            smallwindows2caption = [w for w in smallwindows if f"[{w[0]}s-{w[1]}s]" not in all_captions_dict]
        else:
            all_captions_dict = {}
            smallwindows2caption = smallwindows

        num_inferences = len(smallwindows2caption)
        t_caption_init = time.time()
        for smallwindow in smallwindows2caption:
            smallwindow_s, smallwindow_e = smallwindow[0], smallwindow[1]
            try:
                frames, frames_times, video_time = self.load_video(video_path, self.vlm_max_frames_num, window_time=smallwindow)
                if self.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from the clip {smallwindow}. These frames are located at {frames_times}.Please answer the following questions related to this video."
                    contexts = f"{time_instruciton}\n{contexts}"
    
            except Exception as e:
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {video_path} can not load, check the source")
                continue
            
            if len(frames) > 0:
                caption = self.vlm.inference(frames, self.vlm_caption_prompt, gen_kwargs)
                all_captions_dict[f"[{smallwindow_s}s-{smallwindow_e}s]"] = caption
            else:
                print(f"No frames loaded for window [{smallwindow_s}s-{smallwindow_e}s] in video {video_path} of total duration {video_info['total_duration']}")
                smallwindows.remove(smallwindow)

        caption_time = time.time() - t_caption_init     
        if self.save_captions:
            file_dict = {"captions": all_captions_dict, "num_inferences": 0, "total_time": 0}
            self.captions[filename] = file_dict
            with open(self.captions_filename, "w") as f:
                json.dump(self.captions, f)

        times_and_inferences["num_caption_inferences"] += num_inferences
        times_and_inferences["caption_time"] += caption_time   

        captions_window = [f"[{w[0]}s-{w[1]}s]: " + all_captions_dict[f"[{w[0]}s-{w[1]}s]"] for w in smallwindows]
        return captions_window


    def generate_candidates_with_llm_reasoning(self, video_path, video_info, context, window_reason, explored_windows, times_and_inferences, gen_kwargs):
        window_start, window_end = window_reason[0], window_reason[1]
        
        all_captions = self.generate_captions(video_path, video_info, window_start, window_end, explored_windows, times_and_inferences, gen_kwargs)
        # complete_context = f"{self.llm_reasoning_prompt1}. Question: {contexts}. Captions: {all_captions}. {self.llm_reasoning_prompt2}"
        all_captions = "\n".join(all_captions)
        messages = [
                {"role": "system", "content": self.llm_reasoning_prompt1},
                {"role": "user", "content": context},
                {"role": "user", "content": all_captions},
                {"role": "system", "content": self.llm_reasoning_prompt2}
            ]

        t_llm_init = time.time()
        completion = self.llm_reasoning.inference_format(self.llm_resp_format, messages)
        times_and_inferences["num_llm_inferences"] += 1
        times_and_inferences["llm_reasoning_time"] += time.time() - t_llm_init
        times_and_inferences["llm_tokens_usage"] += completion.usage.total_tokens
        response = completion.choices[0].message
        scenes = [
            {
                "explanation": scene.explanation,
                "time": scene.cliptimeinterval
                
            }
            for scene in response.parsed.scenes
        ]
        new_candidates = []
        for scene in scenes:
            try:
                # new_candidate = [float(scene["time"].split("-")[0].replace("[", "").replace("s", "").strip()), float(scene["time"].split("-")[1].replace("]", "").replace("s", "").strip())]
                new_candidate = [scene["time"].start, scene["time"].end]
                new_candidates.append(new_candidate)
            except Exception as e:
                print(f"Error parsing scene: {scene}")
                print(e)
        return new_candidates

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:

            times_and_inferences = {
                "caption_time": 0,
                "llm_reasoning_time": 0,
                "windows_candidates_generation_time": 0,
                "vqa_time": 0,
                "total_time": 0,
                "num_caption_inferences": 0,
                "num_vqa_inferences": 0,
                "num_llm_inferences": 0,
                "llm_tokens_usage": 0,
            }

            t_init = time.time()
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])

            vqa_gen_kwargs = copy.deepcopy(gen_kwargs)
            vqa_gen_kwargs["return_dict_in_generate"] = True
            vqa_gen_kwargs["output_scores"] = True
            vqa_gen_kwargs["output_logits"] = True

            video_path = visuals[0]
            video_info = self.get_video_info(video_path)
            
            candidates = [[0,  video_info["total_duration"]]]
            candidates2reason = []

            num_inferences = 0

            best_confidence = 0
            best_outputs, best_window = None, None

            explored_windows = []
            caption_time = 0
            while best_confidence < self.conf_thres:
                print(f"Exploring window {candidates[0]} in the video with total duration {video_info['total_duration']}")
                window_candidate = candidates.pop(0)
                window_candidate_duration = window_candidate[1] - window_candidate[0]
                t_vqa_init = time.time()
                try:
                    frames, frames_times, video_time = self.load_video(video_path, self.vlm_max_frames_num, window_time=window_candidate)
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
                
                if len(frames) == 0:
                    print(f"No frames loaded for window {window_candidate} in video {video_path} of total duration {video_info['total_duration']}")
                    times_and_inferences["vqa_time"] += time.time() - t_vqa_init
                else:
                    outputs = self.vlm.inference(frames, contexts, vqa_gen_kwargs)
                    choices_in_question = "choices" in contexts
                    confidence = self.get_confidence_from_outputs(outputs, choices_in_question)
                    times_and_inferences["num_vqa_inferences"] += 1
                    times_and_inferences["vqa_time"] += time.time() - t_vqa_init

                    if confidence > best_confidence:
                        best_outputs = outputs
                        best_confidence = confidence
                        best_window = window_candidate
                    
                    if best_confidence > self.conf_thres:
                        print(f"Confidence threshold reached after {num_inferences} inferences, breaking")
                        break
                    elif window_candidate_duration > self.min_window_duration:
                        candidates2reason.append(window_candidate)
                
                if len(candidates) == 0:
                    # Generate new candidates for the next window from LLM reasoning
                    if len(candidates2reason) == 0:

                        if times_and_inferences["num_vqa_inferences"] > self.max_num_vqa_inf:
                            print(f"Explored in detail 15 minutes after {num_inferences} inferences, breaking")
                            break

                        window_reason = [0, video_info["total_duration"]]
                    else:
                        window_reason = candidates2reason.pop(0)
                    
                    candidates = self.generate_candidates_with_llm_reasoning(video_path, video_info, contexts, window_reason, explored_windows, times_and_inferences, gen_kwargs)
                    
                    if window_reason == [0, video_info["total_duration"]]:
                        explored_windows.extend(candidates)

                    if len(candidates) == 0:
                        print(f"NO CANDIDATES. FAIL? AFTER {num_inferences} INFERENCES")
                        break
            
            print(f"Response to question {contexts} after {num_inferences} inferences with confidence {best_confidence} is: {best_outputs['response']} in window {best_window}")
            t_end = time.time()
            total_time = t_end - t_init
            times_and_inferences["total_time"] = total_time
            times_and_inferences["num_vlm_inferences"] = times_and_inferences["num_caption_inferences"] + times_and_inferences["num_vqa_inferences"]
            answer = best_outputs
            answer["temporal_window"] = best_window
            answer["num_inferences"] = num_inferences
            answer["caption_time"] = caption_time
            answer["vqa_time"] = total_time - caption_time
            answer["total_time"] = total_time
            res.append(answer)
                
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAVid")
