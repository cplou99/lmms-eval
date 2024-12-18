
import numpy as np
import torch
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image


def load_video(video_file, max_num_frames=16):
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = len(vr)
    num_frames = min(max_num_frames, int(total_valid_frames))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def load_image(image_file):
    return Image.open(image_file).convert("RGB")