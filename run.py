import subprocess

def run_model_evaluation(models_dict, selected_models, num_processes=8, task="VASTbench_vqa", batch_size=1):
    """
    Runs the specified evaluation command for selected models.

    Args:
        models_dict (dict): Dictionary of all models with configurations.
        selected_models (list): List of model names to run.
        num_processes (int): Number of processes to use for acceleration.
        tasks (str): Tasks to evaluate the models on.
        batch_size (int): Batch size for evaluation.
    """
    for short_model_name in selected_models:
        # Get the model configuration from the main dictionary
        if short_model_name not in models_dict:
            print(f"Model {model_name} not found in the main models dictionary. Skipping...")
            continue

        model_config = models_dict[short_model_name]
        model_name = model_config["model_name"]
        model_args = model_config["model_args"]
        log_suffix = model_config["log_suffix"]

        if "output_path" in model_config.keys():
            output_path = model_config["output_path"] 
        else:
            output_path = "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/"

        # Construct the command
        command = [
            "python3", "-m", "accelerate.commands.launch",
            f"--num_processes={num_processes}",
            "-m", "lmms_eval",
            f"--model={model_name}",
            f"--model_args={model_args}",
            f"--tasks={task}",
            f"--batch_size={batch_size}",
            "--log_samples",
            f"--log_samples_suffix={log_suffix}",
            f"--output_path={output_path}"
        ]

        # Run the command
        print(f"Running evaluation for model: {model_name}")
        try:
            result = subprocess.run(command, stdout=None, stderr=None, text=True)
            print(f"Model: {model_name} - Completed")
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error for model {model_name}: {result.stderr}")
        except Exception as e:
            print(f"An error occurred while running model {model_name}: {e}")


# Main dictionary containing all model configurations
models_dict = {
    "llava_onevision": {
        "model_name": "llava_onevision",
        "model_args": "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov",
        "log_suffix": "llava_onevision.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs"
    },
    "llava": {
        "model_name": "llava",
        "model_args": "pretrained=liuhaotian/llava-v1.5-7b",
        "log_suffix": "llava.VAST"
    },
    "llavanext-mistral":{
        "model_name": "llava",
        "model_args": "pretrained=liuhaotian/llava-v1.6-mistral-7b,conv_template=mistral_instruct",
        "log_suffix": "llavanext_mistral.VAST"
    },
    "llava_nextvideo": {
        "model_name": "llava_vid",
        "model_args": "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32,mm_spatial_pool_mode=average",
        "log_suffix": "llava_nextvideo.VAST"
        # Max frames num is set to 32 to fit into 24Gb GPU memory but by default was 64
    },
    "llama": {
        "model_name": "llama_vision",
        "model_args": "pretrained=meta-llama/Llama-3.2-11B-Vision-Instruct,max_frames_num=2",
        "log_suffix": "llama.VAST"
        # Max frames num is set to 1 to fit into 24Gb GPU memory by default was 32 
    },
    "llama_vid": {
        "model_name": "llama_vid",
        "model_args": "pretrained=YanweiLi/llama-vid-7b-full-224-video-fps-1",
        "log_suffix": "llama_vid.VAST"
        # Error with the key resgistration of the tokenizer
    },
    "videochat2_mistralHD": {
        "model_name": "videochat2",
        "model_args": "pretrained=OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",
        "log_suffix": "videochat2_mistral.VAST"
    },
    "videollava": {
        "model_name": "video_llava",
        "model_args": "pretrained=LanguageBind/Video-LLaVA-7B-hf",
        "log_suffix": "videollava.VAST"
        # Load the model with FP16 instead of FP32 to fit into 24Gb GPU memory
    },
    "vila":
    {
        "model_name": "vila",
        "model_args": "pretrained=Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix",
        "log_suffix": "vila.VAST"
    },
    "moviechat": {
        "model_name": "llava_onevision_moviechat",
        "model_args": "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov",
        "log_suffix": "moviechat.VAST"
    },
    "mplugowl": {
        "model_name": "mplug_owl_video",
        "model_args": "mPLUG/mPLUG-Owl3-7B-240728",
        "log_suffix": "mplugowl.VAST"
    },
    "instructblip": {  
        "model_name": "instructblip",
        "model_args": "pretrained=Salesforce/instructblip-vicuna-7b",
        "log_suffix": "instructblip.VAST"
    },
    "qwen2": {
        "model_name": "qwen2_vl",
        "model_args": "pretrained=Qwen/Qwen2-VL-7B-Instruct",
        "log_suffix": "qwen2.VAST"
    },
    "gpt4o_image": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-2024-08-06,modality=image",
        "log_suffix": "gpt4.VAST"
    },
    "gpt4o_mini_image": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-mini-2024-07-18,modality=image",
        "log_suffix": "gpt4.VAST"
    },
    "gpt4_video": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-2024-08-06,modality=video",
        "log_suffix": "gpt4.VAST"
    },
    "gpt4_blind": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-2024-08-06,modality=blind",
        "log_suffix": "gpt4.VAST"
    },
    "gpt4o_mini_blind": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-mini-2024-07-18,modality=blind",
        "log_suffix": "gpt4.VAST"
    },
    "vast_model": {
        # Put the config of the model betwen brackets {}, change = by # and , by ;
        "model_name": "vast_model",
        "model_args": "vlm_pred_name=llava_onevision,vlm_pred_config={pretrained#lmms-lab/llava-onevision-qwen2-7b-ov}",
        # "model_args": "vlm_pred_name=llava_vid,vlm_pred_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average}",
        # "model_args": "vlm_pred_name=llama_vision,vlm_pred_config={pretrained#meta-llama/Llama-3.2-11B-Vision-Instruct;max_frames_num#2}",
        "log_suffix": "VAST_model.VAST"
    },
    "sequential_llava_vid": {
        "model_name": "sequential_model",
        "model_args": "vlm_pred_name=llava_vid,vlm_pred_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/sequential"
    },
    "sequential_llava_ov": {
        "model_name": "sequential_model",
        "model_args": "vlm_pred_name=llava_onevision,vlm_pred_config={pretrained#lmms-lab/llava-onevision-qwen2-7b-ov}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/sequential"
    },
    "sequential_llama": {
        "model_name": "sequential_model",
        "model_args": "vlm_pred_name=llama_vision,vlm_pred_config={pretrained#meta-llama/Llama-3.2-11B-Vision-Instruct;max_frames_num#2}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/sequential"
    },
    "sequential_end_llava_vid": {
        "model_name": "sequential_end_model",
        "model_args": "vlm_pred_name=llava_vid,vlm_pred_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/sequential_end"
    },
    "sequential_end_llava_ov": {
        "model_name": "sequential_end_model",
        "model_args": "vlm_pred_name=llava_onevision,vlm_pred_config={pretrained#lmms-lab/llava-onevision-qwen2-7b-ov}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/sequential_end"
    },
    "sequential_end_llama": {
        "model_name": "sequential_end_model",
        "model_args": "vlm_pred_name=llama_vision,vlm_pred_config={pretrained#meta-llama/Llama-3.2-11B-Vision-Instruct;max_frames_num#2}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/sequential_end"
    },
    "it_sampling_llava_vid": {
        "model_name": "sampling_iterative_model",
        "model_args": "vlm_pred_name=llava_vid,vlm_pred_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/iterative_sampling"
    },
    "it_sampling_llava_ov": {
        "model_name": "sampling_iterative_model",
        "model_args": "vlm_pred_name=llava_onevision,vlm_pred_config={pretrained#lmms-lab/llava-onevision-qwen2-7b-ov}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/iterative_sampling"
    },
    "it_sampling_llama": {
        "model_name": "sampling_iterative_model",
        "model_args": "vlm_pred_name=llama_vision,vlm_pred_config={pretrained#meta-llama/Llama-3.2-11B-Vision-Instruct;max_frames_num#2}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/iterative_sampling"
    },
    "socratic_llava_vid_gpt4o": {
        "model_name": "socratic_model",
        "model_args": "vlm_caption_name=llava_vid,vlm_caption_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average},llm_vqa_name=gpt4v,llm_vqa_config={model_version#gpt-4o-2024-08-06;modality#blind}",
        "log_suffix": "socratic_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/socratic"
    },
    "socratic_llava_vid_gpt4omini": {
        "model_name": "socratic_model",
        "model_args": "vlm_caption_name=llava_vid,vlm_caption_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average},llm_vqa_name=gpt4v,llm_vqa_config={model_version#gpt-4o-mini-2024-07-18;modality#blind}", 
        "log_suffix": "socratic_model.VAST",
        "output_path": "/home/cplou/PycharmProjects/VLM/VASTbench_lmmseval/logs/socratic"   
    }
}

# List of models to run
all_models = ["llava_nextvideo", "videochat2_mistralHD", "videollava", "llava", "llavanext-mistral", "moviechat"]

all_tasks = ["VASTbench_vqa", "VASTbench_fullvideo", "VASTbench_gtinterval", "VASTbench_gt1mininterval", "VASTbench_gtimage", "VASTbench_gtimage_gpteval", "VASTbench_gt1mininterval_gpteval"]
tasks = ["VASTbench_fullvideo"]

# tasks = ["VASTbench_gt1mininterval"]
# tasks = ["VASTbench_gtinterval"]
tasks_to_models = {
    "VASTbench_vqa": ["llava_nextvideo"],
    "VASTbench_fullvideo": ["it_sampling_llava_vid"], #, "sequential_llava_vid", "sequential_llava_ov", "sequential_llama", "sequential_end_llava_vid", "sequential_end_llava_ov", "sequential_end_llama"
    "VASTbench_gtinterval": ["llava_nextvideo"],
    "VASTbench_gt1mininterval": ["llava_onevision"],
    "VASTbench_gtimage": ["llava_onevision"],
    "VASTbench_gtimage_gpteval": ["gpt4o_image", "gpt4o_mini_image"],
    "VASTbench_gt1mininterval_gpteval": ["llama"],
    "VASTbench_fullvideo_gpteval": ["gpt4o_mini_blind"],
    "VASTbench_gtinterval_loglikelihood": ["llama"],
    "VASTbench_gtinterval_loglikelihood_gpteval": ["llava_nextvideo"]
}

# Run the evaluations
for task in tasks:
    selected_models = tasks_to_models[task]
    run_model_evaluation(models_dict, selected_models, task=task)

import subprocess

def run_model_evaluation(models_dict, selected_models, num_processes=8, num_machines=1, gpu_ids=0, task="VASTbench_vqa", batch_size=1):
    """
    Runs the specified evaluation command for selected models.

    Args:
        models_dict (dict): Dictionary of all models with configurations.
        selected_models (list): List of model names to run.
        num_processes (int): Number of processes to use for acceleration.
        tasks (str): Tasks to evaluate the models on.
        batch_size (int): Batch size for evaluation.
    """
    for short_model_name in selected_models:
        # Get the model configuration from the main dictionary
        if short_model_name not in models_dict:
            print(f"Model {model_name} not found in the main models dictionary. Skipping...")
            continue

        model_config = models_dict[short_model_name]
        model_name = model_config["model_name"]
        model_args = model_config["model_args"]
        log_suffix = model_config["log_suffix"]

        if "output_path" in model_config.keys():
            output_path = model_config["output_path"] 
        else:
            output_path = "/disk/VAST/VASTbench_lmmseval/logs/"

        num_params = int(1e7)
        # Construct the command
        command = [
            "python3", "-m", "accelerate.commands.launch",
            # "multi-gpu",
            # "--mixed_precision=fp16", 
            # "--use_fsdp",
            # "--fsdp_auto_wrap_policy=size_based_wrap", 
            # f"--fsdp_min_num_params={num_params}",
            f"--num_processes={num_processes}", 
            f"--num_machines={num_machines}",
            f"--gpu_ids={gpu_ids}",
            "--machine_rank=0",
            "--main_process_ip=127.0.0.1",
            "--main_process_port=29500",
            "-m", "lmms_eval",
            f"--model={model_name}",
            f"--model_args={model_args}",
            f"--tasks={task}",
            f"--batch_size={batch_size}",
            "--log_samples",
            f"--log_samples_suffix={log_suffix}",
            f"--output_path={output_path}"
        ]

        # Run the command
        print(f"Running evaluation for model: {model_name}")
        try:
            result = subprocess.run(command, stdout=None, stderr=None, text=True)
            print(f"Model: {model_name} - Completed")
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error for model {model_name}: {result.stderr}")
        except Exception as e:
            print(f"An error occurred while running model {model_name}: {e}")


# Main dictionary containing all model configurations
models_dict = {
    "llava_onevision": {
        "model_name": "llava_onevision",
        "model_args": "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov",
        "log_suffix": "llava_onevision.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs"
    },
    "llava": {
        "model_name": "llava",
        "model_args": "pretrained=liuhaotian/llava-v1.5-7b",
        "log_suffix": "llava.VAST"
    },
    "llavanext-mistral":{
        "model_name": "llava",
        "model_args": "pretrained=liuhaotian/llava-v1.6-mistral-7b,conv_template=mistral_instruct",
        "log_suffix": "llavanext_mistral.VAST"
    },
    "llava_nextvideo": {
        "model_name": "llava_vid",
        "model_args": "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32,mm_spatial_pool_mode=average",
        "log_suffix": "llava_nextvideo.VAST"
        # Max frames num is set to 32 to fit into 24Gb GPU memory but by default was 64
    },
    "llama": {
        "model_name": "llama_vision",
        "model_args": "pretrained=meta-llama/Llama-3.2-11B-Vision-Instruct,max_frames_num=32",
        "log_suffix": "llama.VAST"
        # Max frames num is set to 1 to fit into 24Gb GPU memory by default was 32 
    },
    "llama_vid": {
        "model_name": "llama_vid",
        "model_args": "pretrained=YanweiLi/llama-vid-7b-full-224-video-fps-1",
        "log_suffix": "llama_vid.VAST"
        # Error with the key resgistration of the tokenizer
    },
    "videochat2_mistralHD": {
        "model_name": "videochat2",
        "model_args": "pretrained=OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",
        "log_suffix": "videochat2_mistral.VAST"
    },
    "videollava": {
        "model_name": "video_llava",
        "model_args": "pretrained=LanguageBind/Video-LLaVA-7B-hf",
        "log_suffix": "videollava.VAST"
        # Load the model with FP16 instead of FP32 to fit into 24Gb GPU memory
    },
    "vila":
    {
        "model_name": "vila",
        "model_args": "pretrained=Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix",
        "log_suffix": "vila.VAST"
    },
    "moviechat": {
        "model_name": "llava_onevision_moviechat",
        "model_args": "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov",
        "log_suffix": "moviechat.VAST"
    },
    "mplugowl": {
        "model_name": "mplug_owl_video",
        "model_args": "mPLUG/mPLUG-Owl3-7B-240728",
        "log_suffix": "mplugowl.VAST"
    },
    "instructblip": {  
        "model_name": "instructblip",
        "model_args": "pretrained=Salesforce/instructblip-vicuna-7b",
        "log_suffix": "instructblip.VAST"
    },
    "qwen2": {
        "model_name": "qwen2_vl",
        "model_args": "pretrained=Qwen/Qwen2-VL-7B-Instruct",
        "log_suffix": "qwen2.VAST"
    },
    "gpt4o_image": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-2024-08-06,modality=image",
        "log_suffix": "gpt4.VAST"
    },
    "gpt4o_mini_image": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-mini-2024-07-18,modality=image",
        "log_suffix": "gpt4.VAST"
    },
    "gpt4_video": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-2024-08-06,modality=video",
        "log_suffix": "gpt4.VAST"
    },
    "apollo": {
        "model_name": "apollo",
        "model_args": "pretrained=GoodiesHere/Apollo-LMMs-Apollo-7B-t32",
        "log_suffix": "apollo.VAST"
    },
    "gpt4_blind": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-2024-08-06,modality=blind",
        "log_suffix": "gpt4.VAST"
    },
    "gpt4o_mini_blind": {
        "model_name": "gpt4v",
        "model_args": "model_version=gpt-4o-mini-2024-07-18,modality=blind",
        "log_suffix": "gpt4.VAST"
    },
    "llava_nextvideo_blind": {
        "model_name": "llava_vid",
        "model_args": "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32,mm_spatial_pool_mode=average,video_decode_backend=blind",
        "log_suffix": "llava_nextvideo.VAST"
        # Max frames num is set to 32 to fit into 24Gb GPU memory but by default was 64
    },
    "vast_model": {
        # Put the config of the model betwen brackets {}, change = by # and , by ;
        "model_name": "vast_model",
        "model_args": "vlm_name=llava_vid,vlm_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average},llm_reasoning_name=gpt4v,llm_reasoning_config={model_version#gpt-4o-mini-2024-07-18;modality#blind}", 
        "log_suffix": "socratic_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/vast"   
    },
    "sequential_llava_vid": {
        "model_name": "sequential_model",
        "model_args": "vlm_pred_name=llava_vid,vlm_pred_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/sequential"
    },
    "sequential_llava_ov": {
        "model_name": "sequential_model",
        "model_args": "vlm_pred_name=llava_onevision,vlm_pred_config={pretrained#lmms-lab/llava-onevision-qwen2-7b-ov}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/sequential"
    },
    "sequential_llama": {
        "model_name": "sequential_model",
        "model_args": "vlm_pred_name=llama_vision,vlm_pred_config={pretrained#meta-llama/Llama-3.2-11B-Vision-Instruct;max_frames_num#2}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/sequential"
    },
    "sequential_end_llava_vid": {
        "model_name": "sequential_end_model",
        "model_args": "vlm_pred_name=llava_vid,vlm_pred_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/sequential_end"
    },
    "sequential_end_llava_ov": {
        "model_name": "sequential_end_model",
        "model_args": "vlm_pred_name=llava_onevision,vlm_pred_config={pretrained#lmms-lab/llava-onevision-qwen2-7b-ov}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/sequential_end"
    },
    "sequential_end_llama": {
        "model_name": "sequential_end_model",
        "model_args": "vlm_pred_name=llama_vision,vlm_pred_config={pretrained#meta-llama/Llama-3.2-11B-Vision-Instruct;max_frames_num#2}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/sequential_end"
    },
    "it_sampling_llava_vid": {
        "model_name": "sampling_iterative_model",
        "model_args": "vlm_pred_name=llava_vid,vlm_pred_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/iterative_sampling"
    },
    "it_sampling_llava_ov": {
        "model_name": "sampling_iterative_model",
        "model_args": "vlm_pred_name=llava_onevision,vlm_pred_config={pretrained#lmms-lab/llava-onevision-qwen2-7b-ov}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/iterative_sampling"
    },
    "it_sampling_llama": {
        "model_name": "sampling_iterative_model",
        "model_args": "vlm_pred_name=llama_vision,vlm_pred_config={pretrained#meta-llama/Llama-3.2-11B-Vision-Instruct;max_frames_num#2}",
        "log_suffix": "sequential_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/iterative_sampling"
    },
    "socratic_llava_vid_gpt4o": {
        "model_name": "socratic_model",
        "model_args": "vlm_caption_name=llava_vid,vlm_caption_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average},llm_vqa_name=gpt4v,llm_vqa_config={model_version#gpt-4o-2024-08-06;modality#blind}",
        "log_suffix": "socratic_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/socratic"
    },
    "socratic_llava_vid_gpt4omini": {
        "model_name": "socratic_model",
        "model_args": "vlm_caption_name=llava_vid,vlm_caption_config={pretrained#lmms-lab/LLaVA-Video-7B-Qwen2;conv_template#qwen_1_5;max_frames_num#32;mm_spatial_pool_mode#average},llm_vqa_name=gpt4v,llm_vqa_config={model_version#gpt-4o-mini-2024-07-18;modality#blind}", 
        "log_suffix": "socratic_model.VAST",
        "output_path": "/disk/VAST/VASTbench_lmmseval/logs/socratic"   
    }
}

# List of models to run
all_models = ["llava_nextvideo", "videochat2_mistralHD", "videollava", "llava", "llavanext-mistral", "moviechat"]

all_tasks = ["VASTbench_vqa", "VASTbench_fullvideo", "VASTbench_gtinterval", "VASTbench_gt1mininterval", "VASTbench_gtimage", "VASTbench_gtimage_gpteval", "VASTbench_gt1mininterval_gpteval"]
tasks = ["VASTbench_fullvideo", "VASTbench_fullvideo_gpteval"]

tasks = ["VASTbench_fullvideo"]
# tasks = ["VASTbench_gtinterval"]

tasks_to_models = {
    "VASTbench_vqa": ["llava_nextvideo"],
    "VASTbench_fullvideo": ["vast_model"],  #"sequential_llava_vid", "sequential_llava_ov", "sequential_llama", "sequential_end_llava_vid", "sequential_end_llava_ov", "sequential_end_llama", "socratic_llava_vid_gpt4omini", "socratic_llava_vid_gpt4o", "it_sampling_llava_vid", "it_sampling_llava_ov", "it_sampling_llama"],
    "VASTbench_gtinterval": ["llava_nextvideo"],
    "VASTbench_gt1mininterval": ["apollo"],
    "VASTbench_gtimage": ["llava_onevision"],
    "VASTbench_gtimage_gpteval": ["gpt4o_image", "gpt4o_mini_image"],
    "VASTbench_gt1mininterval_gpteval": ["apollo"],
    "VASTbench_fullvideo_gpteval": ["sequential_llava_vid"],
    "VASTbench_gtinterval_loglikelihood": ["llama"],
    "VASTbench_gtinterval_loglikelihood_gpteval": ["llava_nextvideo"]
}

# Run the evaluations
for task in tasks:
    selected_models = tasks_to_models[task]
    run_model_evaluation(models_dict, selected_models, task=task, num_processes=1, num_machines=1, gpu_ids="0")
