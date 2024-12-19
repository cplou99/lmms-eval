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
            f"--log_samples_suffix={log_suffix}"
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
        "log_suffix": "llava_onevision.VAST"
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
    }
}

# List of models to run
all_models = ["llava_nextvideo", "videochat2_mistralHD", "videollava", "llava", "llavanext-mistral", "moviechat"]

all_tasks = ["VASTbench_vqa", "VASTbench_fullvideo", "VASTbench_gtinterval", "VASTbench_gt1mininterval", "VASTbench_gtimage", "VASTbench_gtimage_gpteval", "VASTbench_gt1mininterval_gpteval"]
tasks = ["VASTbench_gt1mininterval_gpteval", "VASTbench_gtimage_gpteval"]

tasks = ["VASTbench_gt1mininterval"]

# tasks = ["VASTbench_gtinterval"]
tasks_to_models = {
    "VASTbench_vqa": ["llava_nextvideo"],
    "VASTbench_fullvideo": ["llava_nextvideo"],
    "VASTbench_gtinterval": ["llava_nextvideo"],
    "VASTbench_gt1mininterval": ["llava_onevision"],
    "VASTbench_gtimage": ["gpt4_image", "gpt4o_mini_image"],
    "VASTbench_gtimage_gpteval": ["gpt4o_image"],
    "VASTbench_gt1mininterval_gpteval": ["llama"],
    "VASTbench_fullvideo_gpteval": ["videochat2_mistralHD"],
    "VASTbench_gtinterval_loglikelihood": ["llama"],
    "VASTbench_gtinterval_loglikelihood_gpteval": ["llava_nextvideo"]
}

# Run the evaluations
for task in tasks:
    selected_models = tasks_to_models[task]
    run_model_evaluation(models_dict, selected_models, task=task)
