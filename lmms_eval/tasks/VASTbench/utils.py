import json
import os
import random
import re
import sys
import time
import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import requests
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
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


def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]

def VASTbench_doc_to_choice(doc):
    return doc["options"]

def VASTbench_doc_to_target(doc):
    return doc["answer"]

def VASTbench_doc_to_text(doc, lmms_eval_specific_kwargs):
    candidates = []

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(doc["options"])])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}\n{post_prompt}"

def VASTbench_doc_to_text_no_options(doc, lmms_eval_specific_kwargs):
    candidates = []

    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}\n{post_prompt}"


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)



def VASTbench_doc_to_visual_gtinterval(doc):
    gt_interval_dir = yaml.safe_load("../VASTbench_lmmseval/gt_intervals")
    gt_interval_name = doc["question_id"]
    gt_interval_path = os.path.join(gt_interval_dir, gt_interval_name)+".mp4"
    return [gt_interval_path]

def VASTbench_doc_to_visual_gt1mininterval(doc):
    gt_interval_dir = yaml.safe_load("../VASTbench_lmmseval/gt_intervals_1min")
    gt_interval_name = doc["question_id"]
    gt_interval_path = os.path.join(gt_interval_dir, gt_interval_name)+".mp4"
    return [gt_interval_path]

def VASTbench_doc_to_visual_fullvideo(doc):
    video_name = os.path.join(doc["path"], doc["video_names"][0]) 
    return [video_name]


def VASTbench_doc_to_visual_gt_image(doc):
    with open(Path(__file__).parent / "VASTbench_gtimage.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    # cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    gt_images_dir = yaml.safe_load("../VASTbench_lmmseval/gt_images")
    # cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    
    gt_image_name = doc["question_id"]
    gt_image_path = os.path.join(gt_images_dir, gt_image_name)+".png"
    gt_image = Image.open(gt_image_path)
    return [gt_image.convert("RGB")]

def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original VASTbench paper, if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        for letter, option in index2ans.items():
            if option.lower() in s.lower():
                return letter
        return random.choice(all_choices)
    return matches[0]


def evaluate_VASTbench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num

def calculate_ins_level_acc_score(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    score = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        score += cat_results["score"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return {"acc": acc / ins_num, "score": score / ins_num}


NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = "gpt-4o-mini"

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

def get_eval_generic(question, answer, pred, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
        },
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # "response_format": {"type": "json_object"},
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        if "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt." in json.loads(response.content)["error"]["message"]:
            eval_logger.error(f"Repetitive patterns in prompt. Drop this data.")
            return "", ""

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed.")
            return "", ""

    return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        score = review_dict.get("score", 0)
        return int(score)
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return 0
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return 0
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return 0


def parse_acc(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        pred = review_dict.get("pred", "no")
        return str(pred)
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return "no"
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return "no"
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return "no"


def gpt_eval(data_dict):
    evaluated_results = []

    try:
        question = data_dict["question"]
        answer = data_dict["answer"]
        pred = data_dict["pred"]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval_generic(question, answer, pred, 64)
        score = parse_score(review)
        acc = parse_acc(review)
    except Exception as e:
        eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = ""
        score = 0
        acc = "no"

    # Update the dictionary with the new entries
    updated_dict = {
        "question_id": data_dict["question_id"],
        "review": review,
        "score": score,
        "acc": acc,
    }

    return updated_dict


# Process result for evaluation in generic task
def VASTbench_process_results_gpt_eval(doc, result):
    pred = result[0]
    doc["pred"] = pred
    eval_results = gpt_eval(doc)

    return {
        "gpt_eval_score_acc": {"question_id": doc["question_id"], "dataset": doc["dataset"], "category": doc["category"],  "question": doc["question"], "answer": doc["answer"], "pred": pred, "score": eval_results["score"], "acc":eval_results["acc"], "review": eval_results["review"]}
    }


def evaluate_VASTbench_gpteval(results):
    score, acc = 0, 0
    for result in results:
        eval_score = result["score"]
        try:
            eval_score = int(eval_score)
        except:
            eval_score = 0.0
        score += eval_score

        eval_acc = result["acc"]
        try:
            eval_acc = str(eval_acc)
            if eval_acc == "yes":
                acc += 1
        except:
            acc += 0
    return {"score": score / len(results), "acc": acc / len(results)}



def VASTbench_aggregate_results_gpt_eval(results):
    complete_evaluation_result = {'dataset': {}, 'dataset_category': {}, 'category': {}}
    to_eval_samples = {'dataset': defaultdict(list), 'dataset_category': defaultdict(list), 'category': defaultdict(list)}
    for result in results:
        to_eval_samples['dataset'][result["dataset"]].append(result)
        to_eval_samples['dataset_category'][result["category"]].append(result)
        to_eval_samples['category'][result["category"].split("-")[-1]].append(result)
    
    for split, samples in to_eval_samples.items():
        evaluation_result = {}
        for subset, sub_eval_samples in samples.items():
            metric_dict = evaluate_VASTbench_gpteval(sub_eval_samples)
            metric_dict.update({"num_example": len(sub_eval_samples)})
            evaluation_result[subset] = metric_dict
        printable_results = {}

        for cat_name, cat_results in evaluation_result.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
                "score": round(cat_results["score"], 5),
            }

        all_ins = calculate_ins_level_acc_score(evaluation_result)
        
        all_ins_score = all_ins["score"]
        all_ins_acc = all_ins["acc"]

        printable_results["Overall"] = {
            "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
            "acc": round(all_ins_acc, 5),
            "score": round(all_ins_score, 5),
        }
        eval_logger.info(printable_results)
        complete_evaluation_result[split] = evaluation_result
    complete_evaluation_result['Overall'] = printable_results["Overall"]
    return complete_evaluation_result


def VASTbench_process_results(doc, results):
    pred = results[0]
    if type(pred) == dict:
        pred_dict = pred
        pred = pred_dict["response"]
    else:
        pred_dict = None
    all_choices = []
    index2ans = {}
    for i in range(4):
        option = doc['options'][i]
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["question_id"]
    if pred_dict is not None:
        acc = {"id": id, "dataset": doc["dataset"], "category": doc["category"], "answer": doc["gt_option"], "parsed_pred": parsed_pred, "pred_dict": pred_dict}
    else:
        acc = {"id": id, "dataset": doc["dataset"], "category": doc["category"], "answer": doc["gt_option"], "parsed_pred": parsed_pred}
    return {
        "acc": acc,
        "submission": {
            id: pred,
        },
    }


def VASTbench_aggregate_results(results):
    complete_evaluation_result = {'dataset': {}, 'dataset_category': {}, 'category': {}}
    to_eval_samples = {'dataset': defaultdict(list), 'dataset_category': defaultdict(list), 'category': defaultdict(list)}
    for result in results:
        to_eval_samples['dataset'][result["dataset"]].append(result)
        to_eval_samples['dataset_category'][result["category"]].append(result)
        to_eval_samples['category'][result["category"].split("-")[-1]].append(result)
    
    for split, samples in to_eval_samples.items():
        evaluation_result = {}
        for subset, sub_eval_samples in samples.items():
            judge_dict, metric_dict = evaluate_VASTbench(sub_eval_samples)
            metric_dict.update({"num_example": len(sub_eval_samples)})
            evaluation_result[subset] = metric_dict
        printable_results = {}

        for cat_name, cat_results in evaluation_result.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
        all_ins_acc = calculate_ins_level_acc(evaluation_result)
        printable_results["Overall"] = {
            "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
            "acc": round(all_ins_acc, 5),
        }
        eval_logger.info(printable_results)
        complete_evaluation_result[split] = evaluation_result
    complete_evaluation_result['Overall'] = printable_results["Overall"]
    return complete_evaluation_result


def VASTbench_aggregate_results_for_submission(results, args):
    path = generate_submission_file("VASTbench_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")
