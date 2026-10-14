# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ToolRL (RLLA) rewards: format in {0, 1} and tool-call correctness in [-3, 3], scored against the original label."""

import json
import re
from collections import Counter

FORMAT_MAX, FORMAT_MIN = 1.0, 0.0
CORRECTNESS_MAX, CORRECTNESS_MIN = 3.0, -3.0


def match_score(list1, list2):
    """Similarity of two lists counting element frequency and ignoring order (intersection over union)."""
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    count1 = Counter(list1)
    count2 = Counter(list2)
    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection
    return intersection / max_possible if max_possible > 0 else 0.0


def format_reward(response, label, max_possible_reward=FORMAT_MAX, min_possible_reward=FORMAT_MIN):
    """1 when the response has exactly the tag structure the label uses, else 0."""
    reward = min_possible_reward
    if "<response>" in label and "<tool_call>" not in label:
        pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
        if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
            reward = max_possible_reward
    elif "<response>" not in label and "<tool_call>" in label:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
        if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
            reward = max_possible_reward
    elif "<response>" in label and "<tool_call>" in label:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
        if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and response.count("<response>") == 1 and response.count("</response>") == 1:
            reward = max_possible_reward
    else:
        pattern = r"^<think>.*?</think>$"
        if re.search(pattern, response, re.DOTALL):
            reward = max_possible_reward
    return reward


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    """Tool names (frequency-aware match), then for each reference call the best unused predicted call with the same
    name: parameter-name match plus one point per exactly equal parameter value; rescaled to [min, max]."""
    if gt_tools == pd_tools:
        return max_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        local_max_possible += 1.0 + len(gt_params)

        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)
            total_score = param_score + correctness_score
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def correctness_reward(response, label, max_possible_reward=CORRECTNESS_MAX, min_possible_reward=CORRECTNESS_MIN):
    """0 for direct-answer labels; otherwise the tool-call score, or the minimum if the response has no parsable
    <tool_call> block."""
    if "<tool_call>" not in label:
        return 0.0
    gt_tool_call = label.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    gt_tools = [json.loads(tool) for tool in gt_tool_call.split("\n")]
    try:
        assert "<tool_call>" in response
        assert "</tool_call>" in response
        pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
        pd_tools = [json.loads(tool) for tool in pd_tools]
        return compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward)
    except Exception:
        return min_possible_reward


def assistant_turn(solution_str):
    """The assistant turn of a decoded Qwen chat sequence."""
    return solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()


def compute_score(solution_str, ground_truth):
    """Return (total, format, correctness) for one decoded prompt+response sequence."""
    response = assistant_turn(solution_str)
    format_score = format_reward(response, ground_truth)
    correctness_score = correctness_reward(response, ground_truth)
    return format_score + correctness_score, format_score, correctness_score
