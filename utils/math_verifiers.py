import json
import os
import random
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from math_verify import parse, verify
from .parser_helper import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def manual_gsm8k_verify(answer, raw_generation):
    gt_answer = parse(answer.split("####")[-1])
    gt_answer_value = gt_answer[0] if isinstance(gt_answer, list) else gt_answer
    parsed_answer = None
    # First, try to find boxed answers
    boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
    if boxed_matches:
        for boxed_content in boxed_matches:
            boxed_content = boxed_content.strip()
            if (
                boxed_content
                and boxed_content != "..."
                and not re.match(r"^\.+$", boxed_content)
            ):
                try:
                    parsed_answer = float(boxed_content)
                    break
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[0])
                            break
                        except ValueError:
                            pass

    # If no boxed answer found, try answer tags
    if parsed_answer is None:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if answer_text:
                try:
                    parsed_answer = float(answer_text)
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[-1])
                        except ValueError:
                            pass

    # If still no answer found, try to find answer after </think> tag or similar patterns
    if parsed_answer is None:
        # Look for content after </think> tag or similar patterns
        think_patterns = [
            r"</think>\s*(.*?)(?:<\|endoftext\|>|<\|eot_id\|>|$)",
            r"think>\s*(.*?)(?:<\|endoftext\|>|<\|eot_id\|>|$)",  # Handle malformed think> tags
            r"\\think\s*(.*?)(?:<\|endoftext\|>|<\|eot_id\|>|$)",  # Handle \think pattern
        ]

        content_after_think = ""
        for pattern in think_patterns:
            think_match = re.search(pattern, raw_generation, re.IGNORECASE | re.DOTALL)
            if think_match:
                content_after_think = think_match.group(1).strip()
                break

        if content_after_think:
            # Case 1: Direct number at the start
            direct_number_match = re.match(r"^(-?\d+\.?\d*)", content_after_think)
            if direct_number_match:
                try:
                    parsed_answer = float(direct_number_match.group(1))
                except ValueError:
                    pass

            # Case 2: Look for incomplete or complete boxed content
            if parsed_answer is None:
                # Try complete boxed first
                boxed_complete = re.search(r"\\boxed{(.*?)}", content_after_think)
                if boxed_complete:
                    boxed_content = boxed_complete.group(1).strip()
                    if (
                        boxed_content
                        and boxed_content != "..."
                        and not re.match(r"^\.+$", boxed_content)
                    ):
                        try:
                            parsed_answer = float(boxed_content)
                        except ValueError:
                            numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                            if numbers:
                                try:
                                    parsed_answer = float(numbers[0])
                                except ValueError:
                                    pass

                # Try incomplete boxed (missing closing brace)
                if parsed_answer is None:
                    boxed_incomplete = re.search(
                        r"\\boxed{([^}]*?)(?:<\|endoftext\|>|<\|eot_id\|>|$)",
                        content_after_think,
                    )
                    if boxed_incomplete:
                        boxed_content = boxed_incomplete.group(1).strip()
                        if (
                            boxed_content
                            and boxed_content != "..."
                            and not re.match(r"^\.+$", boxed_content)
                        ):
                            try:
                                parsed_answer = float(boxed_content)
                            except ValueError:
                                numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                                if numbers:
                                    try:
                                        parsed_answer = float(numbers[0])
                                    except ValueError:
                                        pass

            # Case 3: Look for sentences with numbers (e.g., "Becca has 169 cards.")
            if parsed_answer is None:
                # Find the last number in the content after think tags
                numbers = re.findall(r"-?\d+\.?\d*", content_after_think)
                if numbers:
                    try:
                        parsed_answer = float(numbers[-1])
                    except ValueError:
                        pass

    # Final fallback: extract the last number from the entire generation
    if parsed_answer is None:
        all_numbers = re.findall(r"-?\d+\.?\d*", raw_generation)
        if all_numbers:
            try:
                parsed_answer = float(all_numbers[-1])
            except ValueError:
                pass
    if parsed_answer is not None:
        parsed_answer = float(parsed_answer)
    if gt_answer_value is not None:
        gt_answer_value = float(gt_answer_value)

    is_correct = parsed_answer is not None and parsed_answer == gt_answer_value
    return is_correct, parsed_answer, gt_answer_value



def manual_math_verify(answer, raw_generation):
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
    except:
        parsed_answer = None
    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()
    is_correct = False
    if parsed_answer is not None:
        is_correct = is_equiv(parsed_answer, answer)

    return bool(is_correct), answer, parsed_answer

