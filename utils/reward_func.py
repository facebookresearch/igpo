import re
from typing import List, Optional, Union

import numpy as np
from math_verify import parse, verify
from .math500_utils import (
    boxed_in_answer,
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)
from .math_verifiers import (
    manual_gsm8k_verify,
    manual_math_verify,
)


def extract_boxed_answer(text: str) -> str:
    # Extract content after </reasoning> tag
    # if "</reasoning>" in text:
    #     answer = text.split("</reasoning>")[-1]
    # else:
    answer = text

    # Extract content from \boxed{...}
    if "\\boxed{" in answer:
        start = answer.find("\\boxed{") + len("\\boxed{")

        # Find the matching closing brace by counting brackets
        brace_count = 1
        end = start
        while end < len(answer) and brace_count > 0:
            if answer[end] == "{":
                brace_count += 1
            elif answer[end] == "}":
                brace_count -= 1
            end += 1

        if brace_count == 0:
            return answer[start : end - 1].strip()

    return ""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func_box(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_boxed_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [3.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def correctness_reward_func(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]

    # # Extract content after </reasoning> tags
    # extracted_responses = []
    # for r in responses:
    #     if "</reasoning>" in r:
    #         after_reasoning = r.split("</reasoning>")[-1]
    #     else:
    #         after_reasoning = r
    #     extracted_responses.append(after_reasoning)
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    do_print = np.random.rand() < 0.2
    if do_print:
        print(
            "-" * 20,
            f"\n{RED}Prompt:{RESET}\n{q}\n",
            "-" * 20,
            f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
            "-" * 20,
            f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
            "-" * 20,
            f"\n{YELLOW}Box Extracted:{RESET}\n{extracted_responses[0]}\n",
        )
    rewards = []
    for extracted, ans in zip(extracted_responses, answer):
        is_correct, gt_answer, parsed_answer = manual_math_verify(ans, extracted)
        # print(
        #     f"Manual verification: {'✅' if is_correct else '❌'} (parsed: {parsed_answer}, gt: {gt_answer})"
        # )
        rewards.append(3.0 if is_correct else 0.0)
    return rewards


def int_reward_func(completions, **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("</reasoning>") == 1:
        count += 0.125
    if text.count("<answer>\n") == 1:
        count += 0.125
    if text.count("</answer>") == 1:
        count += 0.125
    if re.search(r"\\boxed\{", text):
        count += 0.25
    return count


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(
    prompts, completions, run_name, step=None, rank=None, **kwargs
) -> List[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(
            1 for i in empty_indices if solution_str[i] == ground_truth[i]
        )
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(
    prompts, completions, run_name, step=None, rank=None, **kwargs
) -> List[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = (
            0.0
            if solution is None
            else validate_sudoku_solution(solution, ground_truth, puzzle)
        )
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(
                f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})"
            )
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> List[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> List[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards

