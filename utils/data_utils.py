# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import random

import numpy as np
import torch
from datasets import Dataset, load_dataset
from scripts.feiyuc.torchx.d1.diffugrpo.reward_func import extract_hash_answer
from scripts.feiyuc.torchx.d1.diffugrpo.utils import (
    manifold_download_file,
    wait_for_file,
)


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SYSTEM_PROMPT = "Answer the following math question. You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here."

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
\boxed{<Your answer>}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
            "reasoning_and_answer": x["answer"],
        }
    )


def get_metamath_questions(split) -> Dataset:
    # Explicitly load JSON file
    if split == "gsm_ansaug":
        manifold_filename = "gsm_ansaug_unique.json"
    elif split == "gsm_fobar":
        manifold_filename = "gsm_fobar_unique.json"
    elif split == "gsm_rephrased":
        manifold_filename = "gsm_rephrased_unique.json"
    elif split == "gsm_sv":
        manifold_filename = "gsm_sv_unique.json"
    elif split == "math_ansaug":
        manifold_filename = "math_ansaug_unique.json"
    elif split == "math_fobar":
        manifold_filename = "math_fobar_unique.json"
    elif split == "math_rephrased":
        manifold_filename = "math_rephrased_unique.json"
    elif split == "math_sv":
        manifold_filename = "math_sv_unique.json"
    elif split == "ansaug_combined":
        manifold_filename = "ansaug_combined.json"
    elif split == "sv_combined":
        manifold_filename = "sv_combined.json"
    elif split == "rephrased_combined":
        manifold_filename = "rephrased_combined.json"
    elif split == "fobar_combined":
        manifold_filename = "fobar_combined.json"
    else:
        raise ValueError(
            f"Unknown split: {split}. Available splits: gsm_ansaug, gsm_fobar, gsm_rephrased, gsm_sv, math_ansaug, math_fobar, math_rephrased, math_sv"
        )

    # manifold_filename = "metamath_dataset.json"
    manifold_path = "tree/synthetic/data/dlm/training/unique_by_type_metamath"
    local_path = "/tmp/data"
    # Fix: Create the full file path correctly
    full_local_path = os.path.join(local_path, manifold_filename)

    local_rank = int(os.getenv("LOCAL_RANK", -1))

    # Only download on rank 0 to avoid race conditions
    if local_rank <= 0:
        print("Manifold downloading start...")
        asyncio.run(
            manifold_download_file(
                "gaid",
                os.path.join(manifold_path, manifold_filename),
                full_local_path,
            )
        )
        print("✅ Metamath data Download is complete...")

    # Synchronize if using distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    elif hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()

    # Wait for the file to be ready
    if wait_for_file(full_local_path):
        print("File is ready to be used.")
    else:
        print("File was not found within the specified timeout.")
        raise FileNotFoundError(f"Could not download or find file: {full_local_path}")

    with open(full_local_path, "r") as f:
        data = json.load(f)

    # Convert to Dataset if needed
    if not isinstance(data, Dataset):
        data = Dataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()})

    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["query"]},
            ],
            "answer": x["parsed_answer"],
            "reasoning": "<reasoning>\n" + x["reasoning"],
        }
    )


def get_deepscaler_questions() -> Dataset:
    # Explicitly load JSON file
    manifold_filename = "deepscaler_filtered_below300problem.json"
    # manifold_filename = "metamath_dataset.json"
    manifold_path = (
        "tree/synthetic/data/dlm/training/rl_training_data/DeepScaleR-Preview-Dataset/"
    )
    local_path = "/tmp/data"
    # Fix: Create the full file path correctly
    full_local_path = os.path.join(local_path, manifold_filename)

    local_rank = int(os.getenv("LOCAL_RANK", -1))

    # Only download on rank 0 to avoid race conditions
    if local_rank <= 0:
        print("Manifold downloading start...")
        asyncio.run(
            manifold_download_file(
                "gaid",
                os.path.join(manifold_path, manifold_filename),
                full_local_path,
            )
        )
        print("✅ DeepscaleR data Download is complete...")

    # Synchronize if using distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    elif hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()

    # Wait for the file to be ready
    if wait_for_file(full_local_path):
        print("File is ready to be used.")
    else:
        print("File was not found within the specified timeout.")
        raise FileNotFoundError(f"Could not download or find file: {full_local_path}")

    with open(full_local_path, "r") as f:
        data = json.load(f)

    # Convert to Dataset if needed
    if not isinstance(data, Dataset):
        data = Dataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()})

    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n" + x["problem"]},
            ],
            "answer": x["answer"],
            "reasoning": x["solution"],
        }
    )


def get_hard_questions(dataset, seqlen) -> Dataset:
    # Explicitly load JSON file
    if dataset == "dapo":
        manifold_filename = f"dapo_data_256_cut_lastparagraph_lastsentence_last10.json"
    elif "deepscaler" in dataset:
        manifold_filename = (
            f"deepscaler_data_256_cut_lastparagraph_lastsentence_last10.json"
        )
    elif dataset == "openr1_llama3":
        manifold_filename = f"openr1_llama3_data_{seqlen}_cut.json"
    elif dataset == "openr1_llama4":
        manifold_filename = f"openr1_llama4_data_{seqlen}_cut.json"
    print("loading answer cut dataset")
    manifold_path = "tree/synthetic/data/dlm/training/rl_training_data"
    local_path = "/tmp/data"
    full_local_path = os.path.join(local_path, manifold_filename)

    local_rank = int(os.getenv("LOCAL_RANK", -1))

    # Only download on rank 0 to avoid race conditions
    if local_rank <= 0:
        print("Manifold downloading start...")
        asyncio.run(
            manifold_download_file(
                "gaid",
                os.path.join(manifold_path, manifold_filename),
                full_local_path,
            )
        )
        print(f"✅{manifold_filename} data Download is complete...")

    # Synchronize if using distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    elif hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()

    # Wait for the file to be ready
    if wait_for_file(full_local_path):
        print(f"{manifold_filename} File is ready to be used.")
    else:
        print("File was not found within the specified timeout.")
        raise FileNotFoundError(f"Could not download or find file: {full_local_path}")

    with open(full_local_path, "r") as file:
        data = [json.loads(line) for line in file]

    # Convert to Dataset if needed
    if not isinstance(data, Dataset):
        data = Dataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()})
    print("using the cut answer reasoning!")
    if "rule_cut" in dataset:
        reason_col_key = "closest_256_reason_cut"
    elif "cut10tokens" in dataset:
        reason_col_key = "closest_256_reason_cut_10_before"
    else:
        reason_col_key = "closest_256_reason_cut"
    print("using the key: ", reason_col_key, " to load reasoning")
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["problem"]},
            ],
            "answer": x["answer"],
            "reasoning": x[reason_col_key],
        }
    )


def get_mixed_questions(seqlen) -> Dataset:
    """
    Combine dapo and metamath's ansaug split datasets.

    Args:
        seqlen: Sequence length parameter for dapo data

    Returns:
        Combined dataset with both dapo and metamath ansaug data
    """
    print("Loading mixed dataset: dapo + metamath ansaug")

    # Load dapo data
    dapo_data = get_hard_questions("deepscaler", seqlen)
    print(f"Loaded {len(dapo_data)} deepscaler questions")

    # Load metamath ansaug data
    metamath_data = get_metamath_questions("ansaug_combined")
    print(f"Loaded {len(metamath_data)} metamath ansaug questions")

    # Combine the datasets
    from datasets import concatenate_datasets

    combined_data = concatenate_datasets([dapo_data, metamath_data])
    print(f"Combined dataset size: {len(combined_data)} questions")

    return combined_data

