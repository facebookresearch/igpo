import json
import os
import random

import numpy as np
import torch
from datasets import Dataset, load_dataset
from .reward_func import extract_hash_answer


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

def get_metamath_questions() -> Dataset:
    # Paths
    local_path = "MetaMathQA/MetaMathQA-395K.json"
    cached_path = "MetaMathQA/MetaMathQA-ansaug-filtered.json"

    # Check if filtered data already exists
    if os.path.exists(cached_path):
        print(f"Loading cached filtered data from {cached_path}...")
        with open(cached_path, "r") as f:
            filtered_data = json.load(f)
        print(f"Loaded {len(filtered_data)} questions from cache")
    else:
        # Load from local MetaMathQA dataset
        print(f"Loading MetaMathQA dataset from {local_path}...")
        with open(local_path, "r") as f:
            data = json.load(f)

        # Filter for ansaug_combined types (GSM_AnsAug and MATH_AnsAug)
        filtered_data = [d for d in data if d.get("type") in ["GSM_AnsAug", "MATH_AnsAug"]]

        print(f"Total questions after filtering for AnsAug types: {len(filtered_data)}")
        print(f"Breakdown: GSM_AnsAug={len([d for d in filtered_data if d['type']=='GSM_AnsAug'])}, MATH_AnsAug={len([d for d in filtered_data if d['type']=='MATH_AnsAug'])}")

        # Deduplicate based on "query" column
        seen_queries = set()
        deduplicated_data = []
        for d in filtered_data:
            query = d.get("query", "")
            if query not in seen_queries:
                seen_queries.add(query)
                deduplicated_data.append(d)

        print(f"Total questions after deduplication: {len(deduplicated_data)} (removed {len(filtered_data) - len(deduplicated_data)} duplicates)")
        filtered_data = deduplicated_data

        # Save filtered data to cache
        print(f"Saving filtered data to {cached_path}...")
        with open(cached_path, "w") as f:
            json.dump(filtered_data, f)
        print("Filtered data saved successfully")

    # Convert to Dataset
    dataset = Dataset.from_dict({k: [d[k] for d in filtered_data] for k in filtered_data[0].keys()})

    # Extract answer from response (the text after "The answer is:")
    def extract_answer(response):
        if "The answer is:" in response:
            return response.split("The answer is:")[-1].strip()
        elif "####" in response:
            # For GSM8K style answers
            parts = response.split("####")
            return parts[-1].strip()
        return response.strip()

    return dataset.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["query"]},
            ],
            "answer": extract_answer(x["response"]),
            "reasoning": "<reasoning>\n" + x["response"],
        }
    )


