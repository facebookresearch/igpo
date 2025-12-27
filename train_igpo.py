import json
import os
import sys
from pathlib import Path

import pkg_resources
import torch
import wandb
from accelerate import Accelerator

from utils.data_utils import get_metamath_questions, set_random_seed
from utils.diffu_grpo_config import DiffuGRPOConfig
from trainer_igpo import IGPOTrainer


from trl import ModelConfig, TrlParser
from peft import LoraConfig
from utils.configuration_llada import ActivationCheckpointingStrategy
from utils.reward_func import (
    correctness_reward_func,
    xmlcount_reward_func,
)
from transformers import AutoModel, AutoTokenizer



def create_deepspeed_config(args):
    """Create DeepSpeed configuration for torchrun"""
    # Calculate total batch size for DeepSpeed
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    train_batch_size = (
        args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    )

    config = {
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "contiguous_gradients": True,
            "zero3_init_flag": False,
            "zero3_save_16bit_model": False,
        },
        "checkpoint": {
            "use_node_local_storage": True,
        },
    }

    # Save config to file
    config_path = os.path.join(args.output_dir, "ds_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    # print(config)
    print(f"✅ Created DeepSpeed config at: {config_path}")
    print(f"   📊 World size: {world_size}")
    print(f"   📊 Train batch size: {train_batch_size}")
    print(f"   📊 Micro batch size per GPU: {args.per_device_train_batch_size}")

    return config_path


def setup_models(config):
    """Setup models without explicit device placement - let accelerate handle it"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use local model path directly
    model_path = config.model_path

    # Use the local model path
    directory_to_check = model_path
    print(f"Loading model from: {directory_to_check}")

    tokenizer = AutoTokenizer.from_pretrained(
        directory_to_check, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        directory_to_check, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.model.set_activation_checkpointing(
        ActivationCheckpointingStrategy.whole_layer
    )
    print('Activation checkpointing enabled; strategy set to "whole_layer"')
    print(f"Model initialized with context length: {config.max_seq_length}")
    print(f"Initial batch size: {config.per_device_train_batch_size}")
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.per_device_train_batch_size
    model.config.gradient_checkpointing = True
    model.config.use_cache = False
    return tokenizer, model


# Updated main function
def main(grpo_config, model_config):
    ds_config_path = create_deepspeed_config(grpo_config)
    grpo_config.deepspeed = ds_config_path  # Enable DeepSpeed
    grpo_config.bf16 = True  # Enable bf16 to match DeepSpeed config

    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"  # ← Critical for bf16 support
    os.environ["ACCELERATE_CONFIG_DS_FIELDS"] = "true"  # ← Enable DS config processing
    os.environ["ACCELERATE_DEEPSPEED_ZERO_STAGE"] = "2"  # ← Your ZeRO stage
    os.environ["OMP_NUM_THREADS"] = "1"  # ← Prevent CPU thread oversubscription

    print("=== AFTER DEEPSPEED SETUP ===")
    print(f"After: grpo_config.deepspeed = {grpo_config.deepspeed}")
    print(f"DeepSpeed config path: {ds_config_path}")
    print(
        f"ACCELERATE_USE_DEEPSPEED = {os.environ.get('ACCELERATE_USE_DEEPSPEED', 'NOT SET')}"
    )
    print(
        f"ACCELERATE_MIXED_PRECISION = {os.environ.get('ACCELERATE_MIXED_PRECISION', 'NOT SET')}"
    )
    print(
        f"ACCELERATE_DEEPSPEED_ZERO_STAGE = {os.environ.get('ACCELERATE_DEEPSPEED_ZERO_STAGE', 'NOT SET')}"
    )

    print("Initializing IGPOTrainer with deepspeed passed in json...")
    set_random_seed(grpo_config.seed)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity="zsyucla",
            project="igpo-rebuttal",
            name=grpo_config.run_name,
        )
    grpo_config.report_to = "wandb"
    grpo_config.save_16bit_model = (True,)  # This helps with ZeRO states
    tokenizer, model = setup_models(grpo_config)
    dataset = get_metamath_questions()
    reward_functions = [
        xmlcount_reward_func,
        correctness_reward_func,
    ]
    dataset = dataset.shuffle(seed=grpo_config.seed)
    train_set = dataset
    print("Models setup completed successfully")
    print(f"Model dtype: {model.dtype}")
    print("Initializing IGPOTrainer...")
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    trainer = IGPOTrainer(
        args=grpo_config,
        peft_config=peft_config,
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )

    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
