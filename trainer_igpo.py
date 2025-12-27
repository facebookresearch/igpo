import contextlib
import functools
import json
import os
import pdb
import random
import shutil
import subprocess
import tempfile
import threading
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sized, Union
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from datasets import Dataset, IterableDataset
from packaging import version

from utils.generate import (
    generate_inpainting,
    generate_with_prefix_cache_inpaint,
)
from trl.extras.profiling import (
    profiling_context,
    profiling_decorator,
)
from trl.import_utils import (
    is_rich_available,
    is_vllm_available,
)
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.callbacks import (
    SyncRefModelCallback,
)
from trl.trainer.grpo_config import (
    GRPOConfig,
)
from trl.trainer.grpo_trainer import (
    GRPOTrainer,
)
from trl.trainer.utils import (
    print_prompt_completions_sample,
)

from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    is_wandb_available,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from transformers.utils import is_peft_available

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
import sys

import wandb
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

if is_peft_available():
    from peft import get_peft_model, PeftConfig

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class IGPOTrainer(GRPOTrainer):
    """
    Unified Group Relative Policy Optimization (grpo) Trainer for Diffusion Language Models.

    This class implements the grpo algorithm from the MMaDA paper, with key improvements:
    1. Unmasked questions/prompts during training (aligns with inference)
    2. Iteratively varied answer masking with structured timestep sampling
    3. Multi-step denoising exposure for better diffusion model utilization
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):

        self.has_uploaded_ckpt_steps = []
        self.has_uploaded_logs_steps = []
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "PEFT is required to use `peft_config`. Run `pip install peft`."
                )
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            # # Load model with quantization (matching working version)
            self.ref_model = MMadaModelLM.from_pretrained(
                args.pretrained_model_path,
                torch_dtype=torch.bfloat16,
            )
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left"
            )

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.use_vllm = args.use_vllm

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

        Trainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [
            n_gen
            for n_gen in range(2, global_batch_size + 1)
            if (global_batch_size) % n_gen == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, global_batch_size + 1)
                if (global_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                device_type = PartialState().default_device.type
                device_module = getattr(torch, device_type)
                if vllm_device == "auto":
                    if device_module.device_count() == 1:
                        vllm_device = f"{device_type}:0"  # particular case when training with onyl 1 device: share it
                    else:
                        vllm_device = f"{device_type}:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == f"{device_type}"
                    and int(vllm_device.split(":")[1]) >= device_module.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {device_module.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"{device_type}:{idx}"
                    for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )

                # For Ascend NPU (torch-npu), collective communication requires the establishment of a communication
                # group, and different processes must hold the same group number. However, multiple process groups will
                # be created internally within vLLM. This will cause the group id of the communication group on rank 0
                # to be different from that of other ranks, causing backward to hang on because the communication
                # domain cannot be established. So we need to patch it to make sure the group id of different ranks in
                # the training phase are the same.
                @contextlib.contextmanager
                def new_group_context():
                    new_group = torch.distributed.new_group
                    try:
                        torch.distributed.new_group = functools.partial(
                            new_group, use_local_synchronization=True
                        )
                        torch.npu.mem_get_info = functools.partial(
                            torch.npu.mem_get_info, device=vllm_device
                        )
                        yield
                    finally:
                        torch.distributed.new_group = new_group

                new_group_patch = (
                    new_group_context()
                    if device_type == "npu"
                    else contextlib.nullcontext()
                )
                with world_size_patch, profiling_patch, new_group_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                        max_model_len=self.args.vllm_max_model_len,
                    )

                # Guided decoding, if enabled
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(
                        backend="outlines", regex=args.vllm_guided_decoding_regex
                    )
                else:
                    guided_decoding = None

                # Sampling parameters
                self.sampling_params = SamplingParams(
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=-1 if args.top_k is None else args.top_k,
                    min_p=0.0 if args.min_p is None else args.min_p,
                    repetition_penalty=args.repetition_penalty,
                )

            self._last_loaded_step = (
                0  # tag to avoid useless loading during grad accumulation
            )

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

    def calculate_epoch(self):
        """Calculate the current epoch number based on processed prompts and dataset size"""
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            return 0.0

        # Get the total number of unique prompts processed across all GPUs
        # Each batch contains num_generations copies of each unique prompt
        # CORRECTED: Only count steps where new generations are created (global_step % num_iterations == 0)
        # In GRPO, new prompts are only generated every num_iterations steps
        # Each generation cycle processes gradient_accumulation_steps worth of data
        actual_generation_cycles = (self.state.global_step // self.num_iterations) + (
            1 if self.state.global_step > 0 else 0
        )

        total_unique_prompts_processed = (
            actual_generation_cycles
            * self.args.gradient_accumulation_steps
            * self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            // self.num_generations
        )

        # Get dataset length - if it's an iterable dataset, we might not know the exact length
        try:
            dataset_length = len(self.train_dataset)
        except (TypeError, AttributeError):
            # If we can't determine dataset length, return step number instead
            return float(self.state.global_step)

        # Calculate epoch as number of unique prompts processed divided by dataset length
        print(
            total_unique_prompts_processed,
            dataset_length,
            "total_unique_prompts_processed, dataset_length",
        )
        epoch_calculated = (
            total_unique_prompts_processed / dataset_length
            if dataset_length > 0
            else 0.0
        )
        return epoch_calculated

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Enhanced log method while preserving GRPO functionality.
        """
        # First, get the current mode to understand what we're logging
        mode = "eval" if self.control.should_evaluate else "train"

        # Get the averaged metrics from GRPO trainer (this is important!)
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        # Add eval prefix if in eval mode (matching GRPO trainer logic)
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        # Combine logs with metrics (matching GRPO trainer logic)
        combined_logs = {**logs, **metrics}

        # Call the parent's log method with the combined logs
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(combined_logs, start_time)
        else:  # transformers<=4.46
            super().log(combined_logs)

        # Clear metrics (matching GRPO trainer logic)
        self._metrics[mode].clear()

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The grpo Trainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        inpaint_masks = inputs["inpaint_masks"]

        timesteps = inputs["timesteps"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Get the current iteration index and corresponding timestep
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_timestep = timesteps[this_itr_idx]
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        token_mask = inputs["masked_token_mask"][this_itr_idx]

        input_ids = input_ids.unsqueeze(0)
        # Use grpo-specific method with timestep and inpaint masks
        per_token_logps, _, per_token_entropy = self._get_per_token_logps_grpo(
            model,
            input_ids,
            logits_to_keep,
            [this_itr_timestep],
            [this_itr_mask_seed],
            inpaint_masks,
        )
        per_token_logps = per_token_logps.squeeze(0)  # [B, T_keep]
        per_token_entropy = per_token_entropy.squeeze(0)  # [B, T_keep]

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss using grpo objective (same as GRPO but with grpo masking)
        advantages = inputs["advantages"]

        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )

        # Policy ratio computation
        if hasattr(self.args, "use_gspo") and self.args.use_gspo:
            # GSPO: Sequence-level importance ratio with length normalization
            # s_i = exp((log π_θ(y_i|x) - log π_θ_old(y_i|x)) / |y_i|)
            print("using gspo")
            sequence_logps = per_token_logps.sum(dim=1) / per_token_logps.size(1)  # [B]
            old_sequence_logps = old_per_token_logps.sum(
                dim=1
            ) / old_per_token_logps.size(1)  # [B]

            sequence_ratio = torch.exp(sequence_logps - old_sequence_logps)  # [B]

            # Expand sequence ratio to all tokens in each sequence
            policy_ratio = sequence_ratio.unsqueeze(1).expand(
                -1, per_token_logps.size(1)
            )  # [B, T]
        else:
            # Original GRPO: Token-level importance ratio
            policy_ratio = torch.exp(per_token_logps - old_per_token_logps)

        coef_1 = policy_ratio
        coef_2 = torch.clamp(
            coef_1, 1 - self.args.epsilon_low, 1 + self.args.epsilon_high
        )
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        denom = token_mask.sum()
        if denom == 0:
            return per_token_loss.new_tensor(0.0)

        masked_per_token_loss = per_token_loss * token_mask
        loss = masked_per_token_loss.sum() / denom

        mode = "eval" if self.control.should_evaluate else "train"

        # Standard metrics (always computed)
        if self.beta != 0.0:
            mean_kl = (per_token_kl * token_mask).sum() / denom
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        # Mean per-token entropy across the sequence
        mean_entropy = (per_token_entropy * token_mask).sum() / denom
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather_for_metrics(mean_entropy).mean().item()
        )

        # Policy ratio metrics
        mean_policy_ratio = (policy_ratio * token_mask).sum() / denom
        self._metrics[mode]["policy_ratio"].append(
            self.accelerator.gather_for_metrics(mean_policy_ratio).mean().item()
        )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * token_mask).sum() / denom
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        # Inpaint-related metrics (always computed for consistency)
        completion_inpaint_mask = inpaint_masks[:, -logits_to_keep:]  # [B, T_keep]

        # Separate clipping ratios for hint/inpainted vs generated tokens
        hint_clipped = is_clipped * token_mask * completion_inpaint_mask
        generated_clipped = is_clipped * token_mask * (~completion_inpaint_mask)
        hint_denom = (token_mask & completion_inpaint_mask).sum()
        generated_denom = (token_mask & (~completion_inpaint_mask)).sum()

        hint_tokens_clip_ratio = (
            hint_clipped.sum() / hint_denom
            if hint_denom > 0
            else torch.tensor(0.0, device=is_clipped.device)
        )
        generated_tokens_clip_ratio = (
            generated_clipped.sum() / generated_denom
            if generated_denom > 0
            else torch.tensor(0.0, device=is_clipped.device)
        )

        self._metrics[mode]["hint_tokens_clip_ratio"].append(
            self.accelerator.gather_for_metrics(hint_tokens_clip_ratio).mean().item()
        )
        self._metrics[mode]["generated_tokens_clip_ratio"].append(
            self.accelerator.gather_for_metrics(generated_tokens_clip_ratio).mean().item()
        )
        inpainted_masked_loss = masked_per_token_loss * completion_inpaint_mask
        generated_masked_loss = masked_per_token_loss * (~completion_inpaint_mask)
        inpainted_denom = (token_mask & completion_inpaint_mask).sum()
        generated_denom = (token_mask & (~completion_inpaint_mask)).sum()

        # Compute losses with proper handling of zero denominators
        inpainted_loss = (
            inpainted_masked_loss.sum() / inpainted_denom
            if inpainted_denom > 0
            else torch.tensor(0.0, device=masked_per_token_loss.device)
        )
        generated_loss = (
            generated_masked_loss.sum() / generated_denom
            if generated_denom > 0
            else torch.tensor(0.0, device=masked_per_token_loss.device)
        )

        # Enhanced metrics for advantages and log probabilities
        expanded_advantages = advantages.unsqueeze(1).expand(
            -1, logits_to_keep
        )  # [B, T_keep]
        inpainted_advantages = (
            expanded_advantages * token_mask * completion_inpaint_mask
        )
        generated_advantages = (
            expanded_advantages * token_mask * (~completion_inpaint_mask)
        )

        mean_inpainted_advantages = (
            inpainted_advantages.sum() / inpainted_denom
            if inpainted_denom > 0
            else torch.tensor(0.0, device=advantages.device)
        )
        mean_generated_advantages = (
            generated_advantages.sum() / generated_denom
            if generated_denom > 0
            else torch.tensor(0.0, device=advantages.device)
        )

        # Log probability calculations
        inpainted_logprobs = per_token_logps * token_mask * completion_inpaint_mask
        generated_logprobs = per_token_logps * token_mask * (~completion_inpaint_mask)

        mean_inpainted_logprobs = (
            inpainted_logprobs.sum() / inpainted_denom
            if inpainted_denom > 0
            else torch.tensor(0.0, device=per_token_logps.device)
        )
        mean_generated_logprobs = (
            generated_logprobs.sum() / generated_denom
            if generated_denom > 0
            else torch.tensor(0.0, device=per_token_logps.device)
        )

        num_inpainted_tokens = inpainted_denom.float()
        num_generated_tokens = generated_denom.float()

        # BLOCK 1: Policy ratio separation (R2 W3, R4 W1)
        hint_policy_ratios = policy_ratio * token_mask * completion_inpaint_mask
        generated_policy_ratios = policy_ratio * token_mask * (~completion_inpaint_mask)

        hint_policy_ratio_mean = (
            hint_policy_ratios.sum() / hint_denom
            if hint_denom > 0
            else torch.tensor(0.0, device=policy_ratio.device)
        )
        generated_policy_ratio_mean = (
            generated_policy_ratios.sum() / generated_denom
            if generated_denom > 0
            else torch.tensor(0.0, device=policy_ratio.device)
        )

        self._metrics[mode]["hint_tokens_policy_ratio_mean"].append(
            self.accelerator.gather_for_metrics(hint_policy_ratio_mean).mean().item()
        )
        self._metrics[mode]["generated_tokens_policy_ratio_mean"].append(
            self.accelerator.gather_for_metrics(generated_policy_ratio_mean).mean().item()
        )

        # Log ALL inpaint-related metrics unconditionally for consistency
        self._metrics[mode]["loss_inpainted"].append(
            self.accelerator.gather_for_metrics(inpainted_loss).mean().item()
        )
        self._metrics[mode]["loss_generated"].append(
            self.accelerator.gather_for_metrics(generated_loss).mean().item()
        )
        self._metrics[mode]["advantages_inpainted_mean"].append(
            self.accelerator.gather_for_metrics(mean_inpainted_advantages).mean().item()
        )
        self._metrics[mode]["advantages_generated_mean"].append(
            self.accelerator.gather_for_metrics(mean_generated_advantages).mean().item()
        )
        self._metrics[mode]["logprobs_inpainted_mean"].append(
            self.accelerator.gather_for_metrics(mean_inpainted_logprobs).mean().item()
        )
        self._metrics[mode]["logprobs_generated_mean"].append(
            self.accelerator.gather_for_metrics(mean_generated_logprobs).mean().item()
        )
        self._metrics[mode]["num_inpainted_tokens_in_loss"].append(
            self.accelerator.gather_for_metrics(num_inpainted_tokens).mean().item()
        )
        self._metrics[mode]["num_generated_tokens_in_loss"].append(
            self.accelerator.gather_for_metrics(num_generated_tokens).mean().item()
        )

        return loss

    def generate(
        self,
        model,
        prompt,
        reasoning_traces,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        inpaint_rollouts_per_group=0,
        inpaint_ratios=None,
    ):
        if inpaint_rollouts_per_group == 0:
            print("No inpaint. Running normal generations")
        else:
            print(f"Inpainting {inpaint_rollouts_per_group} / {len(prompt)}")
        result, inpaint_mask, first_input = generate_inpainting(
            question_tokens=prompt,
            tokenizer=self.processing_class,
            model=model,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            max_spacing=self.args.max_spacing,
            inpaint_rollouts_per_group=inpaint_rollouts_per_group,
            reference_answer=reasoning_traces,
            inpaint_ratio=inpaint_ratios,
            min_chunk_size=self.args.min_chunk_size,
            max_chunk_size=self.args.max_chunk_size,
        )

        if hasattr(self, "args") and hasattr(self.args, "output_dir"):
            try:
                # Create output directory if it doesn't exist
                output_dir = os.path.join(
                    self.args.output_dir, "logs", "logs_generations"
                )
                os.makedirs(output_dir, exist_ok=True)

                # Create filename with step/iteration info
                step_info = getattr(self, "_step", 0)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generation_step_{self.state.global_step}_{timestamp}_inpaint{inpaint_rollouts_per_group}.txt"
                filepath = os.path.join(output_dir, filename)

                input_sequence = result.clone()
                input_sequence[inpaint_mask] = (
                    mask_id  # Put mask tokens back in the positions that were generated
                )

                input_text = self.processing_class.batch_decode(
                    first_input, skip_special_tokens=False
                )
                prompt_length = prompt.shape[1]
                result_text = self.processing_class.batch_decode(
                    result[:, prompt_length:], skip_special_tokens=True
                )
                # Prepare data to save
                save_data = {
                    "step": step_info,
                    "timestamp": timestamp,
                    "generation_params": {
                        "steps": steps,
                        "gen_length": gen_length,
                        "block_length": block_length,
                        "temperature": temperature,
                        "cfg_scale": cfg_scale,
                        "remasking": remasking,
                        "mask_id": mask_id,
                    },
                    "input": input_text,
                    "reasoning_traces": reasoning_traces[0],
                    "generated_result": result_text,
                    "inpaint_params": {
                        "min_chunk_size": self.args.min_chunk_size,
                        "max_chunk_size": self.args.max_chunk_size,
                        "inpaint_ratio": self.args.inpaint_ratio,
                        "inpaint_ratio": inpaint_ratios,
                    },
                }

                # Also save as JSON for easier programmatic access
                if step_info % 4 == 0:
                    json_filepath = filepath.replace(".txt", ".json")
                    with open(json_filepath, "w", encoding="utf-8") as f:
                        json_data = save_data.copy()
                        if isinstance(json_data["input"], torch.Tensor):
                            json_data["input"] = input_text
                        if isinstance(json_data["generated_result"], torch.Tensor):
                            json_data["generated_result"] = result_text
                        json.dump(json_data, f, indent=2, ensure_ascii=False)

                    print(f"Generation results saved to: {filepath}")
                    print(f"JSON data saved to: {json_filepath}")

            except Exception as e:
                print(f"Warning: Failed to save generation results: {e}")

        return result, inpaint_mask

    def structured_timestep_sampling(self, num_iterations):
        """
        Implement structured timestep sampling as described in Algorithm 1.

        Returns:
            List of timesteps for each iteration, following the grpo strategy
        """
        t0 = torch.randint(30, self.args.diffusion_steps - 1, (1,)).item()

        if num_iterations == 1:
            return [t0]
        # Generate μ-1 uniformly spaced timesteps from [t0, T]
        if self.args.d1_masking:
            timesteps = [self.args.diffusion_steps]
        else:
            timesteps = [t0]
        for n in range(1, num_iterations):
            # Uniformly divide remaining timesteps: tn = ⌊(n-1)/(μ-1) * (T - t1) + t1⌋
            tn = int(n / (num_iterations - 1) * (self.args.diffusion_steps - t0) + t0)
            if self.args.d1_masking:
                timesteps.append(self.args.diffusion_steps)
            else:
                timesteps.append(tn)
        print("log prob estimate timesteps:", timesteps)
        return timesteps

    def timestep_to_mask_ratio(self, timestep):
        """
        Convert timestep to mask ratio.
        Higher timesteps = more noise = higher mask ratio
        """
        return timestep / self.args.diffusion_steps

    def grpo_forward_process(
        self, batch, prompt_index, mask_id, timestep, seed=None
    ):
        if seed is not None:
            set_seed(seed)
        """
        grpo forward process with structured noising strategy that respects inpaint masks.
        """
        b, l = batch.shape
        device = batch.device

        # Convert timestep to mask ratio - this gives us the base mask ratio
        base_mask_ratio = self.timestep_to_mask_ratio(timestep)

        # Create per-sample mask ratios (initialized with base ratio)
        mask_ratios = torch.full((b,), base_mask_ratio, device=device)

        can_be_masked = ~prompt_index.unsqueeze(0).expand(b, -1)

        # For maskable completion tokens, randomly select which ones to mask based on per-sample mask_ratio
        random_matrix = torch.rand((b, l), device=device)

        # Create per-position mask ratio tensor
        mask_ratio_expanded = mask_ratios.unsqueeze(1).expand(b, l)  # [b, l]

        should_mask = can_be_masked & (random_matrix < mask_ratio_expanded)

        # Create noisy batch - only mask the selected positions
        noisy_batch = torch.where(should_mask, mask_id, batch)

        # Build p_mask: probability that each token is masked (per-sample)
        p_mask = torch.where(
            can_be_masked,
            mask_ratio_expanded,  # Use per-sample mask ratios
            torch.zeros(1, device=device),  # non-maskable tokens: never masked
        )

        return noisy_batch, p_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def _get_per_token_logps_grpo(
        self, model, input_ids, logits_to_keep, timesteps, mask_seeds, inpaint_masks
    ):
        """
        Calculate per-token log probabilities using grpo strategy with inpainting.

        Args:
            model: The diffusion model
            input_ids: [num_iterations, batch_size, seq_len]
            logits_to_keep: Number of completion tokens
            timesteps: List of timesteps for each iteration
            mask_seeds: List of mask seeds for each iteration
            inpaint_masks: [batch_size, seq_len] - True for inpainted tokens that should not be masked

        Returns:
            per_token_logps: [num_iterations, batch_size, logits_to_keep]
            masked_pos: [num_iterations, batch_size, logits_to_keep] - positions that were masked
            per_token_entropy: [num_iterations, batch_size, logits_to_keep] - entropy for each position
        """

        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(
            num_iterations, batch_size, logits_to_keep, device=device
        )
        per_token_entropy = torch.zeros(
            num_iterations, batch_size, logits_to_keep, device=device
        )

        # Verify mask_seeds length
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # Process each iteration with its corresponding timestep
        all_perturbed_seqs = []
        all_expanded_inputs = []

        for iter_idx, timestep in enumerate(timesteps):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            mask_seed = mask_seeds[iter_idx]
            perturbed_seq, _ = self.grpo_forward_process(
                expanded_input,
                prompt_index,
                self.args.mask_id,
                timestep,
                seed=mask_seed,
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(
            all_perturbed_seqs, dim=0
        )  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(
            all_expanded_inputs, dim=0
        )  # [num_iterations * batch_size, seq_len]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, 0.0, self.args.mask_id
        )  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]

        # Calculate entropy for completion tokens
        completion_probs = F.softmax(
            completion_logits, dim=-1
        )  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_entropy = -torch.sum(
            completion_probs * torch.log(completion_probs + 1e-8), dim=-1
        )  # [num_iterations * batch_size, logits_to_keep]

        mask_mat = perturbed_seq[:, -logits_to_keep:] == self.args.mask_id
        masked_pos = mask_mat.view(num_iterations, batch_size, logits_to_keep).to(
            torch.uint8
        )

        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(
            num_iterations, batch_size, logits_to_keep
        )

        # Reshape entropy to match the output format
        per_token_entropy = completion_entropy.view(
            num_iterations, batch_size, logits_to_keep
        )

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()

        return (
            per_token_logps.to(torch.float32),
            masked_pos,
            per_token_entropy.to(torch.float32),
        )

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs for grpo training."""
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[
                    self._step % self.args.gradient_accumulation_steps
                ] = inputs
            else:
                inputs = self._buffered_inputs[
                    self._step % self.args.gradient_accumulation_steps
                ]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_batch(
        self,
        unwrapped_model,
        prompt_ids,
        reasoning_traces,
        steps,
        gen_length,
        block_length,
        temperature,
        cfg_scale,
        inpaint_ratios=None,
        inpaint_rollouts_per_group=0,
    ):
        """Generate completions for a batch of prompts."""
        generation_batch_size = self.args.generation_batch_size
        prompt_completion_ids_all = []
        inpaint_masks_all = []

        for i in range(0, prompt_ids.size(0), generation_batch_size):
            end_idx = min(i + generation_batch_size, prompt_ids.size(0))
            batch_prompt_ids = prompt_ids[i:end_idx]
            batch_reasoning_traces = reasoning_traces[i:end_idx]

            # Extract corresponding inpaint ratios for this batch slice
            batch_inpaint_ratios = None
            if inpaint_ratios is not None:
                batch_inpaint_ratios = inpaint_ratios[i:end_idx]

            batch_prompt_completion_ids, batch_inpaint_masks = self.generate(
                model=unwrapped_model,
                prompt=batch_prompt_ids,
                reasoning_traces=batch_reasoning_traces,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
                inpaint_rollouts_per_group=min(
                    inpaint_rollouts_per_group, end_idx - i
                ),  # Adjust for batch size
                inpaint_ratios=batch_inpaint_ratios,
            )

            prompt_completion_ids_all.append(batch_prompt_completion_ids)
            inpaint_masks_all.append(batch_inpaint_masks)

            del batch_prompt_ids, batch_prompt_completion_ids
            torch.cuda.empty_cache()

        return torch.cat(prompt_completion_ids_all, dim=0), torch.cat(
            inpaint_masks_all, dim=0
        )

    def _evaluate_completions(
        self, inputs, prompts, prompt_completion_ids, prompt_length
    ):
        """Evaluate completions and return rewards and correctness analysis."""
        device = self.accelerator.device

        # Extract and decode completions
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Prepare completions for reward function
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        # Compute rewards
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for reward_idx, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {
                key: [inputs[j][key] for j in range(len(inputs))] for key in keys
            }

            output_reward = reward_func(
                prompts=prompts,
                completions=completions,
                step=self._step,
                run_name=self.args.output_dir,
                **reward_kwargs,
            )
            rewards_per_func[:, reward_idx] = torch.tensor(
                output_reward, dtype=torch.float32, device=device
            )

        # Analyze correctness
        correctness_reward_idx = 1  # Adjust based on your reward function setup
        correctness_scores = rewards_per_func[:, correctness_reward_idx]

        all_correct = (correctness_scores == 3.0).all().item()
        all_wrong = (correctness_scores == 0.0).all().item()

        return rewards_per_func, correctness_scores, all_correct, all_wrong

    def _print_analysis(self, correctness_scores, all_correct, all_wrong, prefix=""):
        """Print correctness analysis results."""
        print(f"\n=== {prefix}ANALYSIS ===")
        print(f"Batch size: {len(correctness_scores)}")
        print(f"Correctness scores: {correctness_scores.tolist()}")

        if all_correct:
            print(f"🟢 {prefix}ALL CORRECT!")
        elif all_wrong:
            print(f"🔴 {prefix}ALL WRONG!")
        else:
            print(f"🟡 {prefix}MIXED")
            correct_count = (correctness_scores == 3.0).sum().item()
            wrong_count = (correctness_scores == 0.0).sum().item()
            other_count = len(correctness_scores) - correct_count - wrong_count
            print(
                f"   Correct: {correct_count}, Wrong: {wrong_count}, Other: {other_count}"
            )

    def _calculate_completion_lengths(self, prompt_completion_ids, prompt_length):
        """Calculate the actual completion lengths (excluding EOS and padding)."""
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Find EOS positions
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_positions = torch.full(
            (completion_ids.size(0),),
            completion_ids.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )

        # For sequences that have EOS, find the first EOS position
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            eos_positions[has_eos] = is_eos[has_eos].int().argmax(dim=1)

        return eos_positions

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        self.model.eval()
        prompts = [x["prompt"] for x in inputs]
        reasoning_traces = [
            x["reasoning"] for x in inputs
        ]  # a batch of same prompt should have same reasonings

        prompts_text = []
        for prompt in prompts:
            formatted_prompt = self.processing_class.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
            prompts_text.append(formatted_prompt)

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=False,
            padding_side="left",
            add_special_tokens=True,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )
        prompt_lengths = prompt_mask.sum(dim=1)  # [B]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = int(gen_length // 2)

        gen_config = {
            "gen_length": gen_length,
            "block_length": block_length,
            "steps": steps,
            "temperature": self.args.temperature or 0.0,
            "cfg_scale": self.args.cfg_scale,
        }
        print(gen_config, "generation configs")

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator
        ) as unwrapped_model:
            # Initial generation (no inpainting)
            prompt_completion_ids, inpaint_masks = self._generate_batch(
                unwrapped_model, prompt_ids, reasoning_traces, **gen_config
            )

            # Evaluate initial generation
            rewards_per_func, correctness_scores, all_correct, all_wrong = (
                self._evaluate_completions(
                    inputs, prompts, prompt_completion_ids, prompt_ids.size(1)
                )
            )

            # Monitor completion lengths for initial generation
            initial_completion_lengths = self._calculate_completion_lengths(
                prompt_completion_ids, prompt_ids.size(1)
            )
            print(
                f"📊 INITIAL GENERATION - Completion lengths: {initial_completion_lengths.tolist()}"
            )

            self._print_analysis(correctness_scores, all_correct, all_wrong)

            # Log initial generation correctness
            initial_correct_count = (correctness_scores == 3.0).sum().item()
            initial_total_count = len(correctness_scores)
            initial_correctness_ratio = (
                initial_correct_count / initial_total_count
                if initial_total_count > 0
                else 0.0
            )

            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["initial_correctness_ratio"].append(initial_correctness_ratio)
            self._metrics[mode]["initial_total_count"].append(initial_total_count)

            replaced_indices = set()  # Track which indices were replaced
            # Only execute the strategy if we passed the probability check


            current_chunk_high_ratio = self.args.inpaint_chunk_high_initial_ratio
            if (
                all_wrong
            ):
                print("Entering all wrong inpaint fix")
                replaced_indices = set()
                total_generations = prompt_ids.size(0)
                generations_per_group = self.num_generations
                num_groups = total_generations // generations_per_group
                max_replacements = (
                    self.args.max_replacements
                )  # Maximum replacements per group
                max_resamples = 0  # Maximum resampling attempts

                print(
                    f"Processing {num_groups} groups with {generations_per_group} generations each"
                )
                print(
                    f"Max replacements per group: {max_replacements}, Max resamples: {max_resamples}"
                )

                final_prompt_completion_ids = prompt_completion_ids.clone()
                final_inpaint_masks = inpaint_masks.clone()

                for resample_attempt in range(max_resamples + 1):  # 0, 1, 2
                    print(
                        f"\n--- RESAMPLE ATTEMPT {resample_attempt + 1}/{max_resamples + 1} ---"
                    )

                    # Create inpaint ratios for all samples
                    all_inpaint_ratios = []
                    for group_idx in range(num_groups):
                        mode = "eval" if self.control.should_evaluate else "train"
                        # Generate uniformly sampled ratios [low, high] for this group
                        self._metrics[mode]["current_inpaint_chunk_ratio_high"].append(
                            current_chunk_high_ratio
                        )
                        self._metrics[mode]["current_inpaint_chunk_ratio_low"].append(
                            self.args.inpaint_chunk_ratio_low
                        )
                        group_ratios = (
                            torch.rand(generations_per_group)
                            * (
                                current_chunk_high_ratio
                                - self.args.inpaint_chunk_ratio_low
                            )
                            + self.args.inpaint_chunk_ratio_low
                        )
                        group_ratios = group_ratios.sort()[
                            0
                        ]  # Sort for increasing order
                        all_inpaint_ratios.extend(group_ratios.tolist())
                        print(
                            f"Group {group_idx} inpaint ratios: {group_ratios.tolist()}"
                        )

                    # BLOCK 5: Log mean inpaint ratio used (Transparency)
                    mean_inpaint_ratio = sum(all_inpaint_ratios) / len(all_inpaint_ratios)
                    self._metrics[mode][f"mean_inpaint_ratio_used_attempt_{resample_attempt + 1}"].append(
                        mean_inpaint_ratio
                    )

                    # Generate inpainted completions
                    inpainted_prompt_completion_ids, inpainted_inpaint_masks = (
                        self._generate_batch(
                            unwrapped_model,
                            prompt_ids,
                            reasoning_traces,
                            inpaint_rollouts_per_group=total_generations,
                            inpaint_ratios=all_inpaint_ratios,
                            **gen_config,
                        )
                    )

                    # Evaluate inpainted completions
                    inpainted_rewards_per_func, inpainted_correctness_scores, _, _ = (
                        self._evaluate_completions(
                            inputs,
                            prompts,
                            inpainted_prompt_completion_ids,
                            prompt_ids.size(1),
                        )
                    )

                    self._print_analysis(
                        inpainted_correctness_scores,
                        False,
                        False,
                        f"INPAINTED (ATTEMPT {resample_attempt + 1}) ",
                    )

                    # Calculate inpainted generations' correctness ratio
                    inpainted_correct_count = (
                        (inpainted_correctness_scores == 3.0).sum().item()
                    )
                    inpainted_total_count = len(inpainted_correctness_scores)
                    inpainted_correctness_ratio = (
                        inpainted_correct_count / inpainted_total_count
                        if inpainted_total_count > 0
                        else 0.0
                    )

                    print(
                        f"📊 INPAINTED CORRECTNESS RATIO: {inpainted_correctness_ratio:.3f} ({inpainted_correct_count}/{inpainted_total_count})"
                    )

                    # Log inpainted correctness metrics
                    mode = "eval" if self.control.should_evaluate else "train"
                    self._metrics[mode][f"inpainted_correctness_ratio_attempt_{resample_attempt + 1}"].append(inpainted_correctness_ratio)
                    self._metrics[mode][f"inpainted_correct_count_attempt_{resample_attempt + 1}"].append(inpainted_correct_count)
                    self._metrics[mode][f"inpainted_total_count_attempt_{resample_attempt + 1}"].append(inpainted_total_count)

                    # Check if we have ANY correct samples across all groups
                    total_correct_available = (
                        (inpainted_correctness_scores == 3.0).sum().item()
                    )

                    if total_correct_available == 0:
                        print(
                            f"⚠️  NO CORRECT SAMPLES FOUND in attempt {resample_attempt + 1}. Available: {total_correct_available}"
                        )

                        # Only continue resampling if we haven't reached max attempts
                        if resample_attempt < max_resamples:
                            print(
                                "🔄 Will resample since no correct samples are available..."
                            )
                            continue
                        else:
                            print(
                                "🛑 Reached maximum resampling attempts. Stopping with no replacements."
                            )
                            break
                    else:
                        print(
                            f"✅ Found {total_correct_available} correct samples. Proceeding with replacement..."
                        )

                    # Process each group separately for selective replacement
                    replacement_count = 0

                    for group_idx in range(num_groups):
                        group_start = group_idx * generations_per_group
                        group_end = group_start + generations_per_group

                        # Get correctness for this group
                        group_inpainted_correctness = inpainted_correctness_scores[
                            group_start:group_end
                        ]
                        correct_mask = group_inpainted_correctness == 3.0
                        correct_indices = correct_mask.nonzero(as_tuple=True)[0]

                        num_correct_in_group = len(correct_indices)

                        if num_correct_in_group > 0:
                            # Determine how many to replace (min of available correct and max_replacements)
                            num_to_replace = min(num_correct_in_group, max_replacements)

                            # Randomly select which correct ones to use if we have more than needed
                            if num_correct_in_group > num_to_replace:
                                selected_indices = torch.randperm(num_correct_in_group)[
                                    :num_to_replace
                                ]
                                correct_indices_to_use = correct_indices[
                                    selected_indices
                                ]
                            else:
                                correct_indices_to_use = correct_indices

                            # Replace the first num_to_replace samples in this group
                            for i, local_correct_idx in enumerate(
                                correct_indices_to_use
                            ):
                                global_correct_idx = group_start + local_correct_idx
                                global_replace_idx = (
                                    group_start + i
                                )  # Replace first i samples in group

                                # Perform the replacement
                                final_prompt_completion_ids[global_replace_idx] = (
                                    inpainted_prompt_completion_ids[global_correct_idx]
                                )
                                final_inpaint_masks[global_replace_idx] = (
                                    inpainted_inpaint_masks[global_correct_idx]
                                )
                                replaced_indices.add(
                                    global_replace_idx
                                )  # Track which indices were replaced
                                replacement_count += 1

                                print(
                                    f"✅ Group {group_idx}: Replaced sample {i} with correct inpainted sample {local_correct_idx.item()}"
                                )
                        else:
                            print(
                                f"❌ Group {group_idx}: No correct samples available for replacement"
                            )

                    print(f"📈 RESAMPLE ATTEMPT {resample_attempt + 1} SUMMARY:")
                    print(f"   Total available correct: {total_correct_available}")
                    print(f"   Total replacements made: {replacement_count}")
                    print(
                        f"🎯 Stopping resampling. Found correct samples and made {replacement_count} replacements."
                    )

                    # Log number of prompts replaced by inpaintings
                    mode = "eval" if self.control.should_evaluate else "train"
                    self._metrics[mode][f"num_prompts_replaced_by_inpainting_attempt_{resample_attempt + 1}"].append(replacement_count)
                    self._metrics[mode][f"total_prompts_in_batch_attempt_{resample_attempt + 1}"].append(total_generations)
                    self._metrics[mode][f"replacement_ratio_attempt_{resample_attempt + 1}"].append(
                        replacement_count / total_generations if total_generations > 0 else 0.0
                    )

                    break

                # Final evaluation of the mixed results
                (
                    final_rewards_per_func,
                    final_correctness_scores,
                    final_all_correct,
                    final_all_wrong,
                ) = self._evaluate_completions(
                    inputs, prompts, final_prompt_completion_ids, prompt_ids.size(1)
                )

                self._print_analysis(
                    final_correctness_scores,
                    final_all_correct,
                    final_all_wrong,
                    "FINAL MIXED ",
                )

                # Log whether the all-wrong case was rescued (non-all-wrong after resampling)
                mode = "eval" if self.control.should_evaluate else "train"
                all_wrong_rescued = 1 if not final_all_wrong else 0
                self._metrics[mode]["all_wrong_rescued"].append(all_wrong_rescued)

                if all_wrong_rescued:
                    print("🎉 SUCCESS: All-wrong batch was RESCUED (now has mixed/correct answers)")
                else:
                    print("⚠️  FAILED: Batch is still all-wrong after resampling")

                # BLOCK 3: Final correctness metrics (Show IGPO improvement)
                final_correct_count = (final_correctness_scores == 3.0).sum().item()
                final_total_count = len(final_correctness_scores)
                final_correctness_ratio = (
                    final_correct_count / final_total_count
                    if final_total_count > 0
                    else 0.0
                )

                self._metrics[mode]["final_correctness_ratio"].append(final_correctness_ratio)
                self._metrics[mode]["correctness_improvement"].append(
                    final_correctness_ratio - initial_correctness_ratio
                )


                # Use the final mixed results
                prompt_completion_ids = final_prompt_completion_ids
                inpaint_masks = final_inpaint_masks
                rewards_per_func = final_rewards_per_func
                correctness_scores = final_correctness_scores

                print("✅ SELECTIVE REPLACEMENT STRATEGY COMPLETED!")
        # Extract prompt and completion parts
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        logits_to_keep = completion_ids.size(1)

        if self.args.random_masking:
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(
                0, 2**12, (self.num_iterations,), device=device
            ).tolist()
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        # grpo: Use structured timestep sampling
        timesteps = self.structured_timestep_sampling(self.num_iterations)

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        all_per_token_entropy = []

        with torch.no_grad():
            if self.num_iterations > 1:
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(
                    0
                ).expand(self.num_iterations, -1, -1)
                old_per_token_logps, masked_token_mask, per_token_entropy = (
                    self._get_per_token_logps_grpo(
                        self.model,
                        prompt_completion_ids_expanded,
                        logits_to_keep,
                        timesteps,
                        mask_seeds,
                        inpaint_masks,
                    )
                )
                all_old_per_token_logps = old_per_token_logps
                all_per_token_entropy = per_token_entropy
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(
                    0
                ).expand(self.num_iterations, -1, -1)
                if is_peft_model(self.model):
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, _ = self._get_per_token_logps_grpo(
                            self.model,
                            prompt_completion_ids_expanded,
                            logits_to_keep,
                            timesteps,
                            mask_seeds,
                            inpaint_masks,
                        )
                else:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_grpo(
                            self.ref_model,
                            prompt_completion_ids_expanded,
                            logits_to_keep,
                            timesteps,
                            mask_seeds,
                            inpaint_masks,
                        )
                all_ref_per_token_logps = ref_per_token_logps

        # Apply entropy-based filtering for inpainted tokens
        completion_inpaint_mask = inpaint_masks[
            :, -logits_to_keep:
        ]  # [batch_size, logits_to_keep]

        entropy_keep_ratio = getattr(self.args, "entropy_clipping_ratio_inpaint", 0.2)

        # Metrics BEFORE filtering
        inpaint_tokens_before = (masked_token_mask.bool() & completion_inpaint_mask.unsqueeze(0)).sum().item()
        generated_tokens_before = (masked_token_mask.bool() & (~completion_inpaint_mask.unsqueeze(0))).sum().item()

        filtered_masked_token_mask = (
            masked_token_mask.clone()
        )  # [num_iterations, batch_size, logits_to_keep]

        for iter_idx in range(self.num_iterations):
            iter_entropy = all_per_token_entropy[
                iter_idx
            ]  # [batch_size, logits_to_keep]
            iter_masked_pos = masked_token_mask[
                iter_idx
            ]  # [batch_size, logits_to_keep]
            for batch_idx in range(completion_inpaint_mask.shape[0]):
                # Get positions that are both masked (by grpo) and inpainted (from prefix)
                masked_and_inpainted = (
                    iter_masked_pos[batch_idx].bool()
                    & completion_inpaint_mask[batch_idx]
                )

                if masked_and_inpainted.any():
                    # Get entropy values for these positions
                    candidate_entropy = iter_entropy[batch_idx][masked_and_inpainted]

                    # Calculate top k% threshold
                    num_candidates = masked_and_inpainted.sum().item()
                    top_k = max(
                        1, int(num_candidates * entropy_keep_ratio)
                    )  # At least 1 token

                    # Get top-k entropy values and their indices
                    _, top_indices_relative = torch.topk(
                        candidate_entropy, top_k, largest=True
                    )

                    # Convert relative indices to absolute indices
                    candidate_absolute_indices = masked_and_inpainted.nonzero(
                        as_tuple=True
                    )[0]
                    top_indices_absolute = candidate_absolute_indices[
                        top_indices_relative
                    ]

                    # Create mask for positions to keep (top k% entropy)
                    keep_mask = torch.zeros_like(
                        completion_inpaint_mask[batch_idx], dtype=torch.bool
                    )
                    keep_mask[top_indices_absolute] = True

                    # Update masked_token_mask: keep non-inpainted masked tokens, and only top k% of inpainted masked tokens
                    is_inpainted = completion_inpaint_mask[batch_idx]
                    filtered_masked_token_mask[iter_idx, batch_idx] = (
                        iter_masked_pos[batch_idx].bool() & (~is_inpainted | keep_mask)
                    ).to(torch.uint8)


        # Decode completions and compute rewards (same as original)
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {
                    key: [example[key] for example in inputs] for key in keys
                }
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan
                    for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Total gathered rewards divided by num_generations gives total unique prompts across all GPUs
        total_unique_prompts = rewards.shape[0] // self.num_generations

        # Compute grouped-wise rewards and advantages
        mean_grouped_rewards = rewards.view(
            total_unique_prompts, self.num_generations
        ).mean(dim=1)
        std_grouped_rewards = rewards.view(
            total_unique_prompts, self.num_generations
        ).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards

        # FIXED: Enhanced zero std analysis with proper multi-GPU handling
        grouped_rewards = rewards.view(
            total_unique_prompts, self.num_generations
        )  # [total_unique_prompts, num_generations]
        grouped_std = grouped_rewards.std(dim=1)  # [total_unique_prompts]
        grouped_mean = grouped_rewards.mean(dim=1)  # [total_unique_prompts]

        # Count prompts with zero std deviation (based on overall reward)
        zero_std_mask = grouped_std < 1e-6
        zero_std_count = zero_std_mask.sum().item()
        total_prompts = grouped_std.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        # Separate analysis for all_correct and all_wrong (based only on correctness dimension)
        correctness_reward_idx = 1
        if correctness_reward_idx is not None:
            grouped_rewards_per_func = rewards_per_func.view(
                total_unique_prompts,
                self.num_generations,
                rewards_per_func.shape[1],
            )
            # Get correctness scores for ALL prompts (not just zero_std ones)
            correctness_scores = grouped_rewards_per_func[
                :, :, correctness_reward_idx
            ]  # [total_unique_prompts, num_generations]

            if correctness_scores.numel() > 0:
                global_max_score = 3
                global_min_score = 0
                # Check if all generations for each prompt have the same correctness score
                all_correct_mask = (correctness_scores == global_max_score).all(dim=1)
                all_wrong_mask = (correctness_scores == global_min_score).all(dim=1)
                all_correct_count = all_correct_mask.sum().item()
                all_wrong_count = all_wrong_mask.sum().item()
            else:
                all_correct_count = 0
                all_wrong_count = 0
        else:
            all_correct_count = 0
            all_wrong_count = 0
            if self.accelerator.is_main_process:
                print(
                    "Warning: Could not identify correctness reward function for all_correct/all_wrong analysis"
                )

        all_correct_ratio = (
            all_correct_count / total_prompts if total_prompts > 0 else 0.0
        )
        all_wrong_ratio = all_wrong_count / total_prompts if total_prompts > 0 else 0.0
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Add new metrics for all correct/wrong analysis
        self._metrics[mode]["all_correct_ratio"].append(all_correct_ratio)
        self._metrics[mode]["all_wrong_ratio"].append(all_wrong_ratio)

        # BLOCK 2: Group-level raw counts (R4 W2, R4 Q2)
        self._metrics[mode]["all_wrong_groups_count"].append(all_wrong_count)
        self._metrics[mode]["all_correct_groups_count"].append(all_correct_count)
        self._metrics[mode]["mixed_groups_count"].append(
            total_prompts - all_correct_count - all_wrong_count
        )
        self._metrics[mode]["total_groups_count"].append(total_prompts)

        # Log inpainted token statistics
        total_inpainted_masked = (
            (masked_token_mask.bool() & completion_inpaint_mask.unsqueeze(0))
            .sum()
            .item()
        )
        total_inpainted_kept = (
            (filtered_masked_token_mask.bool() & completion_inpaint_mask.unsqueeze(0))
            .sum()
            .item()
        )

        if total_inpainted_masked > 0:
            inpaint_keep_ratio = total_inpainted_kept / total_inpainted_masked
            self._metrics[mode]["inpaint_keep_ratio"].append(inpaint_keep_ratio)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})
        self.model.train()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "inpaint_masks": inpaint_masks,
            "masked_token_mask": filtered_masked_token_mask,  # Use filtered mask
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "timesteps": timesteps,
            "mask_seeds": mask_seeds,
        }
