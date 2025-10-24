# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class DiffuGRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation

        temperature (`float`, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*, defaults to `50`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled.
        min_p (`float` or `None`, *optional*, defaults to `None`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str` or `None`, *optional*, defaults to `None`):
            Implementation of the cache method for faster generation when use_vllm is set to False.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training. This assumes that
            training has not already occupied all available GPUs. If only one device is available, the device will be
            shared between both training and vLLM.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        vllm_enable_prefix_caching (`bool`, *optional*, defaults to `True`):
            Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and the hardware
            support this feature.
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving training
            speed, but may be numerically unstable for long training runs.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )

    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )

    # Parameters that control generation
    temperature: float = field(
        default=1.2,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Implementation of the cache method for faster generation when use_vllm is set to False."
        },
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True` (default), ensure that the model and "
            "the hardware support this feature."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={
            "help": "Number of iterations per batch (denoted as μ in the algorithm)."
        },
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log the completions during training."},
    )

    generation_batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": "Batch size for generation. If not set, the batch size will be equal to the number of generations."
        },
    )

    block_length: Optional[int] = field(
        default=64,
        metadata={"help": "diffusion block length"},
    )
    diffusion_steps: Optional[int] = field(
        default=64,
    )
    cfg_scale: Optional[float] = field(
        default=0.0,
    )
    remasking: Optional["str"] = field(
        default="low_confidence",
    )
    dataset: Optional[str] = field(
        default="gsm8k",
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    p_mask_prompt: float = field(
        default=0.3,
        metadata={"help": "Probability of masking the prompt."},
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default is from Llada"},
    )
    random_masking: bool = field(
        default=True,
        metadata={"help": "Whether to randomly mask tokens."},
    )
    # MMaDA-specific parameters
    encoder_type: Optional[str] = field(
        default="magvitv2",
        metadata={"help": "VQ model type (e.g., 'magvitv2', 'vq16')"},
    )
    vq_model_name: Optional[str] = field(
        default="showlab/magvitv2",
        metadata={"help": "Pre-trained VQ model name or path"},
    )
    tokenizer_path: Optional[str] = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
        metadata={"help": "Path to the tokenizer"},
    )
    pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-trained MMaDA model"},
    )
    w_clip_vit: bool = field(
        default=False,
        metadata={"help": "Whether to use CLIP ViT"},
    )
    new_vocab_size: int = field(
        default=134656,
        metadata={"help": "New vocabulary size after adding special tokens"},
    )
    llm_vocab_size: int = field(
        default=126464,
        metadata={"help": "Original LLM vocabulary size"},
    )
    codebook_size: int = field(
        default=8192,
        metadata={"help": "VQ codebook size"},
    )
    num_vq_tokens: int = field(
        default=1024,
        metadata={"help": "Number of VQ tokens"},
    )
    num_new_special_tokens: int = field(
        default=0,
        metadata={"help": "Number of new special tokens added"},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to tie word embeddings"},
    )
    enable_tf32: bool = field(
        default=True,
        metadata={"help": "Enable TensorFloat-32 for faster training on Ampere GPUs"},
    )
    max_seq_length: int = field(
        default=3072,
        metadata={"help": "Maximum sequence length for text tokens"},
    )
    resolution: int = field(
        default=384,
        metadata={"help": "Image resolution"},
    )
    center_crop: bool = field(
        default=False,
        metadata={"help": "Whether to center crop images"},
    )
    random_flip: bool = field(
        default=False,
        metadata={"help": "Whether to randomly flip images"},
    )
    noise_type: str = field(
        default="mask",
        metadata={"help": "Type of noise to apply"},
    )
    cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Conditional dropout probability"},
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to save deepspeed optimizer states or only the model"
        },
    )
    # inpainting related arguments: start
    inpaint_ratio: float = field(
        default=0.8,
        metadata={"help": "how many reasoning tokens to give as prompt"},
    )
    min_chunk_size: int = field(
        default=4,
        metadata={"help": "min chunk size for inpainting"},
    )
    max_chunk_size: int = field(
        default=8,
        metadata={"help": "max chunk size for inpainting"},
    )
    scaling_factor_power: float = field(
        default=1.2,
        metadata={"help": "scaling factor power for streching in inpainting"},
    )
    max_spacing: int = field(
        default=100,
        metadata={"help": "maximum spacing in streching in inpainting"},
    )
    entropy_clipping_ratio_inpaint: float = field(
        default=0.2,
        metadata={
            "help": "top % of highest entropy tokens to update on the inpainted tokens."
        },
    )
    metamath_split: str = field(
        default="ansaug_combined",
    )
    inpaint_last_chunk_end: bool = field(
        default=False,
        metadata={"help": "Whether to inpaint the last chunk"},
    )
    model_path: Optional[str] = field(
        default="LLaDA-8B-Instruct",
    )
    sft_model_path: Optional[str] = field(
        default=None,
    )
    # clip higher:
    epsilon_high: float = field(
        default=0.28,
    )
    epsilon_low: float = field(
        default=0.2,
    )
    positive_example_higheradvantage: float = field(
        default=0.0,
    )
    max_replacements: int = field(
        default=1,
    )
    use_dapo: bool = field(
        default=False,
    )
    d1_masking: bool = field(
        default=False,
    )
    use_gspo: bool = field(
        default=False,
    )
    random_gen_len: bool = field(
        default=False,
    )
    no_vapo_oninpaint: bool = field(
        default=False,
    )
    # chunk decays: U[low, high]
    inpaint_chunk_high_final_ratio: float = field(
        default=0.0,
        metadata={"help": "Final inpaint ratio at the end of scheduling"},
    )
    inpaint_chunk_decay_steps: int = field(
        default=1500,
        metadata={"help": "Number of steps over which to schedule the inpaint ratio"},
    )
    inpaint_chunk_decay_type: str = field(
        default="none",
        metadata={
            "help": "Type of scheduling: 'linear', 'cosine', 'exponential', or 'step'"
        },
    )
    inpaint_chunk_ratio_low: float = field(
        default=0.1,
        metadata={
            "help": "Probability of applying inpainting when all samples are wrong"
        },
    )
    inpaint_chunk_high_initial_ratio: float = field(
        default=0.5,
        metadata={
            "help": "Probability of applying inpainting when all samples are wrong"
        },
    )
    # inpaint prompt level decay:
    inpaint_prompt_final_ratio: float = field(
        default=0.0,
        metadata={"help": "Final inpaint ratio at the end of scheduling"},
    )
    inpaint_prompt_decay_steps: int = field(
        default=1500,
        metadata={"help": "Number of steps over which to schedule the inpaint ratio"},
    )
    inpaint_prompt_decay_type: str = field(
        default="none",
        metadata={
            "help": "Type of scheduling: 'linear', 'cosine', 'exponential', or 'step'"
        },
    )
    inpaint_prompt_initial_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Probability of applying inpainting when all samples are wrong"
        },
    )

