import json
import os
import random
import re
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

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


@torch.no_grad()
def generate_with_prefix_cache_inpaint(
    tokenizer,
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=1,
    reference_answer: str = None,
    inpaint_ratio: float = 0.7,
    min_chunk_size: int = 4,
    max_chunk_size: int = 8,
    inpaint_rollouts_per_group: int = 1,
    scaling_factor_power: float = 1.2,
    max_spacing: int = 100,
    inpaint_last_chunk_end: bool = False,
    seed: Optional[int] = None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    batch_size = prompt.shape[0]

    if reference_answer is not None and inpaint_rollouts_per_group > 0:
        if isinstance(reference_answer, list):
            reference_answer = reference_answer[
                0
            ]  # this assumes all prompt in a batch uses same reference answer
        # Create mixed batch with inpainting and normal generation
        x = torch.full(
            (batch_size, prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        # Initialize generation mask (all positions in generation area)
        generation_mask = torch.zeros_like(x, dtype=torch.bool)
        generation_mask[:, prompt.shape[1] :] = True

        # Apply inpainting to first inpaint_rollouts_per_group
        if inpaint_rollouts_per_group > 0:
            # Get inpainting sequence for the samples that need it
            inpaint_prompt = prompt[:inpaint_rollouts_per_group]
            inpaint_x, inpaint_generation_mask = (
                create_spacing_scaled_inpainting_sequence(
                    tokenizer,
                    inpaint_prompt,
                    reference_answer,
                    gen_length,
                    mask_ratio=1 - inpaint_ratio,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    scaling_factor_power=scaling_factor_power,
                    max_spacing=max_spacing,
                    mask_id=mask_id,
                    seed=seed,
                    inpaint_last_chunk_end=inpaint_last_chunk_end,
                )
            )

            x[:inpaint_rollouts_per_group] = inpaint_x
            generation_mask[:inpaint_rollouts_per_group] = inpaint_generation_mask

    else:
        # Fall back to original masking for all samples
        x = torch.full(
            (batch_size, prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()
        generation_mask = torch.zeros_like(x, dtype=torch.bool)
        generation_mask[:, prompt.shape[1] :] = True

    first_input = x.clone()
    fixed_positions = ~generation_mask

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        if not block_mask_index.any():
            # No masks to fill in this block, skip
            continue
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = x == mask_id
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                output.logits, temperature, remasking, mask_index, x, None, factor
            )
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (
                    past_key_values[i][j][:, :, :current_block_start],
                )

        past_key_values = new_past_key_values
        nfe += 1

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = x[:, current_block_start:] == mask_id
            mask_index[:, block_length:] = 0

            logits = model(
                x[:, current_block_start:],
                past_key_values=past_key_values,
                use_cache=True,
            ).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:],
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:],
                    None,
                    factor,
                )
            x[:, current_block_start:][transfer_index] = x0[transfer_index]

            i += 1

        eot_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        current_block = x[:, current_block_start:current_block_end]
        eot_count_in_block = (current_block == eot_token_id).sum(dim=1)
        block_size = current_block_end - current_block_start

        if (eot_count_in_block == block_size).all():
            remaining_mask = (x == mask_id) & generation_mask
            x[remaining_mask] = eot_token_id
            return x, fixed_positions, first_input, nfe

    return x, fixed_positions, first_input, nfe


def get_transfer_index(
    logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_dynamic(
    logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        # Skip batch elements with no tokens to transfer
        if num_transfer_tokens[j].item() == 0:
            continue
        ns = list(range(1, num_transfer_tokens[j].item() + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def create_spacing_scaled_inpainting_sequence(
    tokenizer,
    question_tokens: torch.Tensor,
    reference_answer: str,
    gen_length: int,
    mask_ratio: float = 0.7,
    min_chunk_size: int = 3,
    max_chunk_size: int = 8,
    mask_id: int = 126336,
    max_spacing: int = 30,  # Maximum allowed spacing between chunks
    seed: Optional[int] = None,
    inpaint_last_chunk_end: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create an inpainting sequence where chunk spacing is scaled based on the ratio
    of answer length to available generation length.

    Args:
        scaling_factor_power: Power to apply to scaling factor (1.0 = linear, <1.0 = sublinear)
        max_spacing: Maximum allowed spacing between chunks to prevent over-dispersion
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    batch_size = question_tokens.shape[0]
    device = question_tokens.device

    # Calculate effective generation length: leave final 50 tokens for free generation
    effective_gen_length = max(0, gen_length - 30)
    free_generation_length = gen_length - effective_gen_length

    # Start with all mask tokens for the generation area
    generation_area = torch.full(
        (batch_size, gen_length), mask_id, dtype=torch.long, device=device
    )

    # Handle the case where we have a reference answer (only in the effective generation area)
    if (
        reference_answer
        and len(reference_answer.strip()) > 0
        and effective_gen_length > 0
    ):
        # Tokenize the reference answer
        answer_tokens = tokenizer.encode(reference_answer, add_special_tokens=False)
        answer_length = len(answer_tokens)

        if answer_length > 0:
            # Calculate scaling factor
            scaling_factor = effective_gen_length / answer_length
            scaling_factor = max(1, scaling_factor)  # can not shorten
            scaling_factor = min(1.5, scaling_factor)  # don't scale too far.
            # scaling_factor = 1  # TODO: remove this line

            print(
                f"Answer length: {answer_length}, Effective gen length: {effective_gen_length}"
            )
            print(f"Scaling factor: {scaling_factor:.2f}")

            # Use as many answer tokens as fit in the effective generation area
            actual_answer_length = min(answer_length, effective_gen_length)
            answer_tokens = answer_tokens[:actual_answer_length]
            answer_tensor = (
                torch.tensor(answer_tokens, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

            # Create chunks for partial masking (same as original logic)
            answer_mask_positions = torch.zeros(actual_answer_length, dtype=torch.bool)

            # Special case: if mask_ratio is 1.0, mask everything (normal generation)
            if mask_ratio >= 1.0:
                answer_mask_positions[:] = True
            # Special case: if mask_ratio is 0.0, reveal everything
            elif mask_ratio <= 0.0:
                answer_mask_positions[:] = False
            else:
                # Create chunks for partial masking
                chunks = []
                i = 0
                while i < actual_answer_length:
                    # Sample chunk size for this chunk
                    chunk_size = random.randint(min_chunk_size, max_chunk_size)
                    # Don't exceed remaining tokens
                    chunk_size = min(chunk_size, actual_answer_length - i)
                    if (
                        chunk_size < min_chunk_size
                    ):  # don't add too small chunks, it is hard for model to inpaint
                        break
                    chunks.append((i, i + chunk_size))
                    i += chunk_size

                # Calculate how many chunks to mask based on mask_ratio
                total_chunks = len(chunks)
                if total_chunks > 0:
                    num_chunks_to_mask = max(0, int(total_chunks * mask_ratio))

                    if num_chunks_to_mask > 0:
                        # Randomly sample chunks to mask
                        chunks_to_mask = random.sample(
                            chunks, min(num_chunks_to_mask, total_chunks)
                        )

                        # Convert chunk ranges to individual token indices
                        mask_indices = []
                        for start_idx, end_idx in chunks_to_mask:
                            mask_indices.extend(range(start_idx, end_idx))

                        answer_mask_positions[mask_indices] = True

            # Filter out masked chunks - only keep chunks that should be revealed
            revealed_chunks = []

            for i, (start, end) in enumerate(chunks):
                chunk_should_be_revealed = not any(answer_mask_positions[start:end])
                # Always reveal the last chunk
                is_last_chunk = i == len(chunks) - 1
                if chunk_should_be_revealed or (
                    is_last_chunk and inpaint_last_chunk_end
                ):
                    revealed_chunks.append((start, end))

            # Now scale the spacing and place tokens in the generation area
            scaled_positions = scale_chunk_positions(
                chunks=revealed_chunks,
                original_length=actual_answer_length,
                target_length=effective_gen_length,
                scaling_factor=scaling_factor,
                max_spacing=max_spacing,
            )

            # Create the final generation area with scaled positions
            scaled_generation_area = torch.full(
                (batch_size, effective_gen_length),
                mask_id,
                dtype=torch.long,
                device=device,
            )

            # Place revealed chunks at scaled positions
            chunk_idx = 0
            for original_start, original_end in revealed_chunks:
                if chunk_idx < len(scaled_positions):
                    scaled_start, scaled_end = scaled_positions[chunk_idx]
                    chunk_tokens = answer_tokens[original_start:original_end]

                    # Ensure we don't exceed bounds
                    actual_scaled_end = min(
                        scaled_end,
                        scaled_start + len(chunk_tokens),
                        effective_gen_length,
                    )
                    actual_chunk_length = actual_scaled_end - scaled_start

                    if actual_chunk_length > 0:
                        # Place the chunk tokens at scaled position
                        for batch_idx in range(batch_size):
                            scaled_generation_area[
                                batch_idx, scaled_start:actual_scaled_end
                            ] = torch.tensor(
                                chunk_tokens[:actual_chunk_length], device=device
                            )

                    chunk_idx += 1

            # Set the scaled generation area in the full generation area
            generation_area[:, :effective_gen_length] = scaled_generation_area

    # The final 50 tokens (free_generation_length) remain as mask tokens
    # This is already handled since generation_area was initialized with mask_id

    # Create the full input sequence
    input_sequence = torch.full(
        (batch_size, question_tokens.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )

    # Set question tokens (never masked)
    input_sequence[:, : question_tokens.shape[1]] = question_tokens

    # Set the generation area (scaled inpainted area + free generation area)
    answer_start_idx = question_tokens.shape[1]
    input_sequence[:, answer_start_idx : answer_start_idx + gen_length] = (
        generation_area
    )

    # Create target mask: True for positions that need to be generated
    target_mask = torch.zeros_like(input_sequence, dtype=torch.bool)
    target_mask[:, answer_start_idx : answer_start_idx + gen_length] = (
        generation_area == mask_id
    )

    return input_sequence, target_mask


def scale_chunk_positions(
    chunks: List[Tuple[int, int]],
    original_length: int,
    target_length: int,
    scaling_factor: float,
    max_spacing: int = 30,
) -> List[Tuple[int, int]]:
    """
    Scale chunk positions based on the scaling factor while preserving relative spacing.

    Args:
        chunks: List of (start, end) positions in original sequence
        original_length: Length of original answer
        target_length: Length of target generation area
        scaling_factor: Factor by which to scale spacing
        max_spacing: Maximum allowed spacing between chunks

    Returns:
        List of scaled (start, end) positions
    """
    if not chunks:
        return []

    scaled_positions = []

    for i, (start, end) in enumerate(chunks):
        chunk_length = end - start

        if i == 0:
            # First chunk starts at scaled position
            scaled_start = int(start * scaling_factor)
        else:
            # Calculate spacing from previous chunk
            prev_end = chunks[i - 1][1]
            original_spacing = start - prev_end

            # Scale the spacing but cap it at max_spacing
            scaled_spacing = min(int(original_spacing * scaling_factor), max_spacing)
            scaled_start = scaled_positions[-1][1] + scaled_spacing

        scaled_end = scaled_start + chunk_length

        # Ensure we don't exceed target length
        if scaled_end > target_length:
            # If we exceed, compress remaining chunks
            remaining_space = target_length - scaled_start
            if remaining_space > 0:
                scaled_end = scaled_start + min(chunk_length, remaining_space)
            else:
                break  # No more space

        scaled_positions.append((scaled_start, scaled_end))

    return scaled_positions


@torch.no_grad()
def generate_inpainting(
    tokenizer,
    model,
    question_tokens: torch.Tensor,
    reference_answer: str = None,
    inpaint_ratio: Union[float, List[float]] = 0.7,  # Now accepts list
    min_chunk_size: int = 4,
    max_chunk_size: int = 8,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    seed: Optional[int] = None,
    inpaint_rollouts_per_group: int = 1,
    max_spacing: int = 100,
):
    """
    Generate with inpainting where some answer tokens are pre-filled.

    Args:
        Same as original generate function, plus:
        reference_answer: Reference answer to partially reveal
        inpaint_ratio: Single float or list of floats (one per batch sample) for inpaint ratios
        seed: Random seed for mask pattern
        inpaint_rollouts_per_group: Number of samples to apply inpainting to
    """
    import torch.nn.functional as F

    batch_size = question_tokens.shape[0]

    if inpaint_ratio:
        if isinstance(inpaint_ratio, (list, tuple)):
            if len(inpaint_ratio) != batch_size:
                raise ValueError(
                    f"Length of inpaint_ratio list ({len(inpaint_ratio)}) must match batch_size ({batch_size})"
                )
            inpaint_ratios = list(inpaint_ratio)
        else:
            inpaint_ratios = [inpaint_ratio] * batch_size

    if reference_answer is not None and inpaint_rollouts_per_group > 0 and inpaint_ratio[0] > 0:
        if isinstance(reference_answer, list):
            reference_answer = reference_answer[
                0
            ]  # this assumes all prompt in a batch uses same reference answer

        x = torch.full(
            (batch_size, question_tokens.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, : question_tokens.shape[1]] = question_tokens.clone()

        # Initialize generation mask (all positions in generation area)
        generation_mask = torch.zeros_like(x, dtype=torch.bool)
        generation_mask[:, question_tokens.shape[1] :] = True

        num_inpaint_samples = min(inpaint_rollouts_per_group, batch_size)

        if num_inpaint_samples > 0:
            # Process each sample with its specific inpaint ratio
            for sample_idx in range(num_inpaint_samples):
                sample_inpaint_ratio = inpaint_ratios[sample_idx]

                # Create inpainting sequence for this single sample
                single_question_tokens = question_tokens[sample_idx : sample_idx + 1]

                inpaint_x, inpaint_generation_mask = (
                    create_spacing_scaled_inpainting_sequence(
                        tokenizer,
                        single_question_tokens,
                        reference_answer,
                        gen_length,
                        mask_ratio=1
                        - sample_inpaint_ratio,  # Convert inpaint_ratio to mask_ratio
                        min_chunk_size=min_chunk_size,
                        max_chunk_size=max_chunk_size,
                        max_spacing=max_spacing,
                        mask_id=mask_id,
                        seed=seed + sample_idx
                        if seed is not None
                        else None,  # Unique seed per sample
                    )
                )

                # Update the batch tensors for this sample
                x[sample_idx : sample_idx + 1] = inpaint_x
                generation_mask[sample_idx : sample_idx + 1] = inpaint_generation_mask

    else:
        # Fall back to original masking for all samples
        x = torch.full(
            (batch_size, question_tokens.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, : question_tokens.shape[1]] = question_tokens.clone()
        generation_mask = torch.zeros_like(x, dtype=torch.bool)
        generation_mask[:, question_tokens.shape[1] :] = True

    first_input = x.clone()

    # Track which positions should never be modified (question + revealed answer tokens)
    fixed_positions = ~generation_mask

    # Rest of the generation logic remains the same...
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = question_tokens.shape[1] + num_block * block_length
        block_end = question_tokens.shape[1] + (num_block + 1) * block_length

        # Only consider masks in current block that are part of generation mask
        block_generation_mask = generation_mask[:, block_start:block_end]
        block_mask_index = (
            x[:, block_start:block_end] == mask_id
        ) & block_generation_mask

        if not block_mask_index.any():
            # No masks to fill in this block, skip
            continue

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            # Only consider positions that are masked AND part of generation mask
            mask_index = (x == mask_id) & generation_mask

            if not mask_index.any():
                break

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[fixed_positions] = (
                    mask_id  # Mask fixed positions for unconditional
                )
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float32)
            else:
                raise NotImplementedError(remasking)

            # Don't consider positions outside current block or fixed positions
            x0_p[:, block_end:] = -np.inf
            x0_p[fixed_positions] = -np.inf

            # Only update masked positions that are part of generation
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Select positions to update
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                if num_transfer_tokens[j, i] > 0:
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i], largest=True
                    )
                    transfer_index[j, select_index] = True

            # Only transfer if position is both selected and part of generation mask
            transfer_index = transfer_index & mask_index
            x[transfer_index] = x0[transfer_index]

        # Early exit check (same as original)
        eot_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        current_block = x[:, block_start:block_end]
        eot_count_in_block = (current_block == eot_token_id).sum(dim=1)
        block_size = block_end - block_start

        if (eot_count_in_block == block_size).all():
            remaining_mask = (x == mask_id) & generation_mask
            x[remaining_mask] = eot_token_id
            return x, fixed_positions, first_input

    return x, fixed_positions, first_input


def load_existing_results(output_filename):
    """Load existing results if the file exists"""
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    return None


def save_results_incrementally(results, output_filename):
    """Save results incrementally to avoid losing progress"""
    # Create backup
    if os.path.exists(output_filename):
        backup_filename = output_filename.replace(".json", "_backup.json")
        with open(backup_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Save main file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run inpainting evaluation with specified model path"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint or pretrained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting sample index (for resuming, default: 0)",
    )
    parser.add_argument(
        "--end_idx", type=int, default=250, help="Ending sample index (default: 250)"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda"
    model_path = args.model_path

    # Extract model name from path
    if "noinpaint" in model_path:
        ckpt_num = model_path[-5:]
        model_name = "noinpaint" + ckpt_num
    elif "inpaint" in model_path:
        ckpt_num = model_path[-5:]
        model_name = "inpaint" + ckpt_num
    else:
        model_name = "MixCoT_test"
        # model_name = 'inpaint_lora'
    print("model name", model_name)

    # Create output directory if it doesn't exist
    output_dir = "inpaint_rl_eval_results_fix"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{model_name}_results.json"

    # Try to load existing results
    existing_results = load_existing_results(output_filename)
    processed_indices = set()

    if existing_results is not None:
        print(
            f"Found existing results file with {len(existing_results['samples'])} samples"
        )
        processed_indices = {
            sample["sample_id"] for sample in existing_results["samples"]
        }
        results = existing_results
        all_correct = [sample["is_correct"] for sample in existing_results["samples"]]
        print(f"Already processed samples: {sorted(processed_indices)}")
    else:
        # Initialize new results structure
        results = {
            "model_name": model_name,
            "model_path": model_path,
            "total_samples": args.end_idx - args.start_idx,
            "samples": [],
            "accuracy": 0.0,
        }
        all_correct = []

    # Load model and tokenizer only if we have work to do
    remaining_indices = [
        i for i in range(args.start_idx, args.end_idx) if i not in processed_indices
    ]
    if not remaining_indices:
        print("All samples already processed!")
        print(f"Final accuracy: {results['accuracy']:.4f}")
        return

    print(f"Loading model from {model_path}...")
    model = (
        MMadaModelLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
    think_prompt = "Answer the following math question. You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"

    print("Loading dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_ds = list(dataset["test"])

    print(
        f"Processing remaining {len(remaining_indices)} samples: {remaining_indices[:10]}{'...' if len(remaining_indices) > 10 else ''}"
    )

    for i in tqdm(remaining_indices, desc="Processing samples"):
        try:
            question = test_ds[i]["question"]
            answer = test_ds[i]["answer"]
            m = [{"role": "user", "content": think_prompt + question}]
            prompt = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer(
                text=prompt, return_tensors="pt", padding=True, padding_side="left"
            )["input_ids"]
            input_ids = input_ids.to(device)

            out, _, _ = generate_inpainting(
                tokenizer,
                model,
                input_ids,
                reference_answer=answer,
                inpaint_ratio=0.2,
                steps=128,
                gen_length=256,
                block_length=32,
                temperature=0,
                cfg_scale=0.0,
                remasking="low_confidence",
                seed=seed,
                inpaint_ratio_per_group=1,
            )

            generation = tokenizer.batch_decode(
                out[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            is_correct, parsed_answer, gt_answer_value = manual_gsm8k_verify(
                answer, generation
            )

            # Store sample results
            sample_result = {
                "sample_id": i,
                "question": question,
                "ground_truth_answer": answer,
                "generation": generation,
                "parsed_answer": parsed_answer,
                "gt_answer_value": gt_answer_value,
                "is_correct": is_correct,
            }

            results["samples"].append(sample_result)
            all_correct.append(is_correct)

            # Calculate and update accuracy
            results["accuracy"] = sum(all_correct) / len(all_correct)

            # Print progress
            print("-" * 90)
            print("GENERATION:", generation)
            print("-" * 90)
            print("gt answer:", answer)
            print("-" * 90)
            print(is_correct)
            current_accuracy = sum(all_correct) / len(all_correct)
            print(f"accuracy so far: {current_accuracy:.4f}. Step {i}. {model_name}")
            print(model_path)

            # Save results incrementally every 10 samples
            if len(results["samples"]) % 10 == 0:
                save_results_incrementally(results, output_filename)
                print(f"Saved progress: {len(results['samples'])} samples processed")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Save current progress even on error
            save_results_incrementally(results, output_filename)
            continue

    # Final saves
    save_results_incrementally(results, output_filename)

    print(f"\nResults saved to {output_filename}")
    print(f"Final accuracy: {results['accuracy']:.4f}")
    print(f"Total samples processed: {len(results['samples'])}")


if __name__ == "__main__":
    main()

