# =====================================================================>>>>>>> REPLACE
# EXPERIMENT CONFIGURATION
# ============================================================================
declare -a EXPERIMENTS=(
    "dataset=metamath,max_completion_length=256,use_dapo=false,use_gspo=true,d1_masking=true,lr=5e-7,num_iterations=4,temperature=1.2,num_train_epochs=10,max_replacement=4,entropy_clipping_ratio_inpaint=0.2,inpaint_chunk_high_initial_ratio=0.6,inpaint_chunk_ratio_low=0.2"
    # baseline:resample size matched grpo: set inpainting ratio = 0
    # "dataset=metamath,max_completion_length=256,use_dapo=false,use_gspo=true,d1_masking=true,lr=5e-7,num_iterations=4,temperature=1.2,num_train_epochs=10,max_replacement=4,entropy_clipping_ratio_inpaint=0.2,inpaint_chunk_high_initial_ratio=0.0,inpaint_chunk_ratio_low=0.0"
)

echo "Total experiments to run: ${#EXPERIMENTS[@]}"
echo ""

# Function to parse experiment parameters
parse_experiment() {
    local exp_string="$1"
    IFS=',' read -ra PARAMS <<< "$exp_string"
    # Initialize variables with defaults
    dataset=""
    max_completion_length=""
    use_dapo=""
    use_gspo=""
    d1_masking=""
    lr=""
    num_iterations=""
    positive_example_higheradvantage=""
    temperature=""
    num_train_epochs=""
    max_replacement=""
    entropy_clipping_ratio_inpaint=""
    inpaint_chunk_high_initial_ratio=""
    inpaint_chunk_high_final_ratio=""
    inpaint_chunk_decay_steps=""
    inpaint_chunk_decay_type=""
    inpaint_chunk_ratio_low=""
    inpaint_prompt_initial_ratio=""
    inpaint_prompt_final_ratio=""
    inpaint_prompt_decay_steps=""
    inpaint_prompt_decay_type=""

    # Parse each parameter
    for param in "${PARAMS[@]}"; do
        IFS='=' read -ra PAIR <<< "$param"
        key="${PAIR[0]}"
        value="${PAIR[1]}"

        case "$key" in
            "dataset") dataset="$value" ;;
            "max_completion_length") max_completion_length="$value" ;;
            "use_dapo") use_dapo="$value" ;;
            "use_gspo") use_gspo="$value" ;;
            "d1_masking") d1_masking="$value" ;;
            "lr") lr="$value" ;;
            "num_iterations") num_iterations="$value" ;;
            "positive_example_higheradvantage") positive_example_higheradvantage="$value" ;;
            "temperature") temperature="$value" ;;
            "num_train_epochs") num_train_epochs="$value" ;;
            "max_replacement") max_replacement="$value" ;;
            "entropy_clipping_ratio_inpaint") entropy_clipping_ratio_inpaint="$value" ;;
            "inpaint_chunk_high_initial_ratio") inpaint_chunk_high_initial_ratio="$value" ;;
            "inpaint_chunk_high_final_ratio") inpaint_chunk_high_final_ratio="$value" ;;
            "inpaint_chunk_decay_steps") inpaint_chunk_decay_steps="$value" ;;
            "inpaint_chunk_decay_type") inpaint_chunk_decay_type="$value" ;;
            "inpaint_chunk_ratio_low") inpaint_chunk_ratio_low="$value" ;;
            "inpaint_prompt_initial_ratio") inpaint_prompt_initial_ratio="$value" ;;
            "inpaint_prompt_final_ratio") inpaint_prompt_final_ratio="$value" ;;
            "inpaint_prompt_decay_steps") inpaint_prompt_decay_steps="$value" ;;
            "inpaint_prompt_decay_type") inpaint_prompt_decay_type="$value" ;;
        esac
    done
}

# Function to run a single experiment
run_experiment() {
    local exp_num="$1"
    local exp_string="$2"

    echo "========================================="
    echo "EXPERIMENT $exp_num of ${#EXPERIMENTS[@]}"
    echo "========================================="

    # Parse experiment parameters
    parse_experiment "$exp_string"

    # Set derived parameters
    diffusion_steps=$((max_completion_length / 2))

    # Set generation batch size based on completion length
    if [ "$max_completion_length" -eq 512 ]; then
        generation_batch_size=4
    elif [ "$max_completion_length" -eq 256 ]; then
        generation_batch_size=8
    else
        generation_batch_size=8  # default
    fi

    # Set accumulation steps and mu based on num_iterations
    accum=8
    mu=$num_iterations

    # Set component strings for naming
    d1_component=""
    if [ "$d1_masking" == "true" ]; then
        d1_component="d1"
    fi

    # Set algorithm component for naming
    algo_component=""
    if [ "$use_dapo" = "true" ]; then
        algo_component="dapo"
    elif [ "$use_gspo" = "true" ]; then
        algo_component="gspo"
    fi

    # Create additional components for runname
    max_replacement_component="max${max_replacement}"
    entropy_clip_component="topp$(echo "$entropy_clipping_ratio_inpaint * 100" | bc | cut -d. -f1)"

    # Create inpaint chunk and prompt components
    inpaint_chunk_component="chunk${inpaint_chunk_high_initial_ratio}to${inpaint_chunk_high_final_ratio}${inpaint_chunk_decay_type}_${inpaint_chunk_decay_steps}steps"
    inpaint_prompt_component="prompt${inpaint_prompt_initial_ratio}to${inpaint_prompt_final_ratio}${inpaint_prompt_decay_type}_${inpaint_prompt_decay_steps}steps"

    # Create run_name and output_dir
    run_name="basellada_ck510_${d1_component}_${algo_component}_${dataset}_${max_replacement_component}_${entropy_clip_component}_${inpaint_chunk_component}_${inpaint_prompt_component}_${max_completion_length}_accum${accum}"
    output_dir="/genai/fsx-project/siyanzhao/igpo_rebuttal/${run_name}"

    accelerate launch \
        --config_file accelerate.yaml \
        --main_process_port 12334 train_igpo.py \
        --config train.yaml \
        --output_dir "$output_dir" \
        --save_step 96 \
        --num_train_epochs "$num_train_epochs" \
        --model_path /genai/fsx-project/siyanzhao/models/LLaDA-8B-Instruct \
        --save_only_model false \
        --run_name "$run_name" \
        --gradient_accumulation_steps "$accum" \
        --num_iterations "$num_iterations" \
        --num_generations 8 \
        --per_device_train_batch_size 8 \
        --generation_batch_size "$generation_batch_size" \
        --learning_rate "$lr" \
        --beta 0.01 \
        --epsilon_low 0.2 \
        --epsilon_high 0.2 \
        --seed 3 \
        --temperature "$temperature" \
        --min_chunk_size 5 \
        --max_chunk_size 10 \
        --block_length 32 \
        --max_completion_length "$max_completion_length" \
        --diffusion_steps "$diffusion_steps" \
        --entropy_clipping_ratio_inpaint "$entropy_clipping_ratio_inpaint" \
        --inpaint_chunk_high_initial_ratio "$inpaint_chunk_high_initial_ratio" \
        --inpaint_chunk_ratio_low "$inpaint_chunk_ratio_low" \
        --dataset "$dataset" \
        --max_replacements "$max_replacement" \
        --use_gspo "$use_gspo" \
        --d1_masking "$d1_masking" 

    echo "Completed experiment $exp_num: $run_name"
    echo "----------------------------------------"
}

# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

# Loop through all experiments
for i in "${!EXPERIMENTS[@]}"; do
    exp_num=$((i + 1))
    run_experiment "$exp_num" "${EXPERIMENTS[$i]}"
done

echo "All experiments completed!"
echo "Total experiments run: ${#EXPERIMENTS[@]}"
