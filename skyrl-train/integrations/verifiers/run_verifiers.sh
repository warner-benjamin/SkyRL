#!/usr/bin/env bash
# Launches SkyRL training on Verifiers environments (single or multiple).
#
# Examples:
#   # Single env
#   bash integrations/verifiers/run_verifiers.sh
#
#   # Multi env (datasets already prepared into $DATA_DIR)
#   ENV_IDS="will/wordle,verifiers/gsm8k" DATA_DIR="$HOME/data/verifiers_mix" \
#     bash integrations/verifiers/run_verifiers.sh

set -x

# Config (can be overridden via environment)
ENV_ID=${ENV_ID:-"will/wordle"}   # Single environment ID (org/name@version)
ENV_IDS=${ENV_IDS:-""}            # Comma-separated list for multi-env
DATA_DIR=${DATA_DIR:-""}          # Directory with train.parquet/validation.parquet
NUM_GPUS=${NUM_GPUS:-1}
LOGGER=${LOGGER:-"wandb"}         # change to "console" to print to stdout

# Determine dataset directory and label
if [[ -n "$ENV_IDS" ]]; then
  # Multi-env mode: DATA_DIR must point to the combined dataset
  if [[ -z "$DATA_DIR" ]]; then
    FIRST_ENV="${ENV_IDS%%,*}"
    DATA_DIR="$HOME/data/${FIRST_ENV##*/}_multi"
  fi
  ENV_LABEL="verifiers_multi"
else
  # Single-env mode
  DATA_DIR="${DATA_DIR:-$HOME/data/$ENV_ID}"
  ENV_LABEL="$ENV_ID"
fi

# Launch training
uv run --isolated --with verifiers --extra vllm -m integrations.verifiers.entrypoints.main_verifiers \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.n_samples_per_prompt=5 \
  trainer.epochs=20 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=128 \
  trainer.micro_forward_batch_size_per_gpu=32 \
  trainer.micro_train_batch_size_per_gpu=32 \
  trainer.max_prompt_length=8192 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=1024 \
  generator.enable_http_endpoint=true \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  environment.env_class="$ENV_LABEL" \
  trainer.project_name="verifiers" \
  trainer.run_name="verifiers_test" \
  trainer.ckpt_interval=-1 \
  trainer.ckpt_path="$HOME/ckpts/verifiers_ckpt" \
  "$@"