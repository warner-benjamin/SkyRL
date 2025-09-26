## Guide: Verifiers (Environments Hub) + SkyRL

This directory holds the workflow to train on Environments Hub environments with SkyRL.

To start training, follow three simple steps:
1) Install one or more environments from Environments Hub.
2) Prepare training and validation datasets (single or multi-environment).
3) Launch training!

Start by following the SkyRL [installation instructions](https://skyrl.readthedocs.io/en/latest/getting-started/installation.html), then enter the `skyrl-train` directory:
```bash
cd SkyRL/skyrl-train
```

### 1) Install the environment(s)
First, specify your desired environment(s) from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments):
```bash
export ENV_ID="will/wordle"
```

Then, install the environment, which adds it to the `uv` project:
```bash
uv run integrations/verifiers/install_environment.py $ENV_ID
```

For multiple environments, run the install step for each env ID:
```bash
uv run integrations/verifiers/install_environment.py will/wordle
uv run integrations/verifiers/install_environment.py verifiers/gsm8k
```

### 2) Prepare the dataset(s)
Next, load the dataset(s) and convert to SkyRL format.

Single environment:
```bash
uv run --isolated --with verifiers \
  python integrations/verifiers/prepare_dataset.py \
  --env_id $ENV_ID
```

Multiple environments (two equivalent ways):
```bash
# Repeated flags
uv run --isolated --with verifiers \
  python integrations/verifiers/prepare_dataset.py \
  --env_id will/wordle --env_id verifiers/gsm8k \
  --num_train 1000 --num_eval 200 \
  --output_dir "$HOME/data/verifiers_mix" \
  --seed 42 --write_per_env

# Comma-separated list
uv run --isolated --with verifiers \
  python integrations/verifiers/prepare_dataset.py \
  --env_ids will/wordle,verifiers/gsm8k,reverse_text \
  --num_train 500 --num_eval 200 \
  --output_dir "$HOME/data/verifiers_combo" \
  --seed 123
```

`prepare_dataset.py` options:
  - `--env_id` (repeatable): environment identifier(s)
  - `--env_ids`: comma-separated list of environment identifiers
  - `--output_dir`: directory to place datasets (default: `~/data/{first_env}`; if multi-env: `~/data/{first_env}_multi`)
  - `--num_train`: number of training samples per environment (-1 for no limit)
  - `--num_eval`: number of validation samples per environment (-1 for no limit)
  - `--seed`: shuffle seed for combined datasets (default: 42)
  - `--write_per_env`: also write per-env parquets under `output_dir/{env}/`

Notes on dataset generation:
- For single env, this script generates Parquet files under `output_dir`:
  - `train.parquet`
  - `validation.parquet`
- For multiple envs, it concatenates all per-env splits, shuffles deterministically with `--seed`, and writes combined:
  - `train.parquet`
  - `validation.parquet`
- If `--write_per_env` is set, it also writes per-env files:
  - `output_dir/{env}/train.parquet`
  - `output_dir/{env}/validation.parquet`
- For issues in loading the dataset, see the Troubleshooting section below.

### 3) Launch training
Open `run_verifiers.sh`, which specifies the training configuration parameters and is the primary interface for launching training runs.

Modify the commonly-edited training settings as needed:
```bash
ENV_ID="will/wordle"
DATA_DIR="$HOME/data/$ENV_ID"
NUM_GPUS=1
LOGGER="wandb"
```

Finally, launch your training run:

```bash
bash integrations/verifiers/run_verifiers.sh
```

All training parameters can be modified in `run_verifiers.sh`, such as the model choice (`trainer.policy.model.path`), GRPO group size (`generator.n_samples_per_prompt`), or training batch size (`trainer.train_batch_size`).

See all available training configuration parameters in `ppo_base_config.yaml`.


### About multi-environment training
- The combined Parquet produced by `prepare_dataset.py` contains rows from different environments. Each row includes a `verifiers` object with `environment`, `answer`, `task`, and (optional) `info` fields.
- The Verifiers generator supports mixed-environment batches: it groups samples by environment, calls the environment’s `a_generate` concurrently, processes outputs per env, and merges back in-order.
- You don’t need to change training code to use multi-env data—just point `DATA_DIR` to the combined dataset directory.


## Troubleshooting

For issues with SkyRL or the integration with Verifiers, please [open an Issue](https://github.com/NovaSky-AI/SkyRL/issues/new). 


### Datasets
Verifiers environments can handle dataset splits in different ways. Some environments require passing a `dataset_split` argument to `load_environment()` (e.g., to specify `train` vs `test`), others implement both `vf_env.load_dataset()` and `vf_env.load_eval_dataset()`.

The implementation in `prepare_dataset.py` uses `vf_env.get_dataset()` and `vf_env.get_eval_dataset()` per environment where available, skips missing train splits with a warning, and always writes combined evaluation if available. It normalizes the schema to include `verifiers.info` (set to `None` if missing) and removes only the top-level `info` column before Parquet serialization.


## TODOs and Limitations
We welcome any contributions to help resolve the remaining tasks!
* Improve ergonomics for choosing different Verifiers environments used specifically for training vs validation.
* Make it smoother to specify which dataset splits to use (ie, resolve the challenge specified in the `Troubleshooting` section.)
* Consider plumbing Verifiers-specific configuration to the VerifiersGenerator for easy override. For example: `zero_truncated_completions` and `mask_truncated_completions`.
