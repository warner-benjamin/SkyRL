import argparse
import os
from functools import partial
from typing import Any, Dict, List, Optional

from verifiers import load_environment
from datasets import concatenate_datasets


def extract_env_name(env_id: str) -> str:
    """Return only the environment name from strings like 'org/name@version' or 'name@version'."""
    base = env_id.split("/")[-1]
    return base.split("@")[0]


def build_row(
    sample: Dict[str, Any], data_source: str, env_name: str
) -> Dict[str, Any]:
    """Normalize a single example into the expected SkyRL schema.

    Always include verifiers.info (None when missing) to keep schema consistent
    across environments for concatenation and Parquet serialization.
    """
    if "prompt" not in sample:
        raise ValueError("Example must contain a 'prompt' field")
    prompt = sample["prompt"]  # Already formatted by the environment as chat messages

    answer = sample.get("answer", "")
    info = sample.get("info", None)
    task = sample.get("task", "default")

    return {
        "data_source": data_source,
        "prompt": prompt,
        "verifiers": {
            "answer": answer,
            "task": task,
            "environment": env_name,
            "info": info if info else None,
        },
    }


def _resolve_env_list(
    single_ids: Optional[List[str]], csv_ids: Optional[str]
) -> List[str]:
    envs: List[str] = []
    if single_ids:
        envs.extend(single_ids)
    if csv_ids:
        envs.extend([e for e in (x.strip() for x in csv_ids.split(",")) if e])
    # de-dupe while preserving order
    seen = set()
    out: List[str] = []
    for e in envs:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def _write_parquet(ds, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_parquet(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Parquet datasets from one or more verifiers environments."
    )
    parser.add_argument(
        "--env_id",
        action="append",
        default=None,
        help="Environment identifier to load (e.g., 'wordle'). Can be provided multiple times.",
    )
    parser.add_argument(
        "--env_ids",
        default=None,
        help="Comma-separated list of environment identifiers (e.g., 'wordle,gsm8k').",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for Parquet files. Defaults to ~/data/{first_env} or ~/data/{first_env}_multi.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=-1,
        help="Number of training examples per environment. -1 for no limit.",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=-1,
        help="Number of evaluation examples per environment. -1 for no limit.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed for combined datasets."
    )
    parser.add_argument(
        "--write_per_env",
        action="store_true",
        help="Also write per-env parquet files under output_dir/{env}/.",
    )

    args = parser.parse_args()

    env_ids = _resolve_env_list(args.env_id, args.env_ids)
    if not env_ids:
        env_ids = ["wordle"]

    env_names = [extract_env_name(e) for e in env_ids]

    # Resolve output directory
    default_dir = (
        f"~/data/{env_names[0]}"
        if len(env_names) == 1
        else f"~/data/{env_names[0]}_multi"
    )
    output_dir_name = args.output_dir if args.output_dir else default_dir
    output_dir = os.path.expanduser(output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    per_env_train = []
    per_env_eval = []

    for raw_id, env_name in zip(env_ids, env_names):
        print(f"Preparing environment: {raw_id} -> {env_name}")
        vf_env = load_environment(env_id=env_name)
        data_source = f"verifiers/{env_name}"
        map_fn = partial(build_row, data_source=data_source, env_name=env_name)

        # Train split (optional)
        try:
            train_ds = vf_env.get_dataset(args.num_train)
        except ValueError:
            train_ds = None
            print(
                f"WARNING: Environment {raw_id} does not have a training dataset. Continuing with eval only."
            )
        if train_ds and len(train_ds) > 0:
            train_ds = train_ds.map(map_fn, num_proc=16)
            # Drop top-level 'info' column which may be an empty dict and not parquet-serializable
            if "info" in train_ds.column_names:
                train_ds = train_ds.remove_columns("info")
            per_env_train.append((env_name, train_ds))
            if args.write_per_env:
                _write_parquet(
                    train_ds, os.path.join(output_dir, env_name, "train.parquet")
                )
        else:
            print(f"INFO: No train examples for {env_name}.")

        # Eval split
        eval_ds = vf_env.get_eval_dataset(args.num_eval)
        if eval_ds and len(eval_ds) > 0:
            eval_ds = eval_ds.map(map_fn, num_proc=16)
            if "info" in eval_ds.column_names:
                eval_ds = eval_ds.remove_columns("info")
            per_env_eval.append((env_name, eval_ds))
            if args.write_per_env:
                _write_parquet(
                    eval_ds, os.path.join(output_dir, env_name, "validation.parquet")
                )
        else:
            print(f"WARNING: No eval examples for {env_name}. Skipping.")

    # Combine and shuffle
    if per_env_train:
        combined_train = concatenate_datasets([ds for _, ds in per_env_train])
        combined_train = combined_train.shuffle(seed=args.seed)
        _write_parquet(combined_train, os.path.join(output_dir, "train.parquet"))
        print(
            f"Wrote combined train to {os.path.join(output_dir, 'train.parquet')} with {len(combined_train)} rows"
        )
    else:
        print("INFO: No train datasets were available across environments.")

    if per_env_eval:
        combined_eval = concatenate_datasets([ds for _, ds in per_env_eval])
        combined_eval = combined_eval.shuffle(seed=args.seed)
        _write_parquet(combined_eval, os.path.join(output_dir, "validation.parquet"))
        print(
            f"Wrote combined validation to {os.path.join(output_dir, 'validation.parquet')} with {len(combined_eval)} rows"
        )
    else:
        print(
            "WARNING: No eval datasets were available across environments. Nothing to write."
        )
