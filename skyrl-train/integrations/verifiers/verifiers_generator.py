from typing import Optional, Dict, List
import asyncio
from skyrl_train.generators.base import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
)
from omegaconf import DictConfig
from openai import AsyncOpenAI
import httpx
from verifiers import load_environment
from verifiers.types import GenerateOutputs, ProcessedOutputs, GenerateInputs
from skyrl_train.generators.utils import get_rollout_metrics


class VerifiersGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._env_cache: Dict[str, object] = {}

        assert generator_cfg.enable_http_endpoint, (
            "HTTP endpoint must be enabled for VerifiersGenerator"
        )
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}/v1"
        self.client = self._setup_client(
            connection_limit=None
        )  # None means unlimited connections

    def _setup_client(self, connection_limit: Optional[int]) -> AsyncOpenAI:
        timeout = httpx.Timeout(timeout=600, connect=5.0)
        limits = httpx.Limits(
            max_connections=connection_limit,  # OAI default: 1000
            max_keepalive_connections=connection_limit,  # OAI default: 100
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key="dummy",  # Make OAI client happy.
            max_retries=10,  # OAI default: 2
            http_client=http_client,
        )

    def _get_env(self, env_name: str):
        if env_name not in self._env_cache:
            self._env_cache[env_name] = load_environment(env_name)
        return self._env_cache[env_name]

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        assert "env_extras" in input_batch, (
            "Verifiers dataset fields are passed through env_extras"
        )

        # Extract per-sample verifiers config
        verifiers_dicts: List[Dict] = [
            sample["verifiers"] for sample in input_batch["env_extras"]
        ]

        # Group indices by environment
        env_to_indices: Dict[str, List[int]] = {}
        for idx, v in enumerate(verifiers_dicts):
            env_name = v.get("environment")
            if not env_name:
                raise AssertionError("Each sample must include verifiers.environment")
            env_to_indices.setdefault(env_name, []).append(idx)

        # Verifiers requires logprobs from vLLM for post-processing.
        sampling_params = input_batch.get("sampling_params", {}).copy()
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = 1
        sampling_params["extra_body"] = {
            "return_tokens_as_token_ids": True,
        }

        # Clean the sampling params for Verifiers' a_generate.
        extra_body_keys = [
            "min_tokens",
            "skip_special_tokens",
            "include_stop_str_in_output",
            "top_k",
            "min_p",
            "repetition_penalty",
        ]
        for key in extra_body_keys:
            if key in sampling_params:
                sampling_params["extra_body"][key] = sampling_params[key]
                del sampling_params[key]

        # Prepare per-environment GenerateInputs and async calls
        tasks = []
        task_meta = []  # (env_name, indices, vf_env)
        for env_name, indices in env_to_indices.items():
            vf_env = self._get_env(env_name)
            env_inputs = GenerateInputs(
                prompt=[input_batch["prompts"][i] for i in indices],
                answer=[verifiers_dicts[i].get("answer", "") for i in indices],
                info=[verifiers_dicts[i].get("info", {}) for i in indices],
                task=[verifiers_dicts[i].get("task", "default") for i in indices],
            )
            tasks.append(
                vf_env.a_generate(
                    inputs=env_inputs,
                    client=self.client,
                    model=self.model_name,
                    sampling_args=sampling_params,
                )
            )
            task_meta.append((env_name, indices, vf_env))

        # Run all environments concurrently
        per_env_generate_outputs: List[GenerateOutputs] = await asyncio.gather(*tasks)

        # Allocate final outputs aligned with original batch order
        batch_size = len(input_batch["prompts"])
        prompt_token_ids: List[List[int]] = [None] * batch_size  # type: ignore
        response_ids: List[List[int]] = [None] * batch_size  # type: ignore
        rewards: List[float] = [0.0] * batch_size
        loss_masks: List[List[int]] = [None] * batch_size  # type: ignore
        rollout_logprobs: List[List[float]] = [None] * batch_size  # type: ignore

        max_seq_len = (
            self.generator_cfg.max_input_length
            + self.generator_cfg.sampling_params.max_generate_length
        )

        # Process per-environment results and place them back into position
        for (env_name, indices, vf_env), gen_out in zip(
            task_meta, per_env_generate_outputs
        ):
            processed: ProcessedOutputs = vf_env.process_env_results_vllm(
                prompts=gen_out.prompt,
                completions=gen_out.completion,
                states=gen_out.state,
                rewards=gen_out.reward,
                processing_class=self.tokenizer,
                max_seq_len=max_seq_len,
                mask_env_responses=True,
            )

            # Fill back in order
            for local_idx, global_idx in enumerate(indices):
                prompt_token_ids[global_idx] = processed.prompt_ids[local_idx]
                response_ids[global_idx] = processed.completion_ids[local_idx]
                rewards[global_idx] = processed.rewards[local_idx]
                loss_masks[global_idx] = processed.completion_mask[local_idx]
                rollout_logprobs[global_idx] = processed.completion_logprobs[local_idx]

        # Convert output to SkyRL format.
        return GeneratorOutput(
            prompt_token_ids=prompt_token_ids,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            rollout_logprobs=rollout_logprobs,
            rollout_metrics=get_rollout_metrics(response_ids, rewards),
        )
