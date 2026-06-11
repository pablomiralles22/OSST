import torch
import logging

from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VLLMLogprobsExtractor:
    __DEFAULT_PARAMS = {
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.65,
        "async_scheduling": False,
    }
    __MODEL_SPECIFIC_OVERRIDES = {
        "meta-llama/Llama-3.1-8B": { "gpu_memory_utilization": 0.75, },
        "mistralai/Mistral-Small-24B-Base-2501": {
            "gpu_memory_utilization": 0.9,
            "cpu_offload_gb": 3,
        },
    }

    def __init__(self, model_id: str, torch_dtype: str = "auto", hf_token: str | None = None, max_model_len: int = 2048):
        num_available_gpus = torch.cuda.device_count()

        params = self.__DEFAULT_PARAMS.copy()
        params.update(self.__MODEL_SPECIFIC_OVERRIDES.get(model_id, {}))
        if (num_available_gpus % 2 == 1) and (num_available_gpus > 1):
            params["pipeline_parallel_size"] = num_available_gpus
            params["enable_prefix_caching"] = False
        else:
            params["tensor_parallel_size"] = num_available_gpus

        self.llm = LLM(model=model_id, hf_token=hf_token, dtype=torch_dtype, max_model_len=max_model_len, **params)


    def get_logprobs_from_response(
        self, texts: list[str], response_template_tokens: list[int]
    ) -> list[list[float]]:
        sampling_params = SamplingParams(
            prompt_logprobs=0, max_tokens=1,
            # Disable sampling modifiers for logprob extraction
            temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
            # Disable penalties
            presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
        )
        outputs = self.llm.generate(texts, sampling_params)

        response_logprobs_list = []
        for o in outputs:
            assert len(o.prompt_token_ids) == len(o.prompt_logprobs), f"{len(o.prompt_token_ids)=} != {len(o.prompt_logprobs)=}"
            assert (o.prompt_logprobs[0] is None), "The first token should have no logprob"
            assert all(len(lp) == 1 for lp in o.prompt_logprobs[1:]), "All other tokens should have one logprob"

            token_ids = o.prompt_token_ids
            logprobs = [next(iter(lp.values())).logprob for lp in o.prompt_logprobs[1:]]

            last_response_start = -1
            for possible_start in range(len(token_ids) - len(response_template_tokens) + 1):
                subseq = token_ids[possible_start : possible_start + len(response_template_tokens)]
                if subseq == response_template_tokens:
                    last_response_start = possible_start + len(response_template_tokens)

            if last_response_start == -1:
                logger.warning("No response template found in token_ids.")
                continue

            response_logprobs_list.append(logprobs[last_response_start:])

        return response_logprobs_list
