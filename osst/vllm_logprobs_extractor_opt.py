import torch
import logging
import os
import gc

from transformers import AutoModelForCausalLM
from vllm import LLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VLLMLogprobsExtractor:
    __DEFAULT_PARAMS = {
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.75,
        "async_scheduling": False,
    }
    __MODEL_SPECIFIC_OVERRIDES = {
        "mistralai/Mistral-Small-24B-Base-2501": {
            "gpu_memory_utilization": 0.98,
            # "max_num_seqs": 4,
        },
    }
    __LOGPROBS_BLOCK_SIZE = 512

    def __init__(self, model_id: str, torch_dtype: str = "auto", hf_token: str | None = None, max_model_len: int = 3584):
        num_available_gpus = torch.cuda.device_count()
        default_visible_devices = ",".join([str(i) for i in range(num_available_gpus)])
        prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", default_visible_devices)

        if num_available_gpus > 1 and (num_available_gpus % 2 == 1):
            logger.warning(
                f"Odd number of GPUs ({num_available_gpus}) detected. Reducing by 1 for tensor parallelism."
            )
            num_available_gpus -= 1

        params = self.__DEFAULT_PARAMS.copy()
        params.update(self.__MODEL_SPECIFIC_OVERRIDES.get(model_id, {}))
        params["tensor_parallel_size"] = num_available_gpus

        self.llm = LLM(
            model=model_id,
            task="reward",
            override_pooler_config=dict(pooling_type="ALL"),
            hf_token=hf_token,
            dtype=torch_dtype,
            max_model_len=max_model_len,
            **params,
        )

        self.last_device = f"cuda:{num_available_gpus - 1}"
        self.compute_logprobs_blockwise = True

        if params["gpu_memory_utilization"] > 0.9:
            # Load LM head on a separate GPU to reduce memory usage on other GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.__load_lm_head(model_id, torch_dtype, device="cuda")
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices
            self.compute_logprobs_blockwise = False  # In this case, logprobs are computed in their own GPU, no need to blockwise
        else:
            self.__load_lm_head(model_id, torch_dtype, device=self.last_device)



    @torch.no_grad()
    def get_logprobs_from_response(
        self, texts: list[str], response_template_tokens: list[int]
    ) -> list[list[float]]:
        outputs = self.llm.encode(texts)

        response_logprobs_list = []
        for o in outputs:
            last_hidden_states = o.outputs.data  # [L, D]
            token_ids = o.prompt_token_ids  # [L]

            # Only compute logprobs for the response part
            last_response_start = self._get_last_response_start(token_ids, response_template_tokens)
            if last_response_start == -1:
                logger.warning("Response template not found in token IDs.")
                response_logprobs_list.append([1.])  # Dummy logprob
                continue
            
            last_hidden_states = last_hidden_states[last_response_start:]  # [L', D]
            token_ids = token_ids[last_response_start:]  # [L']

            # Compute logprobs
            response_logprobs = self._get_logprobs(last_hidden_states, token_ids)
            response_logprobs = response_logprobs.cpu().numpy().tolist()
            response_logprobs_list.append(response_logprobs)

        return response_logprobs_list

    def _get_logprobs(
        self, hidden_states: torch.Tensor, token_ids: list[int]
    ) -> list[float]:
        hidden_states = hidden_states.to(self.lm_device, dtype=self.lm_dtype)
        block_size = self.__LOGPROBS_BLOCK_SIZE if self.compute_logprobs_blockwise else hidden_states.size(0)
        all_log_probs = []
        for start_idx in range(0, hidden_states.size(0), block_size):
            end_idx = min(start_idx + block_size, hidden_states.size(0))
            block_hidden_states = hidden_states[start_idx:end_idx]  # [B, D]
            block_token_ids = token_ids[start_idx:end_idx]  # [B]
            block_log_probs = self._compute_logprobs_block(block_hidden_states, block_token_ids)  # [B-1]
            all_log_probs.append(block_log_probs)
        return torch.cat(all_log_probs, dim=0)  # [L-1]

    def _compute_logprobs_block(
        self, hidden_states: torch.Tensor, token_ids: list[int]
    ) -> torch.Tensor:
        # Compute logits
        logits = self.lm_head(hidden_states).float()  # [L, V]

        # Compute logsumexp for normalization
        logsumexps = torch.logsumexp(logits, dim=-1)  # [L], computed in float32
        
        # Gather the logits for the following tokens
        following_tokens = torch.tensor(token_ids[1:]).unsqueeze(dim=-1).to(self.lm_device)  # [L-1, 1]
        gathered_logits = logits.gather(dim=-1, index=following_tokens).squeeze(-1)  # [L-1]
        gathered_log_probs = gathered_logits - logsumexps[:-1]  # [L-1]

        return gathered_log_probs

    def _get_last_response_start(
        self, token_ids: list[int], response_template_tokens: list[int]
    ) -> int:
        last_response_start = -1
        for possible_start in range(len(token_ids) - len(response_template_tokens) + 1):
            subseq = token_ids[possible_start : possible_start + len(response_template_tokens)]
            if subseq == response_template_tokens:
                last_response_start = possible_start + len(response_template_tokens)
        return last_response_start


    def __load_lm_head(self, model_id: str, torch_dtype: str, device: str):
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, device_map="cpu")
        self.lm_head = model.lm_head.to(device)
        self.lm_head.eval()
        del model
        gc.collect()

        self.lm_device = self.lm_head.weight.device
        self.lm_dtype = self.lm_head.weight.dtype