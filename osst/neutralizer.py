import logging
import torch

from typing import List, Optional, Tuple, Union
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "sp": "Spanish",
    "it": "Italian",
    "pl": "Polish",
}

PROMPT_NEUTRAL = '''
Text: "{text}"
---
Paraphrase the previous text using the driest, simplest, and most neutral style possible.
Remove all stylistic features, emotion, or rhetorical devices. Replace as many words as possible with simpler or more common alternatives.
Preserve as much of the original meaning as possible. Write in {language}. Output only the rewritten version.
'''.strip()

PROMPT_PARAPHRASE = '''
Text: "{text}"
---
Paraphrase the previous text in a natural way, preserving the original meaning but changing the wording and structure. Write in {language}. Output only the rewritten version.
'''.strip()

PROMPT_SHAKESPEARE = '''
Text: "{text}"
---
Paraphrase the previous text in the style of Shakespeare, using archaic language and poetic devices. Preserve the original meaning but rewrite it in a way that evokes the tone and style of Shakespeare's works. Write in {language}. Output only the rewritten version.
'''.strip()

PROMPT_MAP = {
    "neutral": PROMPT_NEUTRAL,
    "paraphrase": PROMPT_PARAPHRASE,
    "shakespeare": PROMPT_SHAKESPEARE,
}


class Neutralizer:
    __DEFAULT_VLLM_PARAMS = {
        "max_model_len": 8192,
        "enforce_eager": True,
    }
    def __init__(self, model_name="google/gemma-3-12b-it", min_chars=20, prompt_type="neutral", **llm_kwargs):
        self.model_name = model_name
        # Initialize local vLLM runner (offline inference)
        # llm_kwargs can pass device / tensor parallel configs as needed
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus > 1 and (num_available_gpus % 2 == 1):
            logger.warning(
                f"Odd number of GPUs ({num_available_gpus}) detected. Reducing by 1 for tensor parallelism."
            )
            num_available_gpus -= 1
        self.llm_kwargs = {
            "tensor_parallel_size": num_available_gpus,
            **self.__DEFAULT_VLLM_PARAMS,
            **llm_kwargs,
        }
        # self.llm = LLM(model=self.model_name, **self.llm_kwargs)
        self.llm = None
        self.sampling_params = SamplingParams(max_tokens=1024)
        self.min_chars = min_chars
        self.prompt = PROMPT_MAP.get(prompt_type, PROMPT_NEUTRAL)

    def neutralize(self, texts: List[Union[str, Tuple[str, str]]]) -> List[Optional[str]]:
        """
        Neutralize a list of texts using the specified model.
        Accepts a list of (text, lang) tuples, where lang is 'en', 'fr', 'it', or 'es'.
        Returns a list of neutralized texts.
        """
        if not texts:
            return []

        if self.llm is None:
            # Lazy initialization of the LLM to avoid the overhead if not used
            self.llm = LLM(model=self.model_name, **self.llm_kwargs)

        # If user passes just a list of strings, default to English
        texts = [
            (t, "en") if isinstance(t, str) else t
            for t in texts
        ]

        # Build prompts for those that should be neutralized; preserve order
        prompts: List[str] = []
        indices: List[int] = []
        for i, (t, lang) in enumerate(texts):
            if self._should_neutralize(t):
                prompts.append(self._build_prompt(input=t, lang=lang))
                indices.append(i)
            else:
                logger.warning(f"Skipping neutralization for text: {t}")

        outputs: List[Optional[str]] = [None] * len(texts)
        if prompts:
            try:
                generations = self.llm.generate(prompts, self.sampling_params)
                for gen, idx in zip(generations, indices):
                    # vLLM returns a list of `Generation` objects per prompt
                    if not gen.outputs:
                        logger.warning("Empty generation outputs for a prompt")
                        outputs[idx] = None
                        continue
                    content = gen.outputs[0].text or ""
                    outputs[idx] = self.__flatten_text(content)
            except Exception as e:
                logger.error(f"Error during neutralization with vLLM: {e}")
                # In case of error, return None for all requested indices
                for idx in indices:
                    outputs[idx] = None

        return outputs

    def _build_prompt(self, input, lang="en"):
        """
        Build a single-string prompt for vLLM generate.
        """
        # Map 'sp' to Spanish if provided
        lang = "es" if lang == "sp" else lang
        full_language = LANGUAGES.get(lang, "English")
        return self.prompt.format(text=input, language=full_language)

    def _should_neutralize(self, text):
        """Check if the text should be neutralized."""
        if not any(char.isalpha() for char in text):
            return False
        if len(text) < self.min_chars:
            logging.warning(f"Text too short to neutralize: {text}")
            return False
        return True

    def __flatten_text(self, text):
        """Flatten the text to a single line."""
        return " ".join(
            p for p in text.split()
            if any(char.isalpha() for char in p)
        )
