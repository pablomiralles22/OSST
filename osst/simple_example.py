import os
import torch
import numpy as np
from transformers import AutoTokenizer
from vllm_logprobs_extractor_opt import VLLMLogprobsExtractor

# Note: This script requires a GPU with vLLM installed and access to the model.
# It demonstrates the core scoring logic of OSST for a single candidate/target pair.

# 1. Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B"
# If you need an HF token, you can load it from environment variables or hardcode it here
HF_TOKEN = os.getenv("HF_READ_TOKEN") 

# Templates used in OSST
EXAMPLE_TEMPLATE = "Neutral text: {neutral_text}\nStyled text: {text}"
RESPONSE_TEMPLATE = "Styled text:"

def build_example(text, neutral_text):
    """Formats a text and its neutral version into the OSST template."""
    return EXAMPLE_TEMPLATE.format(text=text, neutral_text=neutral_text)

def main():
    # 2. Hardcoded examples
    # Candidate text (known author) and its neutralized version
    candidate_text = "The quick brown fox jumps over the lazy dog. It's a beautiful day in the neighborhood."
    neutral_candidate = "A fast brown fox leaps over a tired dog. The weather is nice nearby."

    # Target text (unknown author) and its neutralized version
    target_text = "The swift auburn fox leaps across the lethargic hound."
    neutral_target = "A fast brown fox jumps over a tired dog."

    print(f"--- Running example ---")
    print(f"Candidate: {candidate_text}")
    print(f"Target:    {target_text}")
    print()

    # 3. Setup Tokenizer and Extractor
    print(f"Loading tokenizer and extractor for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    response_template_tokens = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)

    # Initialize the extractor (VLLM-based)
    # This will load the model into GPU memory
    extractor = VLLMLogprobsExtractor(MODEL_ID, hf_token=HF_TOKEN)

    # 4. Build Prompts
    # Few-shot prompt: [Candidate Example] \n\n [Target Example]
    few_shot_prompt = "\n\n".join([
        build_example(candidate_text, neutral_candidate),
        build_example(target_text, neutral_target)
    ])

    # Zero-shot prompt: [Target Example] (baseline)
    zero_shot_prompt = build_example(target_text, neutral_target)

    # 5. Compute Logprobs
    # get_logprobs_from_response extracts logprobs for the tokens AFTER the last "Styled text:"
    prompts = [few_shot_prompt, zero_shot_prompt]
    logprobs_list = extractor.get_logprobs_from_response(prompts, response_template_tokens)

    few_shot_logprobs = logprobs_list[0]
    zero_shot_logprobs = logprobs_list[1]

    # 6. Calculate Scores
    # The OSST score is typically the average logprob of the target text
    few_shot_score = np.mean(few_shot_logprobs)
    zero_shot_score = np.mean(zero_shot_logprobs)

    print(f"\n--- Results ---")
    print(f"Few-shot Score (with candidate): {few_shot_score:.4f}")
    print(f"Zero-shot Score (baseline):       {zero_shot_score:.4f}")

if __name__ == "__main__":
    main()
