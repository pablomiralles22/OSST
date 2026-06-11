# OSST: One-Shot Style Transfer for Attribution

OSST is a method for identifying the author of a text by comparing its "style" to known examples using LLM logprobs.

## Key Scripts

- `osst/simple_example.py`: A simple, hardcoded example showing how to compute attribution scores for a single pair of texts.
- `osst/vllm_logprobs_extractor_opt.py`: Optimized class for extracting token-level logprobs using vLLM.
- `osst/neutralizer.py`: Logic for "neutralizing" text (removing style) to create baseline comparisons.

## How it Works

1. **Neutralize**: Convert both "candidate" and "target" texts into a neutral, simple version.
2. **Prompt**: Build a prompt: `Neutral text: <neutral> \n Styled text: <original>`.
3. **Score**: Use an LLM to calculate the average logprob of the target text.
4. **Compare**: A higher score when conditioned on a specific candidate suggests that candidate is likely the author.

## Setup

1. Install dependencies: `uv sync`.
2. Set your HuggingFace token: `export HF_READ_TOKEN=your_token_here` or write it to the `.env` file.
3. Run the example: `uv run osst/simple_example.py`.
