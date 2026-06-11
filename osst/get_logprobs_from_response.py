import torch

@torch.no_grad()
def get_logprobs_from_response(texts, tokenizer, model, response_template_tokens):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    inputs = inputs.to("cuda")

    # Generate logits for the input tokens
    batch_logits = model(**inputs, return_dict=True).logits  # [B, L, V]

    response_log_probs_list = []
    for input_ids, logits in zip(inputs.input_ids, batch_logits):
        # Remove padding and eos tokens from consideration
        mask = (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.eos_token_id)
        input_ids = input_ids[mask]  # [L]
        logits = logits[mask]  # [L, V]

        # Find the start index of the last occurrence of the response template
        last_response_start = get_last_response_start(
            input_ids.cpu().numpy().tolist(),
            response_template_tokens,
        )

        if last_response_start == -1:
            print("No response template found in input_ids.")
            response_log_probs_list.append([1.0])  # This way we know this failed, as logprobs will be negative
            continue

        # Restrict to tokens after the response template and cast to float32 for numerical stability
        input_ids = input_ids[last_response_start:]  # [L']
        logits = logits[last_response_start:].float()  # [L', V]

        # Compute logsumexp for normalization
        logsumexps = torch.logsumexp(logits, dim=-1)  # [L], computed in float32
        
        # Gather the logits for the following tokens
        following_tokens = input_ids[1:]  # [L-1]
        gathered_logits = (
            logits.gather(dim=-1, index=following_tokens.unsqueeze(-1)).squeeze(-1)
        )  # [L-1]
        gathered_log_probs = gathered_logits - logsumexps[:-1]  # [L-1]

        # Extract the logits for the response template tokens
        response_log_probs_list.append(gathered_log_probs.cpu().numpy().tolist())
        
    return response_log_probs_list

def get_last_response_start(token_ids, response_template_tokens):
    last_response_start = -1
    for possible_start in range(len(token_ids) - len(response_template_tokens) + 1):
        subseq = token_ids[possible_start : possible_start + len(response_template_tokens)]
        if subseq == response_template_tokens:
            last_response_start = possible_start + len(response_template_tokens)
    return last_response_start