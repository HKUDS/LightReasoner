import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import log
from tqdm import tqdm
import json
import argparse
import os


# === Global Precision Optimization ===
torch.set_float32_matmul_precision('high')  # Unlock fast TF32 matmuls on H100

# === Command-line arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--max_questions", type=int, default=None, help="Maximum number of prompts to process")
args = parser.parse_args()

# === Config ===
# Model paths (replace with your chosen models)
expert_model_path = "<path_to_expert_model>" # e.g., Qwen/Qwen2.5-Math-7B
amateur_model_path = "<path_to_amateur_model>" # e.g., Qwen/Qwen2.5-0.5B

# Device and precision
device = "<device>" # e.g., "cuda", "cuda:0", or "cpu"
torch_dtype = "<torch_dtype>" # e.g., torch.bfloat16 for H100, torch.float16 for A100

# Sampling settings
max_new_tokens = 128
alpha = 0.2 # Plausibility threshold factor (tune per experiment)
beta = 0.4 # KL divergence threshold (tune per experiment)

# Data I/O paths
input_path = "<path_to_input_jsonl>" # e.g., data/gsm8k_train.jsonl
output_path = "<path_to_output_jsonl>" # e.g., artifacts/cd_samples.jsonl
checkpoint_path = "<path_to_checkpoint_jsonl>" # e.g., artifacts/checkpoints/cd_checkpoint.jsonl

# Batch size (adjust based on GPU; 64 is H100-optimized)
batch_size = 64


# === Load Models and Tokenizer ===
expert_model = AutoModelForCausalLM.from_pretrained(
    expert_model_path,
    torch_dtype=torch_dtype,
    device_map="auto"
).eval()

amateur_model = AutoModelForCausalLM.from_pretrained(
    amateur_model_path,
    torch_dtype=torch_dtype,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(expert_model_path)

# === Load Prompts ===
with open(input_path, "r", encoding="utf-8") as f:
    prompts = [json.loads(line) for line in f]

if args.max_questions:
    prompts = prompts[:args.max_questions]

# === Load Checkpoint (if exists) ===
processed_ids = set()
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed_ids.add(record["prompt_id"])
            except:
                continue

# === Run CDRS sampling over prompts ===
sampled_dataset = []

for i, prompt_obj in enumerate(tqdm(prompts, desc="Processing prompts")):
    prompt_id = prompt_obj.get("id", i)
    if prompt_id in processed_ids:
        continue

    question = prompt_obj["question"]
    sys_prompt = "Please reason step by step."

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    outputs = expert_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True
    )

    expert_targets = outputs.sequences[0][model_inputs.input_ids.shape[1]:].tolist()
    expert_probs = [F.softmax(score[0], dim=-1).cpu() for score in outputs.scores]

    # === Optimization: Single Forward Pass for Amateur Model ===
    # Instead of N forward passes for N tokens, we concatenate the prompt and the full expert response
    # and run a single forward pass. Causal masking ensures that at each position t, 
    # the model only sees tokens [0...t], effectively predicting t+1.
    
    # 1. Reconstruct the base prompt (without the expert's generation)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_ids = prompt_inputs.input_ids[0]
    
    # 2. Concatenate Prompt + Expert Response
    expert_ids_tensor = torch.tensor(expert_targets, device=device, dtype=torch.long)
    full_input_ids = torch.cat([prompt_ids, expert_ids_tensor], dim=0).unsqueeze(0) # [1, Seq_Len]
    
    # 3. Single Forward Pass
    with torch.no_grad():
        outputs = amateur_model(full_input_ids)
        logits = outputs.logits # [1, Seq_Len, Vocab]
        
        # 4. Extract relevant logits
        # We need predictions for the expert's tokens.
        # The logit at index `i` predicts the token at `i+1`.
        # Prompt length is P. Expert tokens start at P.
        # We need logits from index P-1 (predicting Expert[0]) to P+E-2 (predicting Expert[E-1]).
        
        start_idx = prompt_ids.size(0) - 1
        end_idx = full_input_ids.size(1) - 1
        
        relevant_logits = logits[:, start_idx:end_idx, :] # [1, Expert_Len, Vocab]
        amateur_probs_sequence = F.softmax(relevant_logits, dim=-1).cpu() # Move to CPU immediately

    samples_this_prompt = 0
    for step in range(1, len(expert_targets)):
        expert_dist = expert_probs[step]
        amateur_dist = amateur_probs_sequence[0, step, :]

        min_vocab_size = min(expert_dist.size(0), amateur_dist.size(0))
        expert_dist = expert_dist[:min_vocab_size]
        amateur_dist = amateur_dist[:min_vocab_size]

        plaus_thresh = alpha * torch.max(expert_dist)
        mask = expert_dist >= plaus_thresh
        if not mask.any():
            continue

        # P_E_V = expert_dist[mask]
        # P_A_V = amateur_dist[mask]

        # P_E_V /= P_E_V.sum()
        # P_A_V /= P_A_V.sum()

        # kl_div = F.kl_div(P_A_V.log(), P_E_V, reduction='sum').item()

        # Full-vocab KL divergence
        epsilon = 1e-12
        P_E = expert_dist + epsilon
        P_A = amateur_dist + epsilon

        kl_div = F.kl_div(P_A.log(), P_E, reduction='sum').item()

        if kl_div < beta:
            continue

        log_PE = torch.log(expert_dist + 1e-12)
        log_PA = torch.log(amateur_dist + 1e-12)
        cd_scores_full = log_PE - log_PA
        cd_scores_V = cd_scores_full[mask]
        weights = torch.softmax(cd_scores_V, dim=-1)

        token_ids = torch.arange(expert_dist.size(0), device=expert_dist.device)[mask].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        sampled_dataset.append({
            "prompt_id": prompt_id,
            "step": step,
            "prefix": tokenizer.decode(expert_targets[:step]),
            "tokens": token_strs,
            "token_ids": token_ids,
            "weights": weights.tolist(),
            "kl_divergence": kl_div
        })
        samples_this_prompt += 1

    # Save checkpoint log entry
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        json.dump({"prompt_id": prompt_id, "num_steps": len(expert_targets), "num_samples": samples_this_prompt}, f)
        f.write("\n")

# === Save Final Sample Dataset ===
with open(output_path, "w", encoding="utf-8") as f:
    for ex in sampled_dataset:
        json.dump(ex, f)
        f.write("\n")

print(f"\nFinished. Saved {len(sampled_dataset)} contrastive distributional samples to {output_path}")
