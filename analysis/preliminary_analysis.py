import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import argparse


# Optimize matmul precision for faster H100 execution
torch.set_float32_matmul_precision('high')

# Optional command-line argument
parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=100, help="Number of prompts to sample")
args = parser.parse_args()

# === Config ===
# Model paths (replace with your chosen models)
expert_model_path = "<path_to_expert_model>" # e.g., Qwen/Qwen2.5-Math-1.5B
amateur_model_path = "<path_to_amateur_model>" # e.g., Qwen/Qwen2.5-0.5B

# Data I/O paths
input_path = "<path_to_input_jsonl>" # e.g., data/gsm8k_train.jsonl
output_path = "<path_to_output_jsonl>" # e.g., artifacts/kl_analysis_log.jsonl

# Device and precision
device = "<device>" # e.g., "cuda", "cuda:0", or "cpu"
torch_dtype = "<torch_dtype>" # e.g., torch.bfloat16 for H100, torch.float16 for A100

# Sampling settings
max_new_tokens = 128


# === Load models & tokenizer ===
expert_model = AutoModelForCausalLM.from_pretrained(
    expert_model_path, torch_dtype=torch_dtype, device_map="auto"
).eval()
amateur_model = AutoModelForCausalLM.from_pretrained(
    amateur_model_path, torch_dtype=torch_dtype, device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(expert_model_path)

# === Load prompts ===
with open(input_path, "r", encoding="utf-8") as f:
    prompts = [json.loads(line) for line in f]
prompts = prompts[:args.num_samples]

# === Main process ===
log_records = []
sys_prompt = "Please reason step by step."

for prompt_obj in tqdm(prompts, desc="Processing prompts"):
    prompt_id = prompt_obj["id"]
    question = prompt_obj["question"]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = expert_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

    expert_targets = outputs.sequences[0][model_inputs.input_ids.shape[1]:].tolist()
    expert_probs = [F.softmax(score[0], dim=-1) for score in outputs.scores]
    expert_top1s = [torch.argmax(prob).item() for prob in expert_probs]

    for step in range(1, len(expert_targets)):
        prefix = tokenizer.decode(expert_targets[:step])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted += prefix

        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            logits = amateur_model(input_ids, attention_mask=attention_mask).logits
            next_token_logits = logits[0, -1, :]
            amateur_probs = F.softmax(next_token_logits, dim=-1)
            amateur_top1 = torch.argmax(amateur_probs).item()

        expert_prob = expert_probs[step]
        min_vocab = min(expert_prob.size(0), amateur_probs.size(0))

        # KL divergence
        kl_div = F.kl_div(
            amateur_probs[:min_vocab].log(),
            expert_prob[:min_vocab],
            reduction='sum'
        ).item()

        # Entropy values
        expert_entropy = -(expert_prob[:min_vocab] * expert_prob[:min_vocab].log()).sum().item()
        amateur_entropy = -(amateur_probs[:min_vocab] * amateur_probs[:min_vocab].log()).sum().item()

        # Token string and top-1 predictions
        token_id = expert_top1s[step]
        token_str = tokenizer.decode([token_id]).strip()
        expert_top1_str = tokenizer.decode([expert_top1s[step]]).strip()
        amateur_top1_str = tokenizer.decode([amateur_top1]).strip()

        log_records.append({
            "prompt_id": prompt_id,
            "step": step,
            "token": token_str,
            "kl_divergence": kl_div,
            "expert_entropy": expert_entropy,
            "amateur_entropy": amateur_entropy,
            "expert_top1_token": expert_top1_str,
            "amateur_top1_token": amateur_top1_str
        })

# Save results
with open(output_path, "w", encoding="utf-8") as f:
    for rec in log_records:
        f.write(json.dumps(rec) + "\n")

print(f"\nFinished logging {len(log_records)} token-level records to {output_path}")

