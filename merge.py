'''Use this to merge the FULL model (base + LoRA) locally so that it behaves like a standalone model with no LoRA dependency.'''

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Config ===
base_model_path = "<path_to_base_model>" # e.g., ./Qwen2.5-Math-7B
lora_ckpt_path = "<path_to_lora_checkpoint>" # e.g., ./ft_qwen2.5_gsm8k/checkpoint-1000
merged_model_path = "<path_to_save_merged_model>" # e.g., ./ft-7B-merged

# === Step 1: Load base model and LoRA adapter ===
print("🔧 Loading base + LoRA model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_ckpt_path)

# === Step 2: Merge weights and unload PEFT structure ===
print("🔗 Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

# === Step 3: Save merged model and tokenizer ===
print(f"💾 Saving merged model to: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)

print("💾 Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)


print("✅ Merge complete! Model saved locally.")
