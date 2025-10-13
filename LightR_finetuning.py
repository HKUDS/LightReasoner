"""
==============================================================
 LightR Fine-tuning Script
==============================================================


This script combines all steps (dataset, LoRA, trainer, training loop)
into a single pipeline for fine-tuning with contrastive soft labels.

⚠️ IMPORTANT:
Before running, edit the config section in this file:
    - Replace <path_to_expert_model> with your base model path
      (e.g., "Qwen/Qwen2.5-Math-7B" or a local folder).
    - Replace <path_to_training_dataset> with your dataset JSONL file.
    - Replace <output_directory> with the directory where you want
      checkpoints and the final model to be saved.
    - Set torch_dtype according to your hardware
      (e.g., torch.bfloat16 for H100, torch.float16 for A100).

==============================================================
 How to Run
==============================================================

Basic run (foreground):
    python LightR_finetuning.py

Run with logging to file (background, recommended for long training):
    nohup python LightR_finetuning.py > finetune.log 2>&1 &

Monitor training progress:
    tail -f finetune.log

After training, the fine-tuned model will be saved under:
    <output_directory>   (as defined in the config)

==============================================================
"""


# ================================
# Fine-tuning step 1
# ================================
import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

class ContrastiveSoftLabelDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, model_vocab_size, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.vocab_size = model_vocab_size
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        assistant_prefix = item["prefix"]
        token_ids = item["token_ids"]
        weights = item["weights"]
        question = item["prompt_id"]

        # Apply chat template to build structured input
        messages = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": question}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_input = formatted + assistant_prefix

        encoding = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = torch.zeros(self.vocab_size, dtype=torch.float)
        for tid, weight in zip(token_ids, weights):
            if tid < self.vocab_size:
                labels[tid] = weight

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ================================
# Fine-tuning step 2
# ================================
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

def load_lora_model(model_path: str, torch_dtype, device_map="auto"):
    base_model = _AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    return get_peft_model(base_model, lora_config)


# ================================
# Fine-tuning step 3
# ================================
import torch.nn.functional as F
from transformers import Trainer

class SoftLabelKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        ).logits

        vocab_size = inputs["labels"].size(-1)
        logits = logits[:, -1, :vocab_size]  # [batch_size, vocab_size]

        log_probs = F.log_softmax(logits, dim=-1)
        soft_labels = inputs["labels"]

        loss = F.kl_div(log_probs, soft_labels, reduction="batchmean")
        return loss


# ================================
# Fine-tuning step 4 (main training)
# ================================
from transformers import TrainingArguments


# === Config (edit these before running) ===

# Model path
model_path = "<path_to_expert_model>"           # e.g., "Qwen/Qwen2.5-Math-7B" or local folder

# Dataset and output
dataset_path = "<path_to_training_dataset>"     # e.g., "./cd_dist_samples_gsm8k.jsonl"
output_dir   = "<output_directory>"             # e.g., "./finetuned_qwen2.5_cd_gsm8k"

# Device and precision
torch_dtype = "<torch_dtype>"                   # e.g., torch.bfloat16 for H100, torch.float16 for A100

# Training hyperparameters
batch_size = 8                                  # Per-device batch size (adjust for your GPU memory)
gradient_accumulation_steps = 2                 # Increase to simulate larger batches
eval_steps = 200                                # Run evaluation every N steps
save_steps = 200                                # Save checkpoint every N steps
logging_steps = 10                              # Log training progress every N steps
max_steps = 1000                                # Total training steps (set depending on dataset/experiment)
lr = 5e-5                                       # Learning rate


# === Global H100 Optimization ===
torch.set_float32_matmul_precision("high")

# === Load tokenizer and dataset ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
vocab_size = tokenizer.vocab_size
train_dataset = ContrastiveSoftLabelDataset(dataset_path, tokenizer, model_vocab_size=vocab_size)

# === Load LoRA-wrapped model ===
model = load_lora_model(
    model_path=model_path,
    torch_dtype=torch_dtype,
    device_map="auto"
)

# === Data Collator ===
def collate_fn(batch):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        ),
        "labels": torch.stack([x["labels"] for x in batch])
    }

# === TrainingArguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    max_steps=max_steps,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    report_to="none",
    disable_tqdm=False,
    remove_unused_columns=False
)

# === Trainer ===
trainer = SoftLabelKLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn
)

# === Train ===
if __name__ == "__main__":
    print("🚀 Starting full fine-tuning on GSM8K contrastive samples...")
    trainer.train()
    print("✅ Fine-tuning complete!")
