import os
import torch
import math
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import json



# === CONFIG (edit as needed) ===
BASE_MODEL_PATH = "<path_to_base_model>" # e.g., ./Qwen2.5-Math-1.5B
CHECKPOINT_DIR = "<path_to_lora_checkpoint>" # e.g., ./ft_qwen2.5_math/checkpoints

# Device & dtype
DEVICE = "<device>" # e.g., "cuda", "cuda:0", or "cpu"
DTYPE = "<torch_dtype>" # e.g., torch.bfloat16 for H100, torch.float16 for A100

# Evaluation settings
EVAL_BATCH_SIZE = 4 # adjust per GPU
MAX_LENGTH = 512
SYS_PROMPT = "Think step by step."

# Data path
gsm8k_path = "<path_to_eval_jsonl>" # e.g., data/gsm8k_test.jsonl


def load_gsm8k_eval_samples(path, limit=None):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            question = item["question"].strip()

            # Extract and clean the final answer
            if "####" in item["answer"]:
                gold_answer = item["answer"].split("####")[-1].strip()
            else:
                gold_answer = item["answer"].strip()

            # Normalize dollar signs and spaces
            gold_answer = gold_answer.lstrip("$").strip()

            samples.append((question, gold_answer))

    # === Print verification ===
    # print("✅ Loaded GSM8K Evaluation Samples:")
    # for i in range(min(5, len(samples))):
    #     q, a = samples[i]
    #     print(f"\nQ{i+1}: {q}\nA{i+1}: {a}")
    # print(f"\nTotal samples loaded: {len(samples)}\n")
    
    return samples

# Load 50 samples for evaluation
eval_samples = load_gsm8k_eval_samples(gsm8k_path, limit=10)


class PPLEvalDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        question, answer = self.samples[idx]
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": question}
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + answer

        full_enc = self.tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt")

        input_ids = full_enc.input_ids.squeeze(0)
        attention_mask = full_enc.attention_mask.squeeze(0)

        assistant_start = prompt_enc.input_ids.shape[-1]
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[assistant_start:] = input_ids[assistant_start:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch):
    return {
        "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "labels": pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100),
    }

def compute_ppl(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()

    ppl = math.exp(total_loss / total_tokens)
    return ppl

# === MAIN SCRIPT ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
results = []

for ckpt in sorted(os.listdir(CHECKPOINT_DIR)):
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
    if not os.path.isdir(ckpt_path) or not ckpt.startswith("checkpoint-"):
        continue

    print(f"\n🔄 Merging and Evaluating {ckpt}...")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=DTYPE,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model = model.merge_and_unload()
    except Exception as e:
        print(f"❌ Failed to load and merge {ckpt}: {e}")
        continue

    dataset = PPLEvalDataset(eval_samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn)
    ppl = compute_ppl(model, dataloader)

    step = int(ckpt.split("-")[1])
    results.append((step, ppl))
    print(f"✅ PPL at step {step}: {ppl:.2f}")
