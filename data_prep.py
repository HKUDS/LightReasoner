'''GSM8K'''

import json
from datasets import load_dataset

# === Step 1: Load GSM8K dataset ===
gsm8k = load_dataset("openai/gsm8k", "main")

# === Step 2: Save Training Set with answers ===
with open("gsm8k_train.jsonl", "w", encoding="utf-8") as f_train:
    for ex in gsm8k["train"]:
        json.dump({
            "id": ex["question"],  # Use question as unique ID
            "question": ex["question"],
            "answer": ex["answer"]
        }, f_train)
        f_train.write("\n")
print(f"✅ Saved {len(gsm8k['train'])} training examples to gsm8k_train.jsonl")

# === Step 3: Save Test Set ===
with open("gsm8k_test.jsonl", "w", encoding="utf-8") as f_test:
    for ex in gsm8k["test"]:
        json.dump({
            "id": ex["question"],  # Use question as unique ID
            "question": ex["question"],
            "answer": ex["answer"]
        }, f_test)
        f_test.write("\n")
print(f"✅ Saved {len(gsm8k['test'])} test examples to gsm8k_test.jsonl")




'''MATH'''

from datasets import load_dataset
import json
from tqdm import tqdm

# Subjects to load
subjects = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus"
]

# Prepare storage
train_examples = []
test_examples = []

# Load each subject and gather examples
for subject in subjects:
    print(f"🔵 Loading subject: {subject}")
    dataset = load_dataset("EleutherAI/hendrycks_math", subject)
    
    # Add subject field manually, since they don't have it
    for ex in tqdm(dataset["train"], desc=f"Processing train split ({subject})"):
        train_examples.append({
            "id": ex["problem"],  # use problem as unique ID
            "problem": ex["problem"],
            "solution": ex["solution"],
            "level": subject  # mark where it comes from
        })
        
    for ex in tqdm(dataset["test"], desc=f"Processing test split ({subject})"):
        test_examples.append({
            "id": ex["problem"],
            "problem": ex["problem"],
            "solution": ex["solution"],
            "level": subject
        })

# === Save Training Set ===
with open("math_train.jsonl", "w", encoding="utf-8") as f_train:
    for ex in train_examples:
        json.dump(ex, f_train)
        f_train.write("\n")
print(f"✅ Saved {len(train_examples)} training examples to math_train.jsonl")

# === Save Test Set ===
with open("math_test.jsonl", "w", encoding="utf-8") as f_test:
    for ex in test_examples:
        json.dump(ex, f_test)
        f_test.write("\n")
print(f"✅ Saved {len(test_examples)} test examples to math_test.jsonl")

