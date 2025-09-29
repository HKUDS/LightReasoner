import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── 2.1 Load GSM8K test set ─────────────────────────────────────────
# GSM8K has ~1.3K test examples of grade‐school word problems :contentReference[oaicite:6]{index=6}
dataset = load_dataset("openai/gsm8k", split="test")

# ─── 2.2 Initialize Qwen2.5-Math‑7B‑Instruct ────────────────────────
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
device = "cuda" # the device to load the model onto
tokenizer  = AutoTokenizer.from_pretrained(model_name)  # handles tokenization :contentReference[oaicite:7]{index=7}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

'''
model      = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,      # required for Qwen custom layers :contentReference[oaicite:8]{index=8}
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()
'''
# ─── 2.3 Extraction Helper ──────────────────────────────────────────
def extract_answer(text: str) -> str:
    """
    1) If the model wraps its answer in \boxed{...}, pull out the inner content.
    2) Otherwise, fall back to grabbing the last numeric token.
    """
    # Try \boxed{…} first (non‐greedy inside braces) :contentReference[oaicite:9]{index=9}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # Fallback: capture all numbers (ints or decimals) and return the last one :contentReference[oaicite:10]{index=10}
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""
#Function Validated

# ─── 2.4 Inference + Accuracy ────────────────────────────────────────
correct = 0
wrong = []
for ex in dataset:
    prompt = ex['question']                    # GSM8K prompt format :contentReference[oaicite:11]{index=11}
    
    messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    '''    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    '''    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #Below is untested, I don't know what GSM8k dataset looks like
    pred_ans = extract_answer(response)
    gold_ans = ex["answer"].strip().split()[-1]             # GSM8K gold ends in "#### <num>"
    if pred_ans == gold_ans:
        correct += 1
    else:
        wrong.append((prompt, response, pred_ans, gold_ans))

accuracy = correct / len(dataset)
print(f"GSM8K Accuracy with Qwen/Qwen2.5-Math-7B-Instruct: {accuracy:.2%}")
wrong = open("wrong_answers.txt", "x")