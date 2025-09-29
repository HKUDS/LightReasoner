from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

'''Do Expert and Amateur share the same tokenizer and vocabulary?'''

# Load tokenizers from both models
expert_tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-Math-7B")
amateur_tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B")

# Sanity check: do they tokenize the same sentence identically?
test_string = "The price is $40 after a 5 percent discount."

expert_tokens = expert_tokenizer.encode(test_string, add_special_tokens=False)
amateur_tokens = amateur_tokenizer.encode(test_string, add_special_tokens=False)

print("Same tokenizer behavior:", expert_tokens == amateur_tokens)


'''Can Qwen expose next-token prediction?'''

# # Load your Qwen model and tokenizer
# model_name = "./Qwen2.5-Math-7B"
# device = "cuda"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Example prefix
# prefix_text = "The price is"

# # Tokenize (without adding special tokens)
# inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).to(device)

# # Forward pass to get logits
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits  # Shape: [1, seq_len, vocab_size]

# # Get logits for the next token (i.e., after the last token in the prefix)
# next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

# # Convert to probabilities
# probs = F.softmax(next_token_logits, dim=-1)

# # Get top 5 most likely next tokens
# top_k = 5
# top_probs, top_indices = torch.topk(probs, top_k)

# print(f"Prefix: {prefix_text}")
# print("Top next-token predictions:")
# for i in range(top_k):
#     token_str = tokenizer.decode([top_indices[i].item()])
#     print(f"{token_str!r}: {top_probs[i].item():.4f}")


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"

# Load models
expert_model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-Math-7B", torch_dtype="auto", device_map="auto")
amateur_model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-0.5B", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-Math-7B")

# Test input
prefix_text = "Each ticket costs $40, so the total is"
target_token_text = " $"

# Tokenize
prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
target_token_id = tokenizer.encode(target_token_text, add_special_tokens=False)[0]
input_ids = torch.tensor([prefix_ids]).to(device)

def get_next_token_info(model, input_ids, target_token_id, top_k=5):
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, seq_len, vocab_size]
    next_token_logits = logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)

    target_prob = probs[target_token_id].item()

    # Get top-1 prediction
    top1_prob, top1_idx = torch.topk(probs, 1)
    top1_token = tokenizer.decode([top1_idx.item()])

    # (Optional) Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probs, top_k)
    top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices]

    return {
        "target_prob": target_prob,
        "top1_token": top1_token,
        "top1_prob": top1_prob.item(),
        "top_k": list(zip(top_k_tokens, top_k_probs.tolist()))
    }

# Run both models
expert_info = get_next_token_info(expert_model, input_ids, target_token_id)
amateur_info = get_next_token_info(amateur_model, input_ids, target_token_id)

# Print results
print(f"Prefix: {prefix_text!r}")
print(f"Target token: {target_token_text!r}")
print()

print("🔹 Expert Model")
print(f"Target token prob: {expert_info['target_prob']:.6f}")
print(f"Top-1 token: {expert_info['top1_token']!r} (prob = {expert_info['top1_prob']:.6f})")
print("Top-k predictions:")
for tok, prob in expert_info["top_k"]:
    print(f"  {tok!r}: {prob:.6f}")

print()

print("🔸 Amateur Model")
print(f"Target token prob: {amateur_info['target_prob']:.6f}")
print(f"Top-1 token: {amateur_info['top1_token']!r} (prob = {amateur_info['top1_prob']:.6f})")
print("Top-k predictions:")
for tok, prob in amateur_info["top_k"]:
    print(f"  {tok!r}: {prob:.6f}")

print()

print(f"🔀 Delta (Expert - Amateur) for target token: {expert_info['target_prob'] - amateur_info['target_prob']:.6f}")
