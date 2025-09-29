from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Model paths
EXPERT_MODEL = "./Llama-3.1-8B"
AMATEUR_MODEL = "./Llama-3.2-1B"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizers
expert_tokenizer = AutoTokenizer.from_pretrained(EXPERT_MODEL)
amateur_tokenizer = AutoTokenizer.from_pretrained(AMATEUR_MODEL)

# Load models on GPU
expert_model = AutoModelForCausalLM.from_pretrained(EXPERT_MODEL, torch_dtype=torch.float16).to(device)
amateur_model = AutoModelForCausalLM.from_pretrained(AMATEUR_MODEL, torch_dtype=torch.float16).to(device)

# Extract shared vocabulary
expert_vocab = expert_tokenizer.get_vocab()
amateur_vocab = amateur_tokenizer.get_vocab()
shared_vocab = list(set(expert_vocab.keys()).intersection(amateur_vocab.keys()))

# Create index mappings for shared vocabulary
shared_vocab_dict = {token: i for i, token in enumerate(shared_vocab)}
expert_vocab_idx = torch.tensor([expert_vocab[tok] for tok in shared_vocab if tok in expert_vocab], device=device)
amateur_vocab_idx = torch.tensor([amateur_vocab[tok] for tok in shared_vocab if tok in amateur_vocab], device=device)

def map_logits_to_shared_vocab(logits, vocab_idx):
    """Map model's logits to the shared vocabulary space efficiently."""
    return logits[:, vocab_idx]

def compute_kl_divergence(logits_expert, logits_amateur, epsilon=1e-6):
    """Compute KL divergence between expert and amateur models over shared vocabulary."""
    
    # Align logits using tensor indexing (avoids slow loops)
    aligned_expert_logits = map_logits_to_shared_vocab(logits_expert, expert_vocab_idx)
    aligned_amateur_logits = map_logits_to_shared_vocab(logits_amateur, amateur_vocab_idx)

    # Normalize logits before applying softmax (avoids large exponentials)
    aligned_expert_logits = aligned_expert_logits - aligned_expert_logits.max(dim=-1, keepdim=True).values
    aligned_amateur_logits = aligned_amateur_logits - aligned_amateur_logits.max(dim=-1, keepdim=True).values

    # Convert logits to probability distributions
    probs_expert = F.softmax(aligned_expert_logits, dim=-1)
    probs_amateur = F.softmax(aligned_amateur_logits, dim=-1)

    # **Ensure numerical stability**
    probs_expert = torch.clamp(probs_expert, min=epsilon)
    probs_amateur = torch.clamp(probs_amateur, min=epsilon)

    # **Re-normalize** after clamping to ensure valid probability distribution
    probs_expert = probs_expert / probs_expert.sum(dim=-1, keepdim=True)
    probs_amateur = probs_amateur / probs_amateur.sum(dim=-1, keepdim=True)

    # Compute KL divergence using PyTorch's stable function
    kl_div = F.kl_div(probs_amateur.log(), probs_expert, reduction="batchmean")

    return kl_div


def generate_cot_and_compare_kl(prompt, max_steps=256):
    """Generates Chain of Thought reasoning while tracking KL divergence."""
    
    input_ids = expert_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    kl_divergences = torch.zeros(max_steps, device=device)  # Store KL values on GPU
    generated_text_expert = ""
    generated_text_amateur = []

    for i in range(max_steps):
        # Get logits from both models in one forward pass
        with torch.no_grad():
            expert_outputs = expert_model(input_ids)
            amateur_outputs = amateur_model(input_ids)
            
            expert_logits = expert_outputs.logits[:, -1, :]
            amateur_logits = amateur_outputs.logits[:, -1, :]

        # Compute KL divergence
        kl_divergences[i] = compute_kl_divergence(expert_logits, amateur_logits)

        # Select next token using expert model
        next_token_expert = torch.argmax(expert_logits, dim=-1)
        next_token_amateur = torch.argmax(amateur_logits, dim=-1)

        # Decode and update input
        next_word_expert = expert_tokenizer.decode(next_token_expert[0])
        generated_text_expert += " " + next_word_expert
        input_ids = torch.cat([input_ids, next_token_expert.unsqueeze(0)], dim=-1)

        next_word_amateur = expert_tokenizer.decode(next_token_amateur[0])
        generated_text_amateur.append(next_word_amateur)

    return generated_text_expert, generated_text_amateur, kl_divergences.cpu().tolist()


# Example usage
# math_problem = "A farmer has 12 apples. He gives 3 to his friend and buys 7 more. How many does he have now? Let's think step by step."
math_problem = "A concert ticket costs $40. Mr. Benson bought 12 tickets and received a 5 percent discount for every ticket bought that exceeds 10. How much did Mr. Benson pay in all? Solve this problem step by step."

generated_text_expert, generated_text_amateur, kl_values = generate_cot_and_compare_kl(math_problem)

print("\nGenerated Expert CoT:\n", generated_text_expert)
print("\nGenerated Amateur CoT:\n", generated_text_amateur)
print("\nKL Divergence Across Steps:\n", kl_values)
