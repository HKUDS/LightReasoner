import json

# === Case Study Paths (edit as needed) ===
# These two inputs should point to the JSONL evaluation outputs produced by the
# official Qwen2.5-Math evaluation toolkit for the BASE and the FINE‑TUNED models.
# Example patterns (adjust to your run structure):
# .../Qwen2.5-Math-1.5B/math_eval/gsm8k/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl
# .../ft_1.5_gsm8k_1000/math_eval/gsm8k/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl
base_model_path = "<path_to_base_model_eval_jsonl>"
fine_tuned_path = "<path_to_ft_model_eval_jsonl>"

# Output comparison file
output_path = "<path_to_save_comparison_txt>" # e.g., artifacts/improved_questions_comparison.txt


def load_report(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data[record["idx"]] = record
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping line {i+1} in {path} due to JSON error: {e}")
    return data

base_data = load_report(base_model_path)
fine_tuned_data = load_report(fine_tuned_path)

# === Write comparison text ===
with open(output_path, 'w', encoding='utf-8') as f:
    count = 0
    for idx in fine_tuned_data:
        base_entry = base_data.get(idx)
        ft_entry = fine_tuned_data[idx]
        
        base_score = base_entry.get("score", [False])[0] if base_entry else False
        ft_score = ft_entry["score"][0]

        if not base_score and ft_score:
            f.write("Base model:\n")
            f.write(json.dumps(base_entry, ensure_ascii=False, indent=2))
            f.write("\n\nFine-tuned model:\n")
            f.write(json.dumps(ft_entry, ensure_ascii=False, indent=2))
            f.write("\n\n" + "="*80 + "\n\n")
            count += 1

print(f"✅ Wrote {count} improved question comparisons to {output_path}")
