"""Analyze KL logs to produce supporting stats + a KL histogram.
Edit placeholders below to point to your log and tweak thresholds.
"""


import json
from collections import defaultdict
from math import floor
from statistics import mean
from typing import List

# =====================
# Config (edit as needed)
# =====================
input_path = "<path_to_kl_log_jsonl>"   # e.g., artifacts/kl_analysis_log.jsonl

# Thresholds for analysis claims
expert_confident_H = 1.0   # Expert entropy < 1.0 ⇒ confident
amateur_uncertain_H = 3.0  # Amateur entropy > 3.0 ⇒ uncertain
high_kl_threshold = 1.0    # KL tail cutoff for strong signal

# Histogram settings
bin_width = 0.1
max_bin = 5.0              # values ≥ max_bin go to overflow bin "max_bin+"

# =====================
# Helpers
# =====================
def _safe_avg(values: List[float]) -> float:
    return mean(values) if values else float("nan")

# =====================
# Load records once
# =====================
with open(input_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

total = len(records)

# =====================
# Claim-supporting statistics
# =====================
top1_match_kl: List[float] = []
top1_mismatch_kl: List[float] = []
confidence_mismatch_kl: List[float] = []
high_kl_tokens: List[float] = []

for r in records:
    kl = r["kl_divergence"]
    expert_top1 = r["expert_top1_token"]
    amateur_top1 = r["amateur_top1_token"]
    expert_entropy = r["expert_entropy"]
    amateur_entropy = r["amateur_entropy"]

    # 1) Match/mismatch of top‑1 tokens
    (top1_match_kl if expert_top1 == amateur_top1 else top1_mismatch_kl).append(kl)

    # 2) Confidence mismatch (expert confident, amateur uncertain)
    if expert_entropy < expert_confident_H and amateur_entropy > amateur_uncertain_H:
        confidence_mismatch_kl.append(kl)

    # 3) High‑KL tail (strong signal steps)
    if kl > high_kl_threshold:
        high_kl_tokens.append(kl)

print("==== Supporting Statistics ====")

# 1) Top‑1 match vs mismatch
match_ratio = (len(top1_match_kl) / total * 100) if total else 0.0
mismatch_ratio = (len(top1_mismatch_kl) / total * 100) if total else 0.0
print(f"🔹 Top-1 Match:    {match_ratio:.2f}% (Avg KL = {_safe_avg(top1_match_kl):.3f})")
print(f"🔹 Top-1 Mismatch: {mismatch_ratio:.2f}% (Avg KL = {_safe_avg(top1_mismatch_kl):.3f})")

# 2) Confidence mismatch cases
conf_mismatch_ratio = (len(confidence_mismatch_kl) / total * 100) if total else 0.0
print("🔸 Tokens where Expert is confident (H < {0}) & Amateur is uncertain (H > {1}):".format(
    expert_confident_H, amateur_uncertain_H))
print(f"    → {len(confidence_mismatch_kl)} tokens ({conf_mismatch_ratio:.2f}%)")
if confidence_mismatch_kl:
    print(f"    → Avg KL in these cases: {_safe_avg(confidence_mismatch_kl):.3f}")

# 3) KL tail
tail_ratio = (len(high_kl_tokens) / total * 100) if total else 0.0
print(f"🔺 Tokens with KL > {high_kl_threshold:.1f} (Strong signal steps):")
print(f"    → {len(high_kl_tokens)} tokens ({tail_ratio:.2f}%)")

print("✅ These targeted stats are directly usable in Section 2.1 to support empirical claims.")

# =====================
# KL histogram (single pass)
# =====================
num_bins = int(max_bin / bin_width)
bin_counts = defaultdict(int)

for r in records:
    kl = r["kl_divergence"]
    if kl < max_bin:
        bin_index = floor(kl / bin_width)
    else:
        bin_index = num_bins  # overflow bin
    bin_counts[bin_index] += 1

print("🔍 KL Divergence Histogram (Bin Width = {:.1f})".format(bin_width))
count_total = sum(bin_counts.values())
for i in range(num_bins + 1):
    lower = i * bin_width
    upper = (i + 1) * bin_width if i < num_bins else f"{max_bin}+"
    count = bin_counts[i]
    pct = (count / count_total * 100) if count_total > 0 else 0.0
    print(f"[{lower:.1f}, {upper}) : {count} tokens ({pct:.2f}%)")
