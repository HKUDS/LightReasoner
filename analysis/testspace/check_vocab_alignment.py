'''
Usage:
    python check_vocab_alignment.py --expert MODEL_A --amateur MODEL_B

For example:
    python check_vocab_alignment.py --expert ./Qwen2.5-Math-1.5B --amateur ./Qwen2.5-0.5B

Optional:
    --k 5000            # only compare first K ids (instead of min size)
    --show-diffs 10     # print up to N differing positions (for diagnostics)
'''

import argparse
from transformers import AutoTokenizer

def id2tok_list(tok):
    # Build explicit id->token list (order matters!)
    n = len(tok)
    return [tok.convert_ids_to_tokens(i) for i in range(n)]

def compare_prefix(ids_a, ids_b, k=None):
    m = min(len(ids_a), len(ids_b))
    if k is not None:
        m = min(m, k)
    return ids_a[:m] == ids_b[:m], m

def first_diff(ids_a, ids_b, start=0, end=None):
    if end is None:
        end = min(len(ids_a), len(ids_b))
    for i in range(start, end):
        if ids_a[i] != ids_b[i]:
            return i, ids_a[i], ids_b[i]
    return None, None, None

def sample_diffs(ids_a, ids_b, limit=10):
    diffs = []
    m = min(len(ids_a), len(ids_b))
    for i in range(m):
        if ids_a[i] != ids_b[i]:
            diffs.append((i, ids_a[i], ids_b[i]))
            if len(diffs) >= limit:
                break
    return diffs

def main():
    ap = argparse.ArgumentParser(description="Verify tokenizer ID→token alignment between two models.")
    ap.add_argument("--expert", required=True, help="Expert model name/path")
    ap.add_argument("--amateur", required=True, help="Amateur model name/path")
    ap.add_argument("--k", type=int, default=None, help="Compare only first K ids (prefix check). Default: min size")
    ap.add_argument("--show-diffs", type=int, default=10, help="Show up to N differing positions")
    ap.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoTokenizer")
    args = ap.parse_args()

    print(f"Loading tokenizers…")
    tokE = AutoTokenizer.from_pretrained(args.expert, use_fast=True, trust_remote_code=args.trust_remote_code)
    tokA = AutoTokenizer.from_pretrained(args.amateur, use_fast=True, trust_remote_code=args.trust_remote_code)

    idsE = id2tok_list(tokE)
    idsA = id2tok_list(tokA)

    sizeE, sizeA = len(idsE), len(idsA)
    print(f"\nSizes:")
    print(f"  Expert  ({args.expert}): {sizeE}")
    print(f"  Amateur ({args.amateur}): {sizeA}")

    # 1) Your plan: truncate to same length (or K) and compare that segment
    same_prefix_flag, m = compare_prefix(idsE, idsA, k=args.k)
    print(f"\nPrefix comparison (length = {m}): {'MATCH' if same_prefix_flag else 'DIFFERENT'}")

    if not same_prefix_flag:
        idx, a_tok, b_tok = first_diff(idsE, idsA, end=m)
        if idx is not None:
            print(f"  First difference at id={idx}:")
            print(f"    Expert : {repr(a_tok)}")
            print(f"    Amateur: {repr(b_tok)}")
        diffs = sample_diffs(idsE, idsA, limit=args.show_diffs)
        if diffs:
            print(f"\n  Sample of {len(diffs)} differing positions in prefix:")
            for i, a, b in diffs:
                print(f"    id={i}: expert={repr(a)} | amateur={repr(b)}")
    else:
        print("  Prefix segment tokens (ID→token) are identical.")

    # 2) Robust full equality check (only if sizes match)
    print("\nFull mapping equality check:")
    if sizeE != sizeA:
        print("  Skipped (sizes differ). Full ID→token mapping CANNOT be identical.")
    else:
        identical = idsE == idsA
        print(f"  {'IDENTICAL' if identical else 'NOT IDENTICAL'} over all {sizeE} ids.")
        if not identical:
            idx, a_tok, b_tok = first_diff(idsE, idsA)
            if idx is not None:
                print(f"  First full-vocab difference at id={idx}:")
                print(f"    Expert : {repr(a_tok)}")
                print(f"    Amateur: {repr(b_tok)}")

    # 3) Optional: set-wise comparison (what tokens differ, ignoring order)
    vocabE = set(tokE.get_vocab().keys())
    vocabA = set(tokA.get_vocab().keys())
    onlyE = sorted(vocabE - vocabA)
    onlyA = sorted(vocabA - vocabE)

    print("\nSet-wise comparison (ignoring IDs):")
    print(f"  Tokens only in Expert : {len(onlyE)}")
    print(f"  Tokens only in Amateur: {len(onlyA)}")
    if onlyE[:5]:
        print(f"    e.g., {onlyE[:5]}{' …' if len(onlyE) > 5 else ''}")
    if onlyA[:5]:
        print(f"    e.g., {onlyA[:5]}{' …' if len(onlyA) > 5 else ''}")

    print("\nInterpretation:")
    print("  • If the FULL mapping is IDENTICAL, you can safely do by-index subtraction/KL.")
    print("  • If only the PREFIX matches, your by-index math is only safe within that prefix (m).")
    print("  • If sizes differ or mapping differs, align by token strings before any subtraction/KL.")

if __name__ == "__main__":
    main()
