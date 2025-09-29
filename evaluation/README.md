# 📊 Model Evaluation Setup

This guide walks you through the setup process for evaluating math reasoning performance using the [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) repository.

---

## 🔁 Step 1: Clone the Repository

```bash
git clone https://github.com/QwenLM/Qwen2.5-Math.git
```

---

## 📦 Step 2: Install Dependencies

### Standard Installation

```bash
cd Qwen2.5-Math/evaluation/

# Install latex2sympy (symbolic math parser)
cd latex2sympy
pip install -e .
cd ..

# Install core dependencies
pip install -r requirements.txt
pip install vllm==0.5.1 --no-build-isolation
pip install pebble word2number multiprocess timeout_decorator datasets
pip install pyarrow==10.0.1
```

### ⚠️ PyTorch 2.8.0 Special Case

If you're using **PyTorch 2.8.0**, follow these additional steps:

1. **Delete** the line containing `flash_attn` in `requirements.txt`.

2. Re-run installation:

```bash
cd latex2sympy
pip install -e .
cd ..

pip install -r requirements.txt
pip install vllm==0.5.1 --no-build-isolation
pip install pebble word2number multiprocess timeout_decorator datasets
pip install pyarrow==10.0.1

# Reinstall triton to avoid compatibility issues
pip install -U --force-reinstall --no-cache-dir triton
```

---

## ⚙️ Step 3: Configure Evaluation Script

```bash
set -ex

PROMPT_TYPE=$1                   # e.g., "qwen25-math-cot"
MODEL_NAME_OR_PATH=$2            # Path to your model directory
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1               # -1 means evaluate all samples

DATA_NAME="gsm8k"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --data_name ${DATA_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --split ${SPLIT} \
  --prompt_type ${PROMPT_TYPE} \
  --num_test_sample ${NUM_TEST_SAMPLE} \
  --seed 0 \
  --temperature 0 \
  --n_sampling 1 \
  --top_p 1 \
  --start 0 \
  --end -1 \
  --use_vllm \
  --save_outputs \
  --overwrite
```

---

## 🚀 Step 4: Run the Evaluation

```bash
export CUDA_VISIBLE_DEVICES="0"

PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="/workspace/LLM_MCTS/ft_1.5_gsm8k"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

---

## ✅ Notes

- Evaluation output will be saved in `${MODEL_NAME_OR_PATH}/math_eval`.
- Set `NUM_TEST_SAMPLE` to any positive integer to evaluate a subset.
- Ensure `vllm` is properly installed and GPU memory is sufficient.
- For non-GSM8K datasets, change `DATA_NAME` accordingly (e.g., `math`, `minerva`, etc.).