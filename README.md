<h1 align="center">
<img src="./docs/static/images/lightreasoner_logo.png" width="100" alt="lightreasoner-logo" />
<br>
LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?
</h1>

<div align="center">

![](https://img.shields.io/badge/Status-Under%20Review%20(ICLR%202026)-red)
![](https://img.shields.io/badge/Code%20License-MIT-green)

</div>

<p align="center">
  <a href="./9130_LightReasoner_Can_Small_L.pdf"><b>[📜 Paper]</b></a> •
  <a href="https://anonymous.4open.science/r/LightReasoner"><b>[🐱 Anonymous Repo]</b></a> •
  <a href="https://github.com/QwenLM/Qwen2.5-Math"><b>[🔗 Baselines]</b></a>
</p>



<p align="center">
  <img src="./assets/radar_1.5B.png" width="230" />
  <img src="./assets/radar_7B.png" width="230" />
  <br>
  <img src="./assets/radar_ds1.5B.png" width="230" />
  <img src="./assets/radar_1.5Bins.png" width="230" />
  <br>
  <em>Figure 1: LightReasoner consistently improves zero-shot pass@1 accuracy while requiring
  90% less time, 80% fewer sampled problems, and 99% fewer tuned tokens compared to SFT.</em>
</p>





## 🔥 News

- [2025/09] LightReasoner paper submitted to **ICLR 2026**.  
- [2025/08] Released initial implementation and experiments on Qwen2.5-Math and DeepSeek baselines.  



## 💡 Introduction

**LightReasoner** is a self-supervised framework that enhances reasoning in LLMs by contrasting them against smaller, weaker models.  
Instead of treating all tokens equally, LightReasoner focuses only on *informative reasoning steps* identified via **Expert–Amateur KL divergence**.

<p align="center">
  <img src="./assets/lr_new.png" width="600" />
  <br>
  <em>Figure 2: Overview of the LightReasoner framework. Informative step selection and contrastive supervision
  transform Expert–Amateur divergence into efficient reasoning signals.</em>
</p>

- **Stage 1 — Sampling:** Expert and Amateur models generate predictions under the same prefixes. Steps with high divergence are retained as critical reasoning points.  
- **Stage 2 — Fine-tuning:** Contrastive soft labels encode the Expert’s advantage. The Expert is then fine-tuned with LoRA to reinforce its strengths.

This turns weaker models into effective *teaching signals*, enabling order-of-magnitude efficiency gains without relying on ground-truth labels.


## 📊 Results

<p align="center">
    <img src="./docs/static/images/results_table.png" width="1000">
        <br>
    <em>Table 1: LightReasoner achieves comparable or superior accuracy to SFT across 5 models × 7 benchmarks.</em>
</p>

- **+28.1%** improvement on GSM8K with Qwen2.5-Math-1.5B.  
- **+25.1%** improvement on MATH with Qwen2.5-Math-1.5B.  
- Consistent gains across GSM8K, MATH, SVAMP, ASDiv, Minerva Math, OlympiadBench, and MMLU STEM.  
- Efficiency: **90% less time**, **80% fewer problems**, **99% fewer tokens**.  


## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/<your-org>/LightReasoner.git
cd LightReasoner
pip install -r requirements.txt
```

### Sampling
```bash
python sampling.py   --expert Qwen2.5-Math-1.5B   --amateur Qwen2.5-0.5B   --dataset gsm8k
```

### Fine-tuning
```bash
python finetune.py   --model Qwen2.5-Math-1.5B   --data contrastive_samples.jsonl   --lora --steps 1000
```

### Evaluation
```bash
python evaluate.py   --model checkpoints/lightreasoner   --benchmarks gsm8k math svamp
```


## ☕️ Citation

If you find this work useful, please cite our paper:

```
@inproceedings{lightreasoner2026,
  title={LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```


## 📜 License

This project is licensed under the MIT License.
