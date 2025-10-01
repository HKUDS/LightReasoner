<h1 align="center">
<img src="./assets/lr_logo2.png" width="120" alt="lightreasoner-logo" />
<br>
💡 LightReasoner:  
Can Small Language Models Teach Large Language Models Reasoning?
</h1>

<h3 align="center">
<a href="https://scholar.google.com/citations?user=BGT3Gb8AAAAJ&hl=en" target="_blank">Jingyuan Wang</a> ·
<a href="https://scholar.google.com/citations?user=k6yAt6IAAAAJ&hl=en&oi=sra" target="_blank">Yankai Chen</a> ·
<a href="https://scholar.google.com/citations?user=__9uvQkAAAAJ&hl=en" target="_blank">Zhonghang Li</a> ·
<a href="https://scholar.google.com/citations?user=Zkv9FqwAAAAJ&hl=en" target="_blank">Chao Huang</a>
</h3>


<div align="center">

![](https://img.shields.io/badge/Status-Under%20Review%20(ICLR%202026)-red)
![](https://img.shields.io/badge/Code%20License-MIT-green)

</div>

<p align="center">
  <a href="./9130_LightReasoner_Can_Small_L.pdf"><b>[📜 Paper]</b></a> •
  <a href="https://anonymous.4open.science/r/LightReasoner"><b>[🐱 Anonymous Repo]</b></a> •
  <a href="https://github.com/QwenLM/Qwen2.5-Math"><b>[🔗 Baselines]</b></a>
</p>



<!-- <p align="center">
  <img src="./assets/radar_1.5B.png" width="230" />
  <img src="./assets/radar_7B.png" width="230" />
  <br>
  <img src="./assets/radar_ds1.5B.png" width="230" />
  <img src="./assets/radar_1.5Bins.png" width="230" />
  <br>
  <em>Figure 1: LightReasoner consistently improves zero-shot pass@1 accuracy while requiring
  90% less time, 80% fewer sampled problems, and 99% fewer tuned tokens compared to SFT.</em>
</p> -->

<p align="center">
  <img src="./assets/lr_bars.png" width="800" />
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
  <img src="./assets/lr_new.png" width="800" />
  <br>
  <em>Figure 2: Overview of the LightReasoner framework. Informative step selection and contrastive supervision
  transform Expert–Amateur divergence into efficient reasoning signals.</em>
</p>

- **Stage 1 — Sampling:** Expert and Amateur models generate predictions under the same prefixes. Steps with high divergence are retained as critical reasoning points.  
- **Stage 2 — Fine-tuning:** Contrastive soft labels encode the Expert’s advantage. The Expert is then fine-tuned with LoRA to reinforce its strengths.

This turns weaker models into effective *teaching signals*, enabling order-of-magnitude efficiency gains without relying on ground-truth labels.






## 📊 Main Results

| Model                                         | GSM8K | MATH | SVAMP | ASDiv | Minerva Math | Olympiad Bench | MMLU STEM | AVG. |
|-----------------------------------------------|-------|------|-------|-------|-------------------|---------------|----------------|------|
| **<nobr>Qwen2.5-Math-1.5B</nobr>**            |       |      |       |       |                   |               |                |      |
| Baseline                                      | 42.5  | 34.2 | 68.8  | 68.1  | 9.9               | 23.7          | 49.8           | 42.4 |
| + SFT                                         | 69.2  | 57.1 | 64.1  | 70.2  | **15.1**          | **27.6**      | 47.7           | 50.1 |
| + LightR                                      | **70.6** | **59.3** | **76.0** | **79.8** | 11.4 | 27.1 | **54.9** | **54.2** |
| **<nobr>Qwen2.5-Math-1.5B-Instruct</nobr>**   |       |      |       |       |                   |               |                |      |
| Baseline                                      | 84.8  | 75.8 | 94.2  | 94.7  | 29.4              | 37.5          | 57.4           | 67.7 |
| + SFT                                         | 85.4  | 75.8 | 93.5  | 94.7  | 31.6              | 37.5          | 56.2           | 67.8 |
| + LightR                                      | **86.7** | 75.5 | 93.0 | 94.1 | **32.0** | **37.8** | 55.2 | **67.8** |
| **<nobr>DeepSeek-R1-Distill-Qwen-1.5B</nobr>**|       |      |       |       |                   |               |                |      |
| Baseline                                      | 75.2  | 54.2 | 79.9  | 84.9  | 16.2              | 19.1          | 22.3           | 50.3 |
| + SFT                                         | 78.2  | **60.3** | 81.5 | 87.4 | **18.4** | 21.2 | 26.2 | 53.3 |
| + LightR                                      | **79.5** | 60.2 | **83.5** | **87.5** | 18.0 | **36.5** | **26.2** | **55.9** |
| **<nobr>Qwen2.5-Math-7B</nobr>**              |       |      |       |       |                   |               |                |      |
| Baseline                                      | 57.5  | 51.8 | 67.9  | 72.7  | 14.0              | 16.0          | 69.8           | 50.0 |
| + SFT                                         | 64.4  | **63.3** | 76.2 | 76.6 | 12.1 | **20.5** | 68.5 | 54.5 |
| + LightR                                      | **67.9** | 57.8 | **77.2** | **80.6** | 12.1 | 16.9 | **70.5** | **54.7** |
| **<nobr>Qwen2.5-Math-7B-Instruct</nobr>**     |       |      |       |       |                   |               |                |      |
| Baseline                                      | 95.2  | 83.2 | 93.9  | 95.3  | 33.8              | 41.5          | 69.3           | 73.2 |
| + SFT                                         | 95.4  | 83.1 | **94.1** | 95.2 | **38.2** | 40.7 | 68.2 | **73.6** |
| + LightR                                      | **95.8** | **83.6** | 93.1 | 95.2 | 34.2 | 39.0 | 67.8 | 72.7 |




- **+28.1%** improvement on GSM8K with Qwen2.5-Math-1.5B.  
- **+25.1%** improvement on MATH with Qwen2.5-Math-1.5B.  
- Consistent gains across GSM8K, MATH, SVAMP, ASDiv, Minerva Math, OlympiadBench, and MMLU STEM.  
- Efficiency: **90% less time**, **80% fewer problems**, **99% fewer tokens**.  



| **Method** | **Total Time** | **Sampled Problems** | **Tuned Tokens** | **Avg. Gain** |
|------------|----------|------------|------------|----------|
| **Qwen2.5-Math-1.5B** |||||
| + SFT (RFT)      | 4.0h     | 3952       | 1.77M      | +7.7%   |
| **+ LightReasoner** | **0.5h** | **1000**  | **0.02M**  | **+11.8%** |
| **Qwen2.5-Math-7B** |||||
| + SFT (RFT)      | 9.5h     | 6029       | 2.20M      | +4.5%   |
| **+ LightReasoner** | **0.75h** | **1000** | **0.02M**  | **+4.7%** |
| **DeepSeek-R1-Distill-Qwen-1.5B** |||||
| + SFT (RFT)     | 3.6h     | 6023       | 5.95M      | +3.0%   |
| **+ LightReasoner** | **0.5h** | **1000**  | **0.02M**  | **+5.6%** |
| **Qwen2.5-Math-1.5B-Instruct** |||||
| + SFT (RFT)     | 3.4h     | 7153       | 2.08M      | +0.1%   |
| **+ LightReasoner** | **0.4h** | **1000**  | **0.02M**  | +0.1%   |

<p align="center">
  <img src="./assets/radar_1.5B.png" width="230" />
  <img src="./assets/radar_7B.png" width="230" />
  <img src="./assets/radar_ds1.5B.png" width="230" />
  <img src="./assets/radar_1.5Bins.png" width="230" />
  <br>
  <em>Figure 1: LightReasoner consistently improves zero-shot pass@1 accuracy while requiring
  90% less time, 80% fewer sampled problems, and 99% fewer tuned tokens compared to SFT.</em>
</p>

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
