<!-- Icon and title -->
<h1 align="center">
<img src="./assets/lr_logo2.png" width="100" alt="lightreasoner-logo" />
<br>
💡 LightReasoner:  
Can <strong><em>SMALL</em></strong> Language Models Teach <strong><em>LARGE</em></strong> Language Models Reasoning?
</h1>


<!-- Authors -->
<h3 align="center">
<a href="https://scholar.google.com/citations?user=BGT3Gb8AAAAJ&hl=en" target="_blank">Jingyuan Wang</a> ·
<a href="https://scholar.google.com/citations?user=k6yAt6IAAAAJ&hl=en&oi=sra" target="_blank">Yankai Chen</a> ·
<a href="https://scholar.google.com/citations?user=__9uvQkAAAAJ&hl=en" target="_blank">Zhonghang Li</a> ·
<a href="https://scholar.google.com/citations?user=Zkv9FqwAAAAJ&hl=en" target="_blank">Chao Huang</a>
</h3>


<p align="center">
  <img src="./assets/welcome.png" width="500" alt="Welcome banner"/>
</p>


<!-- Quick links -->
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.07962-b31b1b.svg)](https://arxiv.org/abs/2510.07962)
[![License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Baselines](https://img.shields.io/badge/Baselines-Qwen2.5--Math-blue.svg)](https://github.com/QwenLM/Qwen2.5-Math)
![](https://img.shields.io/badge/Python-3.10+-yellow.svg)
![](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)
[![🤗 Models](https://img.shields.io/badge/🤗_Models-Coming_Soon-ffcc4d.svg)](https://huggingface.co/)

</div>


<p align="center">
  <img src="./assets/lr_bars.png" width="800" />
  <br>
  <em>Figure 1: LightReasoner consistently improves zero-shot pass@1 accuracy while requiring
  90% less time, 80% fewer sampled problems, and 99% fewer tuned tokens compared to SFT under same settings.</em>
</p>


## 🎉 News

- [x] [2025/10/12] Released initial implementation and experiments on Qwen2.5-Math and DeepSeek baselines. 



## ⚡ TL;DR
**LightReasoner** is a lightweight and resource-efficient learning framework that turns weaker language models into effective teaching signals for reinforcing stronger models.


## 📄 Abstract
Large language models (LLMs) have demonstrated remarkable progress in reasoning, often through supervised fine-tuning (SFT). However, SFT is resource-intensive, relying on large curated datasets, rejection-sampled demonstrations, and uniform optimization across all tokens—even though only a fraction carry meaningful learning value. In this work, we explore a counterintuitive idea: can smaller language models teach larger language models by revealing high-value reasoning moments that reflect the latter's unique strength? We propose *LightReasoner*, a novel framework that leverages the behavioral divergence between a stronger *expert* model and a weaker *amateur* model. LightReasoner operates in two stages: (1) a *sampling stage* that pinpoints critical reasoning moments and constructs supervision examples capturing the expert's advantage through expert–amateur contrast, and (2) a *fine-tuning stage* that aligns the expert model with these distilled examples, amplifying its reasoning strengths. Across 7 mathematical benchmarks, LightReasoner improves accuracy by up to 28.1%, while reducing time consumption by 90%, sampled problems by 80%, and tuned token usage by 99%, all without relying on ground-truth labels. By turning weaker SLMs into effective teaching signals, LightReasoner offers a scalable and resource-efficient approach for advancing LLM reasoning.




## 🧩 LightReasoner Framework

<p align="center">
  <img src="./assets/lr_new.png" width="800" />
  <br>
  <em>
    <strong>Figure 2: Overview of the LightReasoner framework.</strong> (1) Sampling Stage: Expert and Amateur models generate distributions π<sub>E</sub> and π<sub>A</sub>. Informative step selection retains steps with D<sub>KL</sub>(π<sub>E</sub> ∥ π<sub>A</sub>) > β, and contrastive supervision constructs soft labels v<sub>C</sub> capturing the Expert's advantage through Expert–Amateur contrast. (2) Fine-tuning Stage: The Expert model is enhanced by minimizing the KL divergence between its output and v<sub>C</sub>.
  </em>
</p>




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




- **+28.1%** on GSM8K, **+25.1%** on MATH, **+7.2%** on SVAMP, **+11.7%** on ASDIV for Qwen2.5-Math-1.5B.  

- **+4.3%** on GSM8K, **+6.0%** on MATH, **+17.4%** on OlympiadBench for DeepSeek-R1-Distill-Qwen-1.5B. 

- **+10.4%** on GSM8K, **+6.0%** on MATH, **+9.3%** on SVAMP, **+7.9%** on ASDIV for Qwen2.5-Math-7B.  

- Trained *solely* on GSM8K, LightReasoner generalizes effectively for 5 baseline models, achieving consistent gains across 7 benchmarks.

- Efficiency: **90% less total time**, **80% fewer sampled problems**, **99% fewer tuned tokens**.  


## ⏱️ Efficiency Study

| **Method** | **Total Time** | **Sampled Problems** | **Tuned Tokens** | **Average Gain** |
|------------|----------|------------|------------|----------|
| **Qwen2.5-Math-1.5B** |||||
| + SFT (rejection sampled)      | 4.0h     | 3952       | 1.77M      | +7.7%   |
| **+ LightReasoner** | **0.5h** | **1000**  | **0.02M**  | **+11.8%** |
| **Qwen2.5-Math-7B** |||||
| + SFT (rejection sampled)      | 9.5h     | 6029       | 2.20M      | +4.5%   |
| **+ LightReasoner** | **0.75h** | **1000** | **0.02M**  | **+4.7%** |
| **DeepSeek-R1-Distill-Qwen-1.5B** |||||
| + SFT (rejection sampled)     | 3.6h     | 6023       | 5.95M      | +3.0%   |
| **+ LightReasoner** | **0.5h** | **1000**  | **0.02M**  | **+5.6%** |
| **Qwen2.5-Math-1.5B-Instruct** |||||
| + SFT (rejection sampled)     | 3.4h     | 7153       | 2.08M      | +0.1%   |
| **+ LightReasoner** | **0.4h** | **1000**  | **0.02M**  | +0.1%   |

<p align="center">
  <img src="./assets/radar_1.5B.png" width="200" />
  <img src="./assets/radar_7B.png" width="200" />
  <img src="./assets/radar_ds1.5B.png" width="200" />
  <img src="./assets/radar_1.5Bins.png" width="200" />
  <br>
  <em>Figure 3: LightReasoner consistently improves zero-shot pass@1 accuracy while requiring
  90% less time, 80% fewer sampled problems, and 99% fewer tuned tokens compared to SFT.</em>
</p>


## 🧠 Impact of Contrast

| **Amateur Model** | **Perf. Gap** | **GSM8K** | **MATH** | **SVAMP** | **ASDiv** | **MMLU STEM** | **AVG.** |
|-------------------|-------------|-----------|----------|-----------|-----------|---------------|----------|
| **Expert: <nobr>Qwen2.5-Math-1.5B</nobr>** |||||||||
| **<nobr>Qwen2.5-0.5B</nobr>**             | **38.2**  | **70.6** | **59.3** | **76.0** | **79.8** | **54.9** | **68.1** |
| <nobr>Qwen2.5-1.5B</nobr>                 | 35.1  | 63.4 | 57.1 | 69.7 | 75.7 | 54.8 | 64.1 |
| <nobr>Qwen2.5-Math-1.5B</nobr>            | /  | / | / | / | / | / | / |
| <nobr>Qwen2.5-Math-1.5B-Ins</nobr>        | -42.3 | 41.4 | 35.5 | 67.5 | 66.4 | 55.0 | 53.2 |
| *Expert Only (Baseline)*                  | /     | 42.5 | 34.2 | 68.8 | 68.1 | 49.8 | 52.7 |
| **Expert: <nobr>Qwen2.5-Math-7B</nobr>** |||||||||
| **<nobr>Qwen2.5-0.5B</nobr>**             | **53.2**  | **67.9** | **57.8** | **77.2** | **80.6** | **70.5** | **70.8** |
| <nobr>Qwen2.5-1.5B</nobr>                 | 50.1  | 69.0 | 56.0 | 77.6 | 78.9 | 69.5 | 70.2 |
| <nobr>Qwen2.5-Math-1.5B</nobr>            | 15.0  | 56.9 | 50.2 | 63.5 | 63.4 | 70.7 | 60.9 |
| <nobr>Qwen2.5-Math-1.5B-Ins</nobr>        | -27.3 | 59.4 | 49.0 | 68.3 | 69.6 | 70.3 | 63.3 |
| *Expert Only (Baseline)*                  | /     | 57.5 | 51.8 | 67.9 | 72.7 | 69.8 | 63.9 |





### 🏆 Comparison with Competing Methods

<table>
<tr>
<td>

<!-- Left Table -->
  
| **Attribute**        | **Time** | **SFT** | **LightR** |
|-----------------------|----------------|---------|------------|
| Full trajectories     | ⬆️          | ✅      | ❌         |
| All-token tuning      | ⬆️          | ✅      | ❌         |
| Prefix termination    | ⬇️          | ❌      | ✅         |
| Selective tokens      | ⬇️          | ❌      | ✅         |
| Verification-free     | ⬇️          | ❌      | ✅         |

</td>
<td>

<!-- Right Table -->

| **Attribute**         | **Utility** | **CD**      | **LightR** |
|------------------------|------------------|-------------|------------|
| Contrast usage         | /                | Inference   | Training   |
| Size-based contrast    | ⬇️            | ✅          | ❌         |
| Expertise contrast     | ⬆️            | ❌          | ✅         |
| Persistent benefits    | ⬆️            | ❌          | ✅         |
| Independent inference  | ⬆️            | ❌          | ✅         |

</td>
</tr>
</table>















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
@misc{wang2025lightreasonersmalllanguagemodels,
      title={LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?}, 
      author={Jingyuan Wang and Yankai Chen and Zhonghang Li and Chao Huang},
      year={2025},
      eprint={2510.07962},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07962}, 
}
```


## 📜 License

This project is licensed under the MIT License.
