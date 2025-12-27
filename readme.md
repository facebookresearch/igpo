<div align="center">
<h1>IGPO: Inpainting-Guided Policy Optimization for Diffusion Large Language Models</h1>
<div>
    <a href="https://siyan-zhao.github.io/" target="_blank">Siyan&nbsp;Zhao</a><sup>1,2,*,†</sup> |
    <a href="https://scholar.google.com/citations?user=cOPQtYgAAAAJ&hl=zh-CN" target="_blank">Mengchen Liu</a><sup>1</sup> |
    <a href="https://superirabbit.github.io/" target="_blank">Jing Huang</a><sup>1</sup> |
    <a href="https://aptx4869lm.github.io/" target="_blank">Miao Liu</a><sup>3</sup> |
    <a href="https://chenyuwang-monica.github.io/" target="_blank">Chenyu Wang</a><sup>1,4</sup> |
    <a href="https://cranial-xix.github.io/" target="_blank">Bo Liu</a><sup>1</sup> |
    <a href="https://yuandong-tian.com/" target="_blank">Yuandong Tian</a><sup>1</sup> |
    <a href="https://ai.meta.com/people/1656270201849143/guan-pang/" target="_blank">Guan Pang</a><sup>1</sup> |
    <a href="https://ai.meta.com/people/426408310068489/sean-bell/" target="_blank">Sean Bell</a><sup>1</sup> |
    <a href="https://aditya-grover.github.io/" target="_blank">Aditya Grover</a><sup>2</sup> |
    <a href="https://scholar.google.com/citations?user=UD08fu0AAAAJ&hl=en" target="_blank">Feiyu Chen</a><sup>1,†</sup>
</div>
<br>
<div>
    <sup>1</sup> Meta Superintelligence Labs <sup>2</sup> UCLA <sup>3</sup> Tsinghua University, College of AI <sup>4</sup> MIT
</div>
<div>
    <sup>*</sup>Work done at Meta <sup>†</sup>Core Contribution
</div>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-pdf%20-b31b1b.svg)](https://arxiv.org/abs/2509.10396)
[![Project Page](https://img.shields.io/badge/Project-Page-4b44ce.svg)](#)

</div>

## Overview

A novel policy optimization framework for diffusion large language models that leverages their unique "inpainting" ability to guide exploration and improve RL training efficiency and model performance.

![Main Figure](https://github.com/facebookresearch/igpo/blob/main/static/igpo_main.png)

<div align="center">
  <hr width="100%">
</div>

## Environment Setup

```bash
conda env create -f env.yml
conda activate igpo
```

## Usage

Download the [MetaMathQA dataset](https://huggingface.co/datasets/meta-math/MetaMathQA) from Hugging Face.

After downloading, the structure should be:
```
igpo/MetaMathQA/
├── MetaMathQA-395K.json
└── README.md
```

To run IGPO:
```bash
sbatch run_igpo.slurm
```
(need to change the wandb api key in the slurm files)


To run GRPO:
```bash
sbatch run_grpo.slurm
```
## Acknowledgement

This code is built on the [D1](https://github.com/dllm-reasoning/d1) codebase. 


## Citation

If you find IGPO useful in your research, please consider citing:

```
@article{zhao2025inpainting,
  title={Inpainting-Guided Policy Optimization for Diffusion Large Language Models},
  author={Zhao, Siyan and Liu, Mengchen and Huang, Jing and Liu, Miao and Wang, Chenyu and Liu, Bo and Tian, Yuandong and Pang, Guan and Bell, Sean and Grover, Aditya and others},
  journal={arXiv preprint arXiv:2509.10396},
  year={2025}
}
```

## License

IGPO is MIT licensed, as found in the LICENSE file.
