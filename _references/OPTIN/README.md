<div align="center">
<h1>ICLR 2024: OPTIN</h1>
<h3>The Need for Speed: Pruning Transformers with One Recipe</h3>

[Samir Khaki](https://samirkhaki.com)<sup>1</sup>,[Konstantinos N. Plataniotis]()<sup>1</sup>

<sup>1</sup>  University of Toronto

([arXiv 2403.17921v1](https://arxiv.org/abs/2403.17921v1)) ([Project Page](http://www.samirkhaki.com/optin-transformer-pruning/))
([HuggingFace](https://huggingface.co/uoft-dsp-lab/OPTIN))
</div>

## News
- [2024/08/11] Initial Codebase for Vision Experiments has been released, and parameter save has been released on our HuggingFace page.
- [2024/04/08] Initial Codebase for Language Experiments has been released.
- [2024/01/16] OPTIN Transformer Pruning is accepted at ICLR 2024

## Abstract

We introduce the **O**ne-shot **P**runing **T**echnique for **I**nterchangeable **N**etworks (OPTIN) framework as a tool to increase the efficiency of pre-trained transformer architectures, across many domains, without requiring re-training. Recent works have explored improving transformer efficiency, however often incur computation- ally expensive re-training procedures or depend on architecture-specific character- istics, thus impeding practical wide-scale adoption across multiple modalities. To address these shortcomings, the OPTIN framework leverages intermediate feature distillation, capturing the long-range dependencies of model parameters (coined trajectory), to produce state-of-the-art results on natural language, image classifica- tion, transfer learning, and semantic segmentation tasks. Our motivation stems from the need for a generalizable model compression framework that scales well across different transformer architectures and applications. Given a FLOP constraint, the OPTIN framework will compress the network while maintaining competitive accuracy performance and improved throughput. Particularly, we show a ≤ 2% accuracy degradation from NLP baselines and a 0.5% improvement from state- of-the-art methods on image classification at competitive FLOPs reductions. We further demonstrate the generalization of tasks and architecture with comparative performance on Mask2Former for semantic segmentation and cnn-style networks. OPTIN presents one of the first one-shot efficient frameworks for compressing transformer architectures that generalizes well across _multiple class domains_, in particular: natural language and image-related tasks, without _re-training_.


## TODO
- [x] Initial Language-based Code Release
- [x] BERT Model Saved Parameter Rankings
- [x] Initial Vision-based Code Release
- [x] Vision Model Saved Parameter Rankings
- [ ] Upload remainder of Saved Parameter Rankings
- [ ] Cleaning/Finalizing OPTIN Code Release

## Getting Started

```bash
Create virtual env with conda/pyvenv -- python >= 3.11
pip install -r requirements.txt
Or: conda env create --name OPTIN --file environment.yaml
```

## File Tree
Download the pre-saved parameter rankings available at: https://huggingface.co/uoft-dsp-lab/OPTIN and re-create the expected file-tree. In particular, ensure the structure of the storage path(s) match the expected format below.
```
.
├── configs/*               # Configuration (.yaml) files for running specefic language or vision experiments
├── data/*                  # Dataloaders for different downstream datasets.
├── evals/*                 # Evaluation methods for different tasks on BERT and Vision Transformers
├── models/*                # Script to generate model instances
├── prune/*                 # Wrappers and core metrics for executing pruning at different granularities on different tasks (language & vision)
├── utils/*                 # Utility functions for model hooks, complexity, and pruning
├── storage/*               # Storage folder: Please create this to store the pre-saved parameter rankings:
├── storage/language/mnli/bert-base-uncased/*           # Parameter Ranking save for BERT on MNLI
├── storage/vision/ImageNet/deit-small-patch16-224/*    # Parameter Ranking save for DeiT-S on ImageNet
└── README.md
```

## Running Instructions

By default, the run will leverage the pre-saved parameter rankings if saved in the correct location; alternatively you can re-generate them on the fly by running the OPTIN search from scratch.

```bash
python main.py --config path/to/config.yaml
```

## Relevant Functions
For any of the applications, the core-pruning loss functions are implemented in [loss_components.py](\prune\loss_components.py). These functions are wrapped in the _head pruning_ and _neuron pruning_ functions under ./prune/*. Sample config.yaml files have been provided to reproduce our results. Ablative components discussed in the main paper can mostly be tested by modifying the specs in these config.yaml files.

## Language Experimental Results:

We provide the pre-computed parameter rankings for several tasks under the language folder on our HuggingFace page: https://huggingface.co/uoft-dsp-lab/OPTIN/tree/main/language. Please download respective parameter rankings, and replicate the above file tree. The pruned models can be evaluated by running the main script and specifying the task config. By default, the pre-downloaded rankings for our framework will be used.

## Vision Experimental Results (ImageNet-1K):
By default the experiments will execute the Tau configuration, which leverages the reduction scheduele derived from OPTIN to reduce tokens dynamically in run-time, hence achieving a more competetive FLOPs reduction ratio. Alternatively, static pruning along (parameter pruning) can be reproduced with "beta_config_only" set to True. We note there may be some variance in the parameter selection process due to the random batch sampled. Please download the available parameter rankings at https://huggingface.co/uoft-dsp-lab/OPTIN/tree/main/vision.


## Citation
```
@inproceedings{khaki2024the,
    title={The Need for Speed: Pruning Transformers with One Recipe},
    author={Samir Khaki and Konstantinos N Plataniotis},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=MVmT6uQ3cQ}
}
```
