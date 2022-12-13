# GWPF
Official PyTorch implementation of "Communication-Efficient Federated Learning with Gradient-Wise Parameter Freezing"<br>
>Communication bottleneck is a critical challenge in federated learning due to limited network bandwidth. Existing compression methods are often accompanied by the problem of slow model convergence caused by loss of necessary information, while most parameter-based freezing mechanisms are without thawing design, resulting in degradation of model accuracy. This article proposes a gradient-wise parameter freezing (GWPF) mechanism with a thawing strategy for parameter synchronization in federated learning. The server makes a global freezing or thawing decision wisely based on the global aggregation of gradients or the trigger of locally accumulated gradients. The insignificant gradients are excluded from being transmitted between server and workers during the frozen period. The communication overhead is much reduced and the model training is greatly accelerated. The frozen period can be ended in time to aggregate the valuable locally accumulated gradients for the global synchronization, or it can be extended to further reduce the communication overhead. Compared with related frozen period controlling method, GWPF yields better synchronization efficiency and computational resource utilization. We perform the theoretical analysis of the convergence of GWPF for non-convex objectives with non-IID data distribution and conduct extensive experiments to demonstrate its effectiveness and reveal its superiority in different scenarios.<br>

We use a testbed with [Nvidia DGX-1](https://www.nvidia.cn/data-center/dgx-1/) with 32 CPU cores and 5 Tesla V100 GPUs with 32GB memory each. 
The driver version is 440.118.02 and CUDA version is 10.2.
For the base settings, the number of the illustrative workers in each round of training in our experiments is set to 10.<br>
# Quick Start
## Installation
```
conda create -n yourname python=3.8
conda activate yourname
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit==10.2 cudnn==7.6.5 nccl==2.9.9.1
```
Find your own install command in PyTorch official website: [https://pytorch.org/](https://pytorch.org/)
## Cloning
```
git clone https://github.com/Dora233/GWPF
cd GWPF
```
## Dataset Preparation
```
from torchvision import datasets
datasets.MNIST('./data', train=True, download=True)
datasets.CIFAR10('./data', train=True, download=True)
datasets.EMNIST('./data', train=True, download=True, split="balanced")
```
WikiText-2 can be downloaded from 
[https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)<br>
Remember to modify "/YOURPATH/" in train10n.sh and "/YOURdataPATH/" in Dataloader.py.

## Run Simulation
The GWPF algorithm can be tested with ResNet-18 on non-IID CIFAR-10 by:
```
sh train10n.sh
(Enter dataset_model: ) C1
(Enter is_iid: ) F
(Specify allocated GPU-ID (world_size: 10): ) 0
```