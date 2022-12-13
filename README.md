# GWPF
Official PyTorch implementation of "Communication-Efficient Federated Learning with Gradient-Wise Parameter Freezing".<br>
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
Note that the versions of packages should correspond to each other.<br>
Find your own install command in the official website of [PyTorch: ](https://pytorch.org/get-started/previous-versions/).<br>
The versions of the following packages can be found on the NVIDIA official website:<br>
[CUDA Toolkit: ](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)<br>
[cuDNN: ](https://developer.nvidia.com/rdp/cudnn-archive)<br>
[NCCL: ](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html)
## Cloning
```
git clone https://github.com/Dora233/GWPF
cd GWPF
```
## Dataset Preparation
```
python
from torchvision import datasets
datasets.MNIST('./data', train=True, download=True)
datasets.CIFAR10('./data', train=True, download=True)
datasets.EMNIST('./data', train=True, download=True, split="balanced")
```
WikiText-2 can be downloaded from 
[https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) .<br>
Remember to modify "/YOURPATH/" in train10n.sh and "/YOURdataPATH/" in Dataloader.py.

## Run Simulation
GWPF can be tested with ResNet-18 on non-IID CIFAR-10 by:
```
sh train10n.sh
(Enter dataset_model: ) C1
#M1:dataset_name = 'mnist' model_name = 'CNN_Mnist'
#C1:dataset_name = 'cifar10' model_name = 'ResNet18_Cifar10'
#W1:dataset_name = 'wikitext2' model_name = 'transformer'
#E1:dataset_name = 'emnist' model_name = 'VGG11_EMNIST'
(Enter is_iid: ) F
#T: True F: False
(Specify allocated GPU-ID (world_size: 10): ) 0
#Deploy all workers starting from gpu 0.
```
The generated log files are in "GWPF/Logs/".