# README

## Overview

The experiments are conducted on the CIFAR-10 SVHN and ImageNet datasets. Here, we provide the guidline for CIFAR-10.


## Pretrained Weight

The weights include both the diffusion model’s weights and the classifier’s weights. We use the same weights as [https://github.com/NVlabs/DiffPure](DiffPure). \
Place the diffusion model weights for CIFAR10, named ```checkpoint_8.pth```, and for ImageNet, named  ```256x256_diffusion_uncond.pt```, in the path ```./pretrained/guided_diffusion/```. \
The classifier weights for WideResNet28-10 (CIFAR10) will be automatically downloaded into ```./models/cifar10/L2/Standard.pt```. For the classifier weights of WideResNet70-16 (CIFAR10), you need to download them manually to  ```./pretrained/``` and load them according to the instructions in [https://github.com/NVlabs/DiffPure](DiffPure).

## DataSet

All datasets should be placed in the path ```./datasets/{}```. CIFAR10 and SVHN will be automatically downloaded by torchvision. For ImageNet, you need to download it manually. Unlike DiffPure, we do not use LMDB.




## Running Experiments

First, install the required environment:
```bash
pip install -r requirements.txt
```


### Example Evaluation

Below is an example of how to run an experiment on CIFAR10 with the WideResNet-28-10 classifier for evaluation using the PGD+EOT $l_{\infty}$ attack:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --standalone --nnodes=1 --nproc_per_node=1 ddp_test.py\
    --strength 0.1 \
    --amplitude_cut_range 3 \
    --phase_cut_range 2 \
    --delta 0.2 \
    --attack_ddim_steps 10\
    --defense_ddim_steps 500 \
    --attack_method pgd\
    --n_iter 200 \
    --eot 20 \
```

Below is an example of how to run an experiment on CIFAR10 with the WideResNet-28-10 classifier for evaluation using the PGD+EOT $l_{2}$ attack:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --standalone --nnodes=1 --nproc_per_node=1 ddp_test.py\
    --strength 0.1 \
    --amplitude_cut_range 3 \
    --phase_cut_range 2 \
    --delta 0.2 \
    --attack_ddim_steps 10\
    --defense_ddim_steps 500 \
    --attack_method pgdl2\
    --n_iter 200 \
    --eot 20 \
```

```
