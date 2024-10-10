# FSANet: Frequency-Separated Attention Network for Image Super-Resolution

Daokuan Qu, Liulian Li and Rui Yao

[![MDPI2024](https://img.shields.io/badge/MDPI-2024-brightgreen.svg?style=plastic)](https://www.mdpi.com/2076-3417/14/10/4238/pdf?version=1716284016)
## Table of Contents

1. [Introduction](#introduction)
2. [Results](#introduction)
3. [Preparation](#preparation)
4. [Testing](#testing)
5. [Training](#training)
6. [Citation](#citation)

## Introduction

The use of deep convolutional neural networks has significantly improved the performance
of super-resolution. Employing deeper networks to enhance the non-linear mapping capability from
low-resolution (LR) to high-resolution (HR) images has inadvertently weakened the information
flow and disrupted long-term memory. Moreover, overly deep networks are challenging to train,
thus failing to exhibit the expressive capability commensurate with their depth. High-frequency and
low-frequency features in images play different roles in image super-resolution. Networks based on
CNNs, which should focus more on high-frequency features, treat these two types of features equally.
This results in redundant computations when processing low-frequency features and causes complex
and detailed parts of the reconstructed images to appear as smooth as the background. To maintain
long-term memory and focus more on the restoration of image details in networks with strong
representational capabilities, we propose the Frequency-Separated Attention Network (FSANet),
where dense connections ensure the full utilization of multi-level features. In the Feature Extraction
Module (FEM), the use of the Res ASPP Module expands the network’s receptive field without
increasing its depth. To differentiate between high-frequency and low-frequency features within
the network, we introduce the Feature-Separated Attention Block (FSAB). Furthermore, to enhance
the quality of the restored images using heuristic features, we incorporate attention mechanisms
into the Low-Frequency Attention Block (LFAB) and the High-Frequency Attention Block (HFAB)
for processing low-frequency and high-frequency features, respectively. The proposed network
outperforms the current state-of-the-art methods in tests on benchmark datasets.

<img src=./assets/FSANet.png />

## Results

<img src=./assets/Results.png />

## Preparation

### Requirements and Dependencies:

Here we list our used requirements and dependencies.

 - Python: 3.11.5
 - PyTorch: 2.2.0
 - Torchvision: 0.17.0

### Dataset：

Download the Div2k dataset at [Div2k](https://drive.google.com/file/d/1Wk_OXbfFkNuWxIz23Ji56Ju4knH4DeFm/view?usp=drive_link). 
Change data to .h5 file

```
python div2h5.py
```

## Testing

Download the pre-trained model [FSANet](https://drive.google.com/drive/folders/1C67kAiexdEogP4PeXQaAtQzE8oqUvVZk?usp=drive_link).

```shell
bash test.sh
```

## Training

```shell
python train.py --patch_size 64 --batch_size 16 --lr 0.0001 --decay 150000  --scale 2 --gpu_index 0 --n_blocks 3
python train.py --patch_size 64 --batch_size 16 --lr 0.0001 --decay 150000  --scale 3 --gpu_index 2 --n_blocks 3 
python train.py --patch_size 64 --batch_size 16 --lr 0.0001 --decay 150000  --scale 4 --gpu_index 3 --n_blocks 3  
```

## Citation

If you find the code useful in your research, please cite:

    @article{qu2024frequency,
      title={Frequency-Separated Attention Network for Image Super-Resolution},
      author={Qu, Daokuan and Li, Liulian and Yao, Rui},
      journal={Applied Sciences},
      volume={14},
      number={10},
      pages={4238},
      year={2024},
      publisher={MDPI}
    }

## License

See [MIT License](/LICENSE)
