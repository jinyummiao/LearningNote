---
description: 'CaTiLoc: Camera Image Transformer for Indoor Localization'
---

# \[ICASSP 2021] CaTiLoc

{% embed url="https://ieeexplore.ieee.org/abstract/document/9414939" %}

&#x20;Abstract

对image transformer进行了一些适应性的改动来对每张图像估计3D位置和4D四元数信息。

### Introduction

本文提出的系统有如下性质： 1.输入为RGB图像; 2. 不需要深度信息; 3. 不需要部署额外的设备; 4. 可以在具有动态物体的多变环境中使用; 5. 在用部分遮挡物体时依然有效; 6. 实时性好，可以用于真实场景; 这篇论文的创新性在于：

1. 用transformer结构代替传统使用的CNN结构来实现相机定位
2. 提出一种将transformer与CNN提取的特征相结合的工程技巧，在保留位置信息的同时考虑空间信息;
3. 提出一种应用双重掩膜的工程技巧，可以提高整个系统对环境变化和物体运动结果的鲁棒性。

### The Proposed Architecture

![](<../../../.gitbook/assets/image (502).png>)

![](<../../../.gitbook/assets/image (10) (1).png>)

对比ViT，用CNN提取特征，输入Transformer Encoder，修改了输出维度。

#### Processing Pipeline

![](<../../../.gitbook/assets/image (320).png>)

输入一个大小为$$H_1 \times W_1$$的图像$$x \in R^{H_1 \times W_1 \times 3}$$，缩放到224x224。使用MobileNet v1结构的前三层提取特征，得到112x112x64的特征，每个特征图被reshape为展平的2D patches，$$x_p \in R^{N \times (P^2 \cdot C)}$$，其中(P,P)为patch的分辨率，C是通道数（C=64），$$N=H_2W_2/P^2$$，$$H_2=W_2=112$$。然后将64维的patches排列起来输入12个transformer encoders的栈。transformer栈的输出然后输入一个有512个神经元的全连接层，dropout=0.15，然后通过两个全连接层得到3D位置和4D四元数向量。

### Experiments

用7scenes训练。输入图像的10%被随机删除，并用其他像素的平均值加上高斯噪声来填充，来让模型对环境变换和移动物体产生鲁棒性。另外，掩去tranformer encoder15%的输入，让模型感知图像patches的位置关系。通过这两种mask，模型可以很好的感知patches在图像中的位置。&#x20;

![](<../../../.gitbook/assets/image (170).png>)

