---
description: 'Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs'
---

# \[CVPR 2022] Mega-NeRF

{% embed url="https://github.com/cmusatyalab/mega-nerf" %}

{% embed url="https://meganerf.cmusatyalab.org/" %}

## Abstract

In contrast to single object scenes (on which NeRFs are traditionally evaluated), our scale poses multiple challenges including (1) the need to model thousands of images with varying lighting conditions, each of which capture only a small subset of the scene, (2) prohibitively large model capacities that make it infeasible to train on a single GPU, and (3) significant challenges for fast rendering that would enable interactive fly-throughs.&#x20;

To address these challenges, we begin by analyzing visibility statistics for large-scale scenes, motivating a sparse network structure where parameters are specialized to different regions of the scene. We introduce a simple geometric clustering algorithm for data parallelism that partitions training images (or rather pixels) into different NeRF submodules that can be trained in parallel. We evaluate our approach on existing datasets (Quad 6k and UrbanScene3D) as well as against our own drone footage, improving training speed by 3x and PSNR by 12%. We also evaluate recent NeRF fast renderers on top of Mega-NeRF and introduce a novel method that exploits temporal coherence. Our technique achieves a 40x speedup over conventional NeRF rendering while remaining within 0.8 db in PSNR quality, exceeding the fidelity of existing fast renderers.

## Introduction

<figure><img src="../../.gitbook/assets/image (639).png" alt=""><figcaption></figcaption></figure>

Mega-NeRF是一个用于训练大规模3D场景的框架，支持可交互的human-in-the-loop fly-throughs。首先分析大规模场景的数据特点（如表1），可知在任一特定的场景点都只能看到一小部分训练图像，作者引入了一个稀疏的网络结构，将网络参数专门用于建模场景的不同区域。作者引入一个简单的几何聚类算法，将训练图像（或像素）划分到不同的NeRF子模块中，各子模块可以并行训练。We further exploit spatial locality at render time to implement a just-in-time visualization technique that allows for interactive fly-throughs of the captured environment.

**Contributions** We propose a reformulation of the NeRF architecture that sparsifies layer connections in a spatially-aware manner, facilitating efficiency improvements at training and rendering time. We then adapt the training process to exploit spatial locality and train the model subweights in a fully parallelizable manner, leading to a 3x improvement in training speed while exceeding the reconstruction quality of existing approaches. In conjunction, we evaluate existing fast rendering approaches against our trained Mega-NeRF model and present a novel method that exploits temporal coherence. Our technique requires minimal preprocessing, avoids the finite resolution shortfalls of other renderers, and maintains a high level of visual fidelity. We also present a new large-scale dataset containing thousands of HD images gathered from drone footage over 100,000 $$m^2$$ of terrain near an industrial complex.

## Approach

### Model Architecture

#### Background

NeRFs represent a scene within a continuous volumetric radiance field that captures both geometry and view-dependent appearance. NeRF encodes the scenes within the weights of a multilayer perceptron (MLP). 在渲染时，NeRF对图像的每个像素投影一个ray r，并在ray上进行采样。对于某个采样点$$p_i$$，NeRF根据位置(x,y,z)​和ray方向$$\textbf{d}=(d_1,d_2,d_3)$$来进行检索，获得占据概率$$\sigma_i$$和颜色$$c_i=(r,g,b)$$，然后通过数值积分$$\sum_{i=0}^{N-1} T_i(1-exp(-\sigma_i\delta_i))c_i$$来预测ray的颜色$$\hat{C}(r)$$，其中$$T_i=exp(-\sum_{j=0}^{i-1}\sigma_j\delta_j)$$，$$\delta_i$$是样本$$p_i$$​和$$p_{i+1}$$​之间的距离。训练过程通过采样图像像素点batches R并最小化损失函数$$\sum_{r\in R} {||C(r)-\hat{C}(r)||}^2$$。NeRF通过两阶段hierarchical采样过程来采样相机rays，并用位置编码来更好的捕获高频细节。

#### Spatial partitioning

Mega-NeRF将一个场景分解为cells，中心点为$$n=(n_x,n_y,n_z)$$​，并初始化一个对应的模型$$f^n$$​。每个子模块是一组全连接层（NeRF）。对于每个输入图像a，作者使用了一个额外的外观编码向量来计算radiance，这使得Mega-NeRF具备应对光照变化的灵活性，对于大场景来说很重要。在检索时间，Mega-NeRF根据距检索点最近的模型$$f^n$$，对给定的位置x、方向d和外观编码$$l^{(a)}$$生成一个占据概率和颜色：

<figure><img src="../../.gitbook/assets/image (593).png" alt=""><figcaption></figcaption></figure>

#### Centroid selection

在实验中，将场景嵌入一个自顶向下的2D grid会带来较好的效果。在本文的场景中，海拔高度上的差异相比经纬度的差异要小很多，所以固定中心的高度为一个常值。

#### Foreground and background decomposition

与NeRF++相似，将场景划分为一个前景volume，包含所有的相机位姿，和一个背景，覆盖互补的区域。每个volume都是用独立的Mega-NeRFs来表征的。作者也使用了像NeRF++一样的4D outer volume parameterization and raycasting formulation，但是用椭球来来代替NeRF++的单位球分区来进行改进，能够更紧密地包围住相机位姿和相关的前景细节。作者还利用相机高度的测量，通过在地面附近终止rays来进一步细化场景的采样边界，因此避免了检索不必要的地下区域，采样更加高效。

<figure><img src="../../.gitbook/assets/image (612).png" alt=""><figcaption></figcaption></figure>

### Training

#### Spatial Data Parallelism

每个Mega-NeRF子模块都是一个独立的MLP，可以并行地训练每个模型，而不需要模型间的交互。由于每张图像只能捕获场景的一小部分，作者将每个子模块的训练集限制在可能相关的像素。具体来说，我们沿着每个训练图像中每个像素对应的相机ray进行采样，并仅将与那些空间cells有交集的像素加入训练集。We include a small overlap factor between cells (15% in our experiments) to further minimize visual artifacts near boundaries.

#### Spatial Data Pruning

上一步的像素分配过程只基于相机位置，没有考虑场景的几何信息。当NeRF获得一个粗略的场景理解能力，可以进一步丢弃掉对特定NeRF没有贡献（比如被遮挡了）的无关像素/rays。

<figure><img src="../../.gitbook/assets/image (599).png" alt=""><figcaption></figcaption></figure>

### Interactive Rendering

