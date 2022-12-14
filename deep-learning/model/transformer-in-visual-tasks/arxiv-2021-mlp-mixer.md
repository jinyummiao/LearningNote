# \[arxiv 2021] MLP-Mixer

{% embed url="https://arxiv.org/abs/2105.01601v1" %}

{% embed url="https://github.com/google-research/vision_transformer" %}

### Abstract

作者用提出了一个完全基于多层感知机MLP的结果，MLP-Mixer。MLP-Mixer包含两种层，一种是在每个图像patch上独立处理的MLP（混合每个位置上的特征），一种是在patches之间处理的MLP（混合空间信息）。在大规模数据集上训练或者使用先进的正则化策略，预训练的MLP-Mixer可以在图像分类任务上获得很好的表现，并且计算消耗与SOTA算法相差不多。

### Introduction

作者提出了一种有效但是理论上很简单的结果，MLP-Mixer，不使用卷积和自注意力机制。Mixer完全基于多层感知机结构，只依赖矩阵乘法运算、数据形状改变和标量非线性化。&#x20;

![](<../../../.gitbook/assets/image (866).png>)

图1给出了Mixer的宏观结构。它的输入是一串线性映射的图像patches（tokens），视作一个形状为“patches x channels"的表。Mixer采用两种MLP层，channel-mixing MLPs和token-mixing MLPs。channel-mixing MLPs在不同channel间作用，他们单独处理每个token，将表中的每一行作为输入。token-mixing MLPs在不同空间位置（token）间作用，单独处理每个channel，将表中的每一列作为输入。两种结构交互排列，来让输入的不同维度间彼此交互。&#x20;

该模型也可以看做是一个非常特殊的CNN结构，用1x1卷积来进行channel mixing，和一个共享参数的、单通道的、全感受野depth-wise卷积来进行token mixing。

### Mixer Architecture

如图1所示，Mixer的输入是一串S个彼此不重叠的图像patch，每个patch被线性映射到为C维向量，产生了一个二维真值输入表，$$X\in \mathbb{R}^{S\times C}$$.如果输入图像大小为HxW，每个patch的大小为PxP，那么patches的数量为$$S=HW/P^2$$.所有patches都被相同的投影矩阵线性映射。Mixer包含多个相同大小的层，每个层包含两个MLP块。第一个为token-mixing MLP块：在X的列上处理，或者说它的输入是X的转置$$X^T$$，得到$$\mathbb{R}^S\rightarrow\mathbb{R}^S$$，在各列间共享参数；第二个为channel-mixing MLP块，在X的行上处理，得到$$\mathbb{R}^C\rightarrow\mathbb{R}^C$$，在各行间共享参数。每个MLP块包含两个全连接层和一个非线性层，对输入的每一行独立处理。Mixer层可以为写为：&#x20;

![](<../../../.gitbook/assets/image (1049).png>)

其中$$\sigma$$是非线性函数GELU。隐层的单元数与输入尺寸无关，所以整个网络的计算复杂度是随着输入patch的长度、图像像素数量而线性增加的，ViT是随输入长度平方增加的。

### Experiments

先预训练再在下流任务上微调参数。&#x20;

![](<../../../.gitbook/assets/image (536).png>)

![](<../../../.gitbook/assets/image (505).png>)
