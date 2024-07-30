---
description: Per-Pixel Classification is Not All You Need for Semantic Segmentation
---

# \[NIPS 2021] Maskformer

{% embed url="https://arxiv.org/abs/2107.06278" %}

{% embed url="https://github.com/facebookresearch/MaskFormer" %}

### Abstract

当前语义分割算法被定义为像素级别的语义分类任务，实例分割则使用一个可替代的mask分类。作者认为使用相同的模型、损失和训练过程，mask分类能够用一个统一的模式来解决语义分割和实例分割。因此，作者提出MaskFormer，预测一组二进制编码，每个编码与一个全局类别标签相关联。所提出的模型简化了语义分割和全景分割任务的方法，并且获得了很好的实验效果。

### Introduction

&#x20;如图1，一般的语义分割（图左）对每个像素位置使用相同的语义分割损失函数，对每个像素进行语义分类，只能输出固定数量的类别，将图像分割成多个语义区域。实例分割（图右）是一种解决图像分割和语义分类问题的替代方案，这种方法预测一组二进制掩码，每个掩码对应着一种全局语义类别，因此作者提出：

> Can a single mask classification model simplify the landscape of effective approaches to semantic- and instance-level segmentation tasks? And can such a mask classification model outperform existing per-pixel classification methods for semantic segmentation?

即能否用实例分类的方法统一语义级别和实例级别的分割任务，并获得更好的表现。本文的主要贡献在于提出了一个统一的模型，用完全相同的模型、损失函数和训练过程完成语义分割、实例分割和全景分割任务。

### From Per-Pixel to Mask Classifiction

#### Per-pixel classification formulation

对于per-pixel classification的模型来说，分割模型要预测$H\times W$图像中每个像素的K个类别的分布概率：$$y={\{p_i|p_i \in \triangle^K\}}^{H \cdot W}_{i=1}$$，其中$$\triangle^K$$是K维的概率向量。训练一个per-pixel classification模型的过程为：给定每个像素的真值类别$$g^{gt}={\{y^{gt}_i|y^{gt}i \in \{1,...,K\}\}}^{H \cdot W}{i=1}$$，用交叉熵损失（负对数似然）来训练网络：&#x20;

![](<../../../.gitbook/assets/image (199).png>)

#### Mask classification formulation

mask classification将分割任务分为：1.将图像划分为N个区域（N不等于K），每个区域用一个二进制mask表示（$${\{m_i|m_i\in {[0,1]}^{H \times W}\}}^N_{i=1}$$）;2.将每个区域作为一个整体与K个类别的分布联系起来。为了实现mask classification，作者定义期望的输出z为一组N个概率-mask对，$$z={{(p_i,m_i)}}^N_{i=1}$$.与对每个像素预测类别概率不同，对于mask classification，概率分布$$p_i \in \triangle^{K+1}$$包含一个额外的"no object"类别$$\phi$$.mask classification允许对每个类别预测多个mask，使得模型可以应用于语义和实例级别的分割任务。&#x20;

为了训练一个mask classification模型，需要获得预测z和$$N^{gt}$$个真值分割$$z^{gt}={\{(c^{gt}_i,m^{gt}_i)|c^{gt}_i\in {1,...,K},m^{gt}i\in {[0,1]}^{H\times W}\}}^{N^{gt}}{i=1}$$之间的匹配$$\sigma$$.$$c^{gt}_i$$是第i个真实分割的真值类别。因为预测的数量|z|=N和真值的数量$$|z^{gt}|=N^{gt}$$一般不一样，假设$$N \geq N^{gt}$$，并用"no object"$$\phi$$来填充真值，便于一对一匹配。&#x20;

在实验中，作者发现基于bipartite matching的匹配比fixed matching效果更好。DETR用bounding box来计算预测和真值之间匹配的的assignment cost，而作者选择直接用类别和mask的预测，即$$-p_i(c^{gt}_j)+L_{mask}(m_i,m^{gt}_j)$$，其中$$L_{mask}$$是二进制mask损失。&#x20;

为了训练模型参数，给定一个匹配，mask classification损失$$L_{mask-cls}$$包含每个预测分割的交叉熵分类损失和二进制mask损失：&#x20;

![](<../../../.gitbook/assets/image (864).png>)

#### MaskFormer

![](<../../../.gitbook/assets/image (159).png>)

MaskFormer模型输出N个概率-mask对$$z={{(p_i,m_i)}}^N_{i=1}$$，该模型包含三个模块，1.一个pixel-level模块来提取per-pixel embeddings，用于生成二进制mask；2.一个transformer模块，是一组Transformer decoder层的堆叠，生成N个per-segment embeddings；3.一个segmentation模块，从这些embeddings中生成预测$${{(p_i,m_i)}}^N_{i=1}$$。在inference时，$$p_i$$和$$m_i$$一起用于最后的预测。

&#x20;结构描述如下：&#x20;

![](<../../../.gitbook/assets/image (206).png>)

实验中的结构细节：&#x20;

![](<../../../.gitbook/assets/image (18) (1).png>)

作者发现最好不用softmax去让mask彼此排斥。在训练师，损失函数包含每个预测分割的交叉熵分类损失和二进制mask损失。简单起见，作者用DETR的$$L_{mask}$$，即focal loss和dice loss的线性组合。

#### Mask-classification inference

![](<../../../.gitbook/assets/image (251).png>)

### Experiments

两个baseline的结构：&#x20;

![](<../../../.gitbook/assets/image (34).png>)

实验结果：&#x20;

![](<../../../.gitbook/assets/image (669).png>)

![](<../../../.gitbook/assets/image (811).png>)

![](<../../../.gitbook/assets/image (524).png>)

![](<../../../.gitbook/assets/image (680).png>)

![](<../../../.gitbook/assets/image (303).png>)

![](<../../../.gitbook/assets/image (842).png>)

![](<../../../.gitbook/assets/image (351).png>)
