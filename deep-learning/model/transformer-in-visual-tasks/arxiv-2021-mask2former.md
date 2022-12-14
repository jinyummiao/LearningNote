---
description: Masked-attention Mask Transformer for Universal Image Segmentation
---

# \[arxiv 2021] Mask2Former

{% embed url="https://bowenc0221.github.io/mask2former" %}

### Abstract

作者提出了一种统一的分割模型，适用于语义、实例和全景级别的分割任务。主要贡献在于masked attention，可以在通过将cross-attention限制在预测的mask区域中，来提取定位的特征。该模型在多个任务上都达到了sota表现。

### Introduction

![](<../../../.gitbook/assets/image (1004).png>)

目前的问题在于统一模型难以在各个任务上与针对各任务特殊设计的算法相比。因此，作者提出了一种统一的分割模型，该模型是在MaskFormer的基础上改进的，并提出了关键性的改进来获得更好的表现，并且网络更容易收敛。首先，提出了Transformer decoder中的masked attention，将attention限制在预测的分割区域中的特征；其次，使用了多尺度的高分辨率特征，帮助模型分割出小目标或小区域；第三，提出了优化方面的改进，比如交换self-attention和cross-attention的顺序，让query可学习，删除dropout，这些改进都提升了效果；最后，通过在少量随机采样的点上计算mask loss来节省训练所需的内存。

### Masked-attention Mask Transformer

![](<../../../.gitbook/assets/image (849).png>)

#### Transformer decoder with masked attention

包含masked attention操作的Transformer decoder将每个query的cross-attention限制在预测mask的前景区域内，而非整张特征图。为了处理小目标，作者采用了高效的多尺度策略，来用高分辨率特征。它以循环的方式将pixel decoder的特征金字塔中的连续特征图提供给连续的Transformer decoder层。最后，作者采用了一些优化方面的改进，来提升效果，并且不引入新的计算负担。

**Masked attention**

标准的cross-attention（带有残差通路的）为：&#x20;

![](<../../../.gitbook/assets/image (696).png>)

masked attention为：&#x20;

![](<../../../.gitbook/assets/image (550).png>)

此外，在(x,y)处的attention mask $$\mathcal{M}_{l-1}$$为：&#x20;

![](<../../../.gitbook/assets/image (831).png>)

此处，$$M_{l-1}\in {\{0,1\}}^{N\times H_lW_l}$$是l-1层Transformer decoder层预测的mask的二值化结果（以0.5为阈值），它被缩放到$$K_l$$的大小。$$M_0$$是没有将query特征输入Transformer decoder时从$$X_0$$获得二进制mask。

**High-resolution features**

Mask2Former不是一直使用高分辨率的特征图，而是使用特征金字塔，它包括低分辨率和高分辨率的特征，并将多尺度特征的一个分辨率一次提供给一个Transformer decoder层。作者使用的是用pixel decoder生成的特征金字塔，得到原图1/32、1/16、1/8大小的特征图。对于每个分辨率，加入sinusoidal positional embedding $$e_{pos}\in R^{H_lW_l\times C}$$和一个可学习的尺度embedding $$e_{1v1}$$.在模型中，这种3层的Transformer层重复了L次，因此最后的Transformer decoder包含3L层。这个模式以循环的方式在下面的所有层中重复。

**Optimization improvements**

一个标准的transformer decoder依次包含三个模块，一个self-attention模块，一个cross-attention模块和一个feed-forward网络（FFN）。此外，query特征（$$X_0$$）在输入Transformer decoder层之前被zero-initialized，并且加上一个可学习的positional embedding。并且在残差连接和attention maps都使用dropout。&#x20;

为了优化Transformer decoder的设计，作者做出如下改进：1.交换self-attention和cross-attention（即作者提出的masked attention）的顺序，来让计算更加高效。query features to the first self-attention layer are not dependent on image features yet, thus applying self-attention does not generate any meaningful features。2.让query特征$$X_0$$可学习，在输入到Transformer decoder层中生成mask $$M_0$$之前，可学习的query特征被直接监督。作者发现可学习的query特征类似于region proposal network，可以产生mask proposals。3.作者发现dropout是不必要的，会损坏效果，所以在decoder中完全删去了dropout。

#### Improving training efficiency

PointRend和Implicit PointRend指出分割模型可以在K个随机采样的点上计算mask loss，来代替在整个mask上计算，在匹配和最终的loss计算中，我们用采样点计算mask loss。在构造bipartite matching的损失矩阵的matching loss中，作者在预测mask和真值mask中均匀采样了相同的K个点。在预测和相匹配的真值间的final loss中，用importance sampling在不同对预测和真值间采样不同的K个点。设置$$K=112\times 112=12544$$

### Implementation details

We adopt settings from MaskFormer with the following differences:&#x20;

**Pixel decoder**. Mask2Former is compatible with any existing pixel decoder module. Since our goal is to demonstrate strong performance across different segmentation tasks, we use the more advanced multi-scale deformable attention Transformer (MSDeformAttn) as our default pixel decoder. Specifically, we use 6 MSDeformAttn layers applied to feature maps with resolution 1/8, 1/16 and 1/32, and use a simple upsampling layer with lateral connection on the final 1/8 feature map to generate the feature map of resolution 1/4 as the per-pixel embedding. In our ablation study, we show that this pixel decoder provides best results across different segmentation tasks.&#x20;

**Transformer decoder**. We use our Transformer decoder with L = 3 (i.e., 9 layers total) and 100 queries by default. An auxiliary loss is added to every intermediate Transformer decoder layer and to the learnable query features before the Transformer decoder.&#x20;

**Loss weights**. We use the binary cross-entropy loss and the dice loss for our mask loss: $$L_{mask} = {\lambda}_{ce} L_{ce} + {\lambda}_{dice}{L}_{dice}$$. We set $${\lambda}_{ce}=5.0$$ and $${\lambda}_{dice}=5.0$$. The final loss is a combination of mask loss and classification loss: $$L_{mask}+\lambda_{cls}L_{cls}$$ and we set $$\lambda_{cls}=2.0$$ for predictions matched with a ground truth and 0.1 for the "no object," i.e., predictions that have not been matched with any ground truth.

### Experiments

![](<../../../.gitbook/assets/image (821).png>)

![](<../../../.gitbook/assets/image (369).png>)

![](<../../../.gitbook/assets/image (489).png>)

![](<../../../.gitbook/assets/image (307).png>)

![](<../../../.gitbook/assets/image (834).png>)

![](<../../../.gitbook/assets/image (184).png>)

![](<../../../.gitbook/assets/image (695).png>)

![](<../../../.gitbook/assets/image (851).png>)

![](<../../../.gitbook/assets/image (217).png>)

![](<../../../.gitbook/assets/image (280).png>)

![](<../../../.gitbook/assets/image (1042).png>)

![](<../../../.gitbook/assets/image (484).png>)

![](<../../../.gitbook/assets/image (4).png>)

![](<../../../.gitbook/assets/image (806).png>)

![](<../../../.gitbook/assets/image (40).png>)

![](<../../../.gitbook/assets/image (374).png>)
