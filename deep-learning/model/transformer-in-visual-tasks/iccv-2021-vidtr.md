---
description: 'VidTr: Video Transformer without Convolutions'
---

# \[ICCV 2021] VidTr

{% embed url="https://arxiv.org/abs/2104.11746" %}

### Abstract

作者在视频分类任务中引入具有separable-attention的video transformer。与惯用的3D网络相比，VidTr可以通过stacked attentions来聚合时空信息。作者首先引入vanilla video transformer，来证明transformer模块可以对图像像素进行时空上的建模，但是消耗内存较大。然后，作者提出VidTr，缩小了3.3倍的内存消耗，但是表现不下降。为了进一步改进模型，作者提出基于标准分离（standard deviation）的topK pooling attention，通过丢弃没有信息量的特征来减少计算。

### Introduction

基于卷积的视频动作分类算法有两个缺点：1）它们每一层的感受野都是有限的；2）通过堆叠的卷积层聚合信息很低效，有时甚至是无效的。这篇论文中，作者提出一种基于transformer的视频网络，直接在视频像素上使用注意力机制，进行视频分类。&#x20;

![](<../../../.gitbook/assets/image (333).png>)

首先，作者引入一种原始的video transformer，通过video transformer直接从视频像素中学习时空特征，证明该模型可以实现像素级别的时空建模。但是，transformer关于序列长度的复杂度为$$\mathbb{O}(n^2)$$，需要大量计算内存。作者进一步引入可分离注意力机制，分别使用空间上和时间上的注意力机制。这一步可以将内存消耗减少3.3倍，而且不损失精度。进一步地，作者发现很多视频的大部分信息是冗余的，因为它们包含了很多邻近的重复帧，因此，作者提出一种基于topK pooling处理的standard deviation，减短了序列长度，让transformer网络专注更具代表性的帧。

### Video Transformer

#### Vanilla Video Transformer

作者使用了transformer encoder结构来实现动作识别，以原始像素为输入。当给定一段视频片段$$V\in \mathbb{R}^{C \times T \times H \times W}$$，其中T表示片段长度，W和H为视频的宽和高，C表示通道数。首先，作者将V转换为一段$$s \times s$$ patches序列，并且对每个patch进行线性embedding，即得到$$S\in \mathbb{R}^{T \frac{H}{s} \frac{W}{s} \times C'}$$，其中C'为线性映射后的通道数。作者还对S加入了一个一维的位置编码，和一个类别token来将整段序列中的特征聚合在一起来进行分类。因此，可以得到$$S'\in \mathbb{R}^{(\frac{THW}{s^2}+1) \times C'}$$，其中$$S'_0 \in \mathbb{R}^{1 \times C'}$$为附加的类别token。S'作为transformer encoder的输入。&#x20;

![](<../../../.gitbook/assets/image (893).png>)

如图2所示，作者将ViT transformer结构进行拓展，以适应3D特征学习。作者堆叠了12个encoder层，每个encoder层包含一个8-head self-attention层和两个分别包含768与3072个隐层单元的全连接层，每个注意力层学习一个spatio-temporal affinity map $$Attn \in \mathbb{R}^{(\frac{TWH}{s^2}+1)\times(\frac{TWH}{s^2}+1)}$$.

#### VidTr

上述affinity attention matrix $$Attn \in \mathbb{R}^{(\frac{TWH}{s^2}+1)\times(\frac{TWH}{s^2}+1)}$$需要保存在内存中来反向传播，因此内存消耗是随序列长度二次方增长的。原始的video transformer会将affinity map的内存消耗从$$\mathbb{O}(W^2H^2)$$增加到$$\mathbb{O}(T^2W^2H^2)$$，在一般的GPU无法正常训练。

**Separable-Attention**

为了解决这一内存约束，作者引入一种multi-head separable-attention (MSA)，该方法将3D的自注意力机制解耦为一个独立的空间注意力机制$${MSA}_s$$和一个独立的时间注意力机制$${MSA}_t$$：&#x20;

![](<../../../.gitbook/assets/image (702).png>)

与原始video transformer直接在S上进行一维序列建模不同，作者将S解耦为2D的序列$$\hat{S}\in \mathbb{R}^{(T+1)\times (\frac{HW}{s^2}+1) \times C'}$$，带有位置编码和两种类别tokens，在空间和时间维度上都加入类别tokens。这里，空间类别token利用空间注意力机制从单帧图像的patches中聚合信息，而时间类别token利用时间注意力机制从多帧间的patches中聚合信息。空间和时间类别tokens的交集$$\hat{S}^{(0,0,:)}$$被用于最后的分类。为了解耦2D序列特征$$\hat{S}$$上解耦1D自注意力，作者首先独立的在每个空间位置上使用时空注意力机制：&#x20;

![](<../../../.gitbook/assets/image (483).png>)

其中$$\hat{S}_t \in \mathbb{R}^{(\tau + 1) \times (\frac{WH}{s^2}+1) \times C}$$为$$MSA_t$$的输出。pool表示下采样方法，减少冗余信息（从T减到$$\tau$$，当$$\tau=T$$时，说明没采用下采样），$$q_t,k_t,v_t$$是在$$\hat{S}$$上应用独立的线性映射得到的：&#x20;

![](<../../../.gitbook/assets/image (1062).png>)

除此之外，$${Attn}_t \in \mathbb{R}^{(\tau + 1)\times (T+1)}$$表示由$$q_t$$和$$k_t$$矩阵相乘得到的时间注意力，在$${MSA}_t$$之后，加入一个相似的在空间维度上的1D序列注意力$${MSA}_s$$:&#x20;

![](<../../../.gitbook/assets/image (167).png>)

其中$$\hat{S}_{st} \in \mathbb{R}^{(\tau + 1) \times (\frac{WH}{s^2}+1) \times C}$$是$${MSA}_s$$的输出，$$q_s,k_s,v_s$$是在$$\hat{S}_t$$上应用独立的线性映射得到的。$${Attn}_s \in \mathbb{R}^{(\frac{WH}{s^2}+1)\times (\frac{WH}{s^2}+1)}$$表示一个spatial-wise affinity map。在空间注意力机制上，没有采用下采样方法。&#x20;

所提出的时空分离注意力机制将transformer的所需内存从$$\mathbb{O}(T^2W^2H^2)$$减少到了$$\mathbb{O}(\tau^2+W^2H^2)$$.

**Temporal Down-sampling method**

视频片段的时间维度上进行存在着冗余信息，作者映入compact VidTr（C-VidTr）来在时间维度上下采样。作者试验了不同的时间上下采样方法（公式3中的pool），包括时间上avg pooling和用步长为2的1D卷积，来将时空维度减半。这些方法的限制在于它们均匀的在时间维度上聚合信息，但是在视频片段中，有信息的帧往往不是均匀分布的。作者注意到，当一个时间实例具有信息时，时间注意力在少量的时间实例上值很高，而如果一个时间实例没有信息性，注意力值有可能在视频片段上平均分布。&#x20;

基于此直觉，作者提出了一种基于topK的pooling方法（topK\_std pooling），根据注意力矩阵中每一行的标准偏差对实例进行排序，选择affinity矩阵中具有topK最高标准偏差的行：&#x20;

![](<../../../.gitbook/assets/image (496).png>)

其中$$\sigma\in \mathbb{R}^T$$是$${Attn}_t^{(1:,:)}$$的row-wise标准偏差。在pooling过程中，保留信息聚合tokens。

### Experiments

模型结构设置&#x20;

![](<../../../.gitbook/assets/image (187).png>)

在Kinetics 400上的结果&#x20;

![](<../../../.gitbook/assets/image (315).png>)

![](<../../../.gitbook/assets/image (173).png>)

![](<../../../.gitbook/assets/image (215).png>)

![](<../../../.gitbook/assets/image (890).png>)

在Kinetics 700数据集上的结果&#x20;

![](<../../../.gitbook/assets/image (364).png>)

在Charades、Something-something-V2、UCF和HMDB数据集上的结果&#x20;

![](<../../../.gitbook/assets/image (375).png>)
