---
description: >-
  REMAP: Multi-Layer Entropy-Guided Pooling of Dense CNN Features for Image
  Retrieval
---

# \[TIP 2019] REMAP

{% embed url="https://ieeexplore.ieee.org/document/8720226" %}

### Abstract

本篇论文提出了一种基于CNN的全局描述子，称为REMAP，从多个CNN层学习并聚合多层次的特征，并用triplet loss进行端到端训练。REMAP可以显式地学习到visual abstraction的不同语义层次中相互支持和互补的有区分力的特征。这些局部特征在聚合为一个全局描述子之前，先在每层的多尺度相互重叠的区域内，被空间上max-pooled。为了识别出语义上对检索有用的区域和层，作者提出用KL散度计算每个区域和层的信息增益。该系统在训练过程中有效的学习不同区域和层的重要性，并相应地给他们加权。

### Introduction

大规模图像检索依旧存在着两个基本问题：

> how to aggregate deep features extracted by a CNN network into compact and discriminative image-level representations how to train the resultant CNN-aggregator architecture for image retrieval tasks

本篇论文提出了一种方法，在多层特征上进行基于区域的聚合，称为REMAP，Region-Entropy（作者用region entropy指示匹配与非匹配图像对距离分布的相对熵，即它们之间的KL散度） based Multi-layer Abstraction Pooling。文章的主要创新点在于：

1. 提出一种聚合多层CNN特征的方法，包含不同层次的视觉抽象特征；
2. 提出一种聚合多分辨率区域特征的方法，显式地利用根据DL散度计算出的区域显著性来控制聚合过程；
3. 所提出聚合方法比传统的CNN聚合方法更好；
4. 比较了经典CNN结构与所提出聚合方法结合后的表现，并与SOTAs进行对比。

### REMAP Representation

REMAP旨在解决基于内容的图像检索任务中的两个基本问题：

> a novel aggregation mechanism for multi-layer deep convolutional features extracted by a CNN network, an advanced assembling of multi-region and multi-layer representations with end-to-end training.

REMAP的第一个创新点在于聚合不同CNN层的特征，用于显式地学习不同的、互补的视觉特征。这意味这多层CNN层被训练为：1.在特定的within layer的聚合策略下是discriminative individually；2.在识别任务中是complementary to each other；3.对下面层提取特征提供支持。REMAP中很重要的一点在于多层端到端的finetuning，CNN的参数、相对熵的参数、PCA和whitening参数都是在triplet loss中用SGD一起训练的。REMAP多层处理过程如图1所示。&#x20;

![](<../../.gitbook/assets/image (526).png>)

相对熵加权是另一个主要贡献。这个方法主要用于估计每个局部区域中特征的discriminatory，用于最优化控制下面的sum-pooling过程。The region entropy is defined as the relative entropy between the distributions of distances for matching and non-matching image descriptor pairs, measured using the KL-divergence function。在匹配和非匹配分布中具有更好区分度的区域对于识别任务来说更加重要，因此应被赋予高权重。REMAP中的KL散度加权模块使用卷积层来实现的，权重用KL散度的值进行初始化，然后用SGD进行优化。&#x20;

最后，聚合得到的向量被拼接、PCA whiten和L2正则化，得到一个全局图像描述子。Additionally, the REMAP signatures for the test datasets are encoded using the Product Quantization (PQ) approach to reduce the memory requirement and complexity of the retrieval system. 作者用到了Region Of Interest函数(Particular object retrieval with integral max-pooling of CNN activations), $$\zeta: R^{w,h,d} \rightarrow R^{r\times d}$$。该函数将大小为$$w\times h \times d$$的tensor划分为r个相互重叠的块，在区域中进行空间最大池化，每个区域中得到一个d维的向量。更准确地说，ROI模块从S个不同尺度的CNN特征图中提取方形区域，在每个尺度中，均匀提取区域，使得相邻区域之间的重叠了靠近40%，所提取区域的数量r取决于图像大小($$1024 \times 768 \times 3$$)和尺度因子S。由表1可知，当r=40时达到最好的检索准确率，后续一直使用这一参数设置。&#x20;

![](<../../.gitbook/assets/image (513).png>)

#### CNN Layer Access Function

REMAP可以使用任何现存的CNN网络。这些CNN实质上都是L个卷积层的序列化堆叠。具体每个模块的内容随算法不同而不同，但是这里简单的视为一个函数$$l_i: R^{w_i \times h_i \times d_i} \rightarrow R^{w'_i \times h'i \times d'i, 1\le i \le L}$$. 因此CNN可以视为组合函数$$f(x)=l_L(l_{L-1}(...(l_1(x))))$$，x是输入图像。就本文而言，作者希望访问某些中间CNN层的输出，即希望构建一个layer assess函数：$$f_l(x)=l_l(l_{l-1}(...(l_1(x)))), 1\le l \le L$$.

#### parallel Divergence-Guided ROI Streams

REMAP通过并行的基于散度的ROI流来对不同CNN层的输出进行独立且不同的处理。每个流输入一些CNN层的输出，并实现ROI pooling。ROI pooling的输出向量然后被L2正则化、基于信息量加权、线性结合到一个单个聚合表征中。&#x20;

假设以CNN中第l'层的输出tensor $$o=f_{l'}(x) \in R^{w,h,d}$$作为ROI处理过程的输入，该tensor随后被输入ROI模块和L2模块，得到结果$$r=L2(\zeta(o))$$，区域向量的线性集合通过加权求和来实现：&#x20;

![](<../../.gitbook/assets/image (141).png>)

其中r(i)表示矩阵r的第i列。总而言之，ROI处理流可以定义为：&#x20;

![](<../../.gitbook/assets/image (19).png>)

在本篇工作中，不同于RMAC中使用固定常数$$\alpha={1,1,1,...,1}$$，作者提出利用每个区域匹配和非匹配描述子对的概率分布之间的类可分离性来度量区域的信息增益。算法通过以下步骤计算相对熵权重：1. 大小为$$1024 \times 768 \times 3$$的图像经过离线的ResNeXt101网络，2. 最后的卷积层输出的特征被输入ROI模块中，将$$32\times 24 \times 2048$$的tensor划分为40个块，并在区域内进行最大池化，得到每个区域/层的2048维向量(**Q:每层得到的向量都是2048维？**)，3. 对每个区域和层，分别计算一对匹配描述子和非匹配描述子的欧式距离y的概率密度函数$$Pr(y/m),Pr(y/n)$$。KL散度用于度量匹配和非匹配概率密度函数的可分离度。如图2所示，不同区域的KL散度的值不同。&#x20;

![](<../../.gitbook/assets/image (1021).png>)

KL散度的值被用于初始化ROI加权向量，并在训练过程中可以参数更新，要求权重向量$\alpha$非负。

#### Final REMAP Architecture

![](<../../.gitbook/assets/image (487).png>)

实验证明，多层CNN特征聚合效果更好，因此，作者选择了layer 1+2用于最后REMAP的表征，来实现效果和实时性的平衡。

![](<../../.gitbook/assets/image (822).png>)

#### Compact REMAP Signature

作者用了两种量化方法，第一种是对REMAP输出的D维向量，只挑选top-d维；第二种是基于Product Quantization (PQ)算法，D维REMAP描述子被分割维m个子集，每个子集长度为D/m，每个子集用一个独立的K-means量化器(k=256)来量化编码，得到$$n={log}_2 (k)$$比特，PQ编码后的描述子为$$m\times n$$维比特。在测试时，向量间的距离用Asymmetric Distance Computation计算。

### Experimental Evaluation

![](<../../.gitbook/assets/image (887).png>)

&#x20;看起来KL散度虽然只是提供了一个训练的初值，但是对于优化效果还是有提升的。&#x20;

![](<../../.gitbook/assets/image (1017).png>)

RMAC只聚合了最后一层特征，多层特征聚合可以提升效果。&#x20;

![](<../../.gitbook/assets/image (195).png>)

![](<../../.gitbook/assets/image (507).png>)

MS-REMAP是将输入缩放到两个大小，得到REMAP描述子后，加权聚合得到MS-REMAP描述子。设$$X_1$$和$$X_2$$为REMAP从$$1024\times 768,1280 \times 96$$ 大小图像中提取的描述子，加权聚合：

&#x20; &#x20;

![](<../../.gitbook/assets/image (704).png>)

![](<../../.gitbook/assets/image (486).png>)

![](<../../.gitbook/assets/image (846).png>)

![](<../../.gitbook/assets/image (373).png>)

![](<../../.gitbook/assets/image (41).png>)

![](<../../.gitbook/assets/image (860).png>)

![](<../../.gitbook/assets/image (1025).png>)
