---
description: 'ASLFeat: Learning Local Features of Accurate Shape and Localization'
---

# \[CVPR 2020] ASLFeat

{% embed url="https://arxiv.org/abs/2003.10071" %}

{% embed url="https://github.com/lzx551402/ASLFeat" %}

### Abstract

这篇论文旨在减轻联合训练局部特征detectors和descriptors时的两个限制: 1. 在提取稠密特征时，估计特征点的局部shape（scale，orientation等）的能力经常被忽视，而shape-awareness是获得更强几何不变性的关键；2. 检测到的关键点的定位精度不足。这篇论文，作者提出了三个轻量级、但是有效的改进，来减轻以上问题。首先，作者凭借deformable convolutional networks来稠密地估计和使用局部变换。第二，作者利用固有的特征层次来恢复空间分辨率和低层次细节，以得到精确的关键点定位。最后，作者使用峰值测量关联特征响应，并得到更具参考价值的检测分数。

### Introduction

作者认为联合训练detectors和descriptors存在着两大局限：1. 没有考虑有关shape的信息，而这些信息有助于获得更强的几何不变性；2. 关键点定位不准确。针对这两个问题，作者提出了三点改进：第一，采用deformable convolutional networks（DCN）来稠密的估计，不仅可以估计pixel-wise局部变换，还能通过堆叠多个DCN来逐渐对shape建模；第二，利用内在的特征层次提出多层检测机制，在不增加额外训练参数的情况下，保留空间分辨率和用于关键点精确定位的低层细节信息；最后，该模型是基于D2-Net进行的改进，提出了一个峰值预测来检测更多可选的关键点。&#x20;

作者还回答了两个问题：1.局部描述子需要什么样的deformable parameterization（geometrically constraint还是free-form modelling）；2.关键点检测需要什么样的特征融合（multi-scale input，in-network mutli-scale inference，还是multi-level fusion）

### Methods

#### Prerequisites

这篇论文网络的backbone是基于deformable convolutional networks和D2-net的。&#x20;

**Deformable convolutional networks**旨在学习动态的感受野来获得对几何变化建模的能力。对于一个在输入特征图x上采样得到的grid R，每个空间位置p上标准卷积的输出特征可以写为：&#x20;

![](<../../.gitbook/assets/image (300).png>)

DCN通过额外学习了采样的偏置（$${\triangle p_n | n=1,...,N}$$）和特征振幅（$$\triangle m_n | n=1,...,N$$）来增强标准卷积，其中N=|R|，因此上式可以写为：&#x20;

![](<../../.gitbook/assets/image (832).png>)

由于偏置$$\triangle p_n$$一般是小数，所以上式一般用线性插值实现，而特征振幅$$\triangle m_n$$被限制在(0,1)间。在训练过程中，$$\triangle p_n$$和$$\triangle m_n$$的初始值设为0和0.5.&#x20;

**D2-Net**提出了一种describe-and-detect策略来一起提取特征描述子和关键点。在最后一层特征图$$y\in \mathbb{R}^{H\times W \times C}$$上，D2-Net利用L2-normalization来获得稠密的特征描述子，而同时通过1）局部分数和2）channel维度分数来检测特征。特别的，对于$$y^c (c=1,2,...,C)$$上的每个位置$(i,j)$，局部分数为：&#x20;

![](<../../.gitbook/assets/image (1056).png>)

其中N(i,j)是(i,j)附近的相邻像素，例如由一个3x3核定义的9-领域。channel维度分数定义为：&#x20;

![](<../../.gitbook/assets/image (57).png>)

最后检测的分数为：&#x20;

![](<../../.gitbook/assets/image (340).png>)

这一分数用于在训练中对loss加权，在测试时提取top-K关键点。

#### DCN with Geometric Constraints

原本的free-form DCN预测高自由度的局部变换，比如对一个3x3核的9x2个偏置。一方面，它能够模拟复杂的变形，如非平面性，而另一方面，它承担了对local shape过度参数化的风险，其中简单的仿射或透视变换往往被认为是一个很好的近似。为了找出在本文中什么样的deformable是必要的，作者通过在DCN中设置不同的几何约束，比较了三种shap建模，包括1）similarity，2）affine，3）homography，每种shape的特性如下表所示：

![](<../../.gitbook/assets/image (31).png>)

**Affine-constrained DCN** 局部shape经常通过similarity transformation来建模，估计旋转和尺度。这种变换可以分解为：&#x20;

![](<../../.gitbook/assets/image (259).png>)

而一些工作，如HesAff和AffNet估计了shearing，因此，作者将affine transformation分解为：

![](<../../.gitbook/assets/image (854).png>)

其中det$$A'=1$$。这个网络用于预测一个代表scale的标量$$\lambda$$，两个代表旋转的标量$$cos(\theta), sin(\theta)$$，还有三个代表shearing的标量A'&#x20;

**Homography-constrained DCN** 实际上，local deformation可以通过一个单应性变换（投影变换）H来更好的近似，作者采用Tensor Direct Linear Transform（Tensor DLT）来用4点法可微的求解H。 建立一个线性系统，求解Mh=0，其中$$M\in \mathbb{R}^{8\times 9}$$，h是一个有个9个元素的向量，由h的元素组成。每对匹配可以提供给M两个等式。通过令h的最后一个元素为1，且忽略平移，作者设置$$H_{33}=1,H_{13}=H_{23}=0$$，重写以上等式为$$\hat{M_{(i)}} \hat{h}=\hat{b_{(i)}}$$，其中$$\hat{M_{(i)}} \in \mathbb{R}^{2\times 6}$$，对于每对匹配，有：&#x20;

![](<../../.gitbook/assets/image (554).png>)

$$\hat{b_{(i)}}={[-v^{'}_i,u^{'}_i]}^T \in \mathbb{R}^{2\times 1}$$，$$\hat{h}$$包括H中前两列的6个元素。通过对4对匹配叠加上式，推导出最终的线性系统：&#x20;

![](<../../.gitbook/assets/image (297).png>)

假设对应点不是共线的，$$\hat{h}$$可以用$$\hat{A}$$（**Q:应该是**$$\hat{M}$$**的伪逆吧**）的可微伪逆来求解。在实践中，作者初始化{(-1,-1),(1,-1),(1,1),(-1,1)}四个角点，并且利用网络预测8个在(-1,1)之间的对应偏移来避免共线性。 通过构建上述的变换$$T\in \{S,A,H\}$$，公式2中的偏移值可以由下式获得：&#x20;

![](<../../.gitbook/assets/image (527).png>)

来在DCN中加入几何约束。

#### Selective and Accurate Keypoint Detection

**Keypoint peakness measurement** D2-Net综合考虑了spatial和channel-wise响应来计算关键点得分，采用了ratio-to-max（公式4）来估计channel-wise extremeness，然而可能存在一个限制：它只与沿通道的所有响应的实际分布有微弱的关系。&#x20;

为了考察这种影响，作者首先将公式4改为了channel-wise softmax，但是这一修改恶化了表现。所以，作者采用了peakness作为D2-Net中的keypoint检测，公式4被重写为：&#x20;

![](<../../.gitbook/assets/image (841).png>)

其中softplus将peakness转换为一个正值。为了平衡两个分数的尺度，公式3被重写为：&#x20;

![](<../../.gitbook/assets/image (348).png>)

然后两种分数如公式5一样再次结合。&#x20;

**Multi-level keypoint detection (MulDet)** 如前文所述，D2-Net的限制在于关键点定位精度不够，因为特征点是在低分辨率的特征图上检测的。有很多恢复分辨率的方法，比如训练一个额外的特征detector（SuperPoint）、使用膨胀卷积（R2D2)。但是这些方法会导致大量需要训练的参数，耗费很多GPU内存和计算资源。因此，作者提出了利用卷积神经网络的inherent pyramidal feature hierarchy，并结合多个特征层来进行检测特征。&#x20;

![](<../../.gitbook/assets/image (544).png>)

当给定一个包含不同层特征图$$\{y^{(1)},y^{(2)},...,y^{(l)}\}$$的feature hierarchy，每个特征图缩放比例为$$\{1,2,...,2^{(l-1)}\}$$。利用上面提到的检测方法在每一层获得一个score map $$\{s^{(1)},s^{(2)},...,s^{(l)}\}$$。接着，每个特征图被上采样到与原图一样的分辨率，最后加权求和：&#x20;

![](<../../.gitbook/assets/image (325).png>)

这种结构有三个好处：首先它隐式地使用了多尺度检测，符合传统的尺度空间理论中用不同感受野来定位特征的规律；其次，与U-Net相比，这种结构不需要引入额外的参数来恢复空间分辨率，来实现pixel-wise accuracy；第三，与U-Net直接融合低层特征和高层特征不同，这一结构保留低层特征不变，但是融合了检测出的多层语义信息，帮助更好地保留低层结构，如角点和边等。

### Learning Framework

#### Network architecture

![](<../../.gitbook/assets/image (308).png>)

网络结构如上，为了减少计算，用L2-Net代替了D2-Net中的VGG backbone。与R2D2相似，将最后的8x8卷积换成了三个3x3的卷积，得到1/4分辨率的128维特征图。最后三个卷积层conv6,conv7,conv8用DCN来代替，在conv1,conv3,conv8这三层中进行MulDet。公式13中的权重设为$$w_i=3,2,1$$.

#### Loss design

利用真值depths和相机参数来将$I$稠密地warp到I'，获得匹配C。利用D2-Net的loss来同时训练detector和descriptor：&#x20;

![](<../../.gitbook/assets/image (371).png>)

其中$$\hat{s_k}$$和$$\hat{s'_k}$$是公式13中得到的分数。$$M(\cdot, \cdot)$$是ranking loss。与使用hardest-triplet loss的D2-Net不同，作者采用了FCGF中的hardest-contrastive形式：&#x20;

![](<../../.gitbook/assets/image (37).png>)

其中D度量了描述子间的欧式距离，$$m_p=0.2，m_n=1.0$$。与D2-Net一样，设置安全半径为3，来避免将过近的点选做负样本。

### Experiments

#### Evaluation Protocols

**HPatches** keypoint repeatability: 所有可能的匹配与共视关键点的最小数量之间的比值；matching score：正确匹配与共视关键点的最小数量之间的比值；mean matching accuracy：正确匹配与所有可能匹配之间的比值。**A match is defined to correspond if the point distance is below some error threshold after homography wrapping, and a correct match is further required to be a mutual nearest neighbor during brute-force searching (感觉说反了吧...)**

#### Results

![](<../../.gitbook/assets/image (183).png>)

![](<../../.gitbook/assets/image (239).png>)

![](<../../.gitbook/assets/image (518).png>)

![](<../../.gitbook/assets/image (1032).png>)

![](<../../.gitbook/assets/image (1051).png>)
