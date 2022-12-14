---
description: 'LoFTR: Detector-Free Local Feature Matching with Transformers'
---

# \[CVPR 2021] LoFTR

{% embed url="https://zju3dv.github.io/loftr" %}

### Abstract

这篇论文提出提出了一个局部图像特征匹配的方法，不同于传统的特征提取、描述再匹配的顺序，作者提出先在coarse level中构建像素级别的稠密匹配，然后在fine level中进一步调整匹配。与利用cost volume来搜索correspondence的稠密匹配方法不同，作者在Transformer中使用了self attention和cross attention来获得在两幅图像中都被约束的特征描述子。Transformer提供的全局感受野使得该算法可以在少纹理区域也可以获得稠密的匹配，这些区域对于传统的特征检测算法来说很难检测到可重复的点。

### Introduction

人类在低纹理区域（indistinctive regions）中搜索匹配时候，不光基于局部的图像邻域，而且基于全局的上下文视觉信息。因此，一个大的感受野对于特征检测模型是很重要的。 作者提出了Local Feature Transformer（LoFTR），一个detector-free的局部特征匹配模型。受SuperGlue模型的启发，作者使用了带有self-attention和cross-attention的Transformer来处理从CNN backbone中提取的局部特征。模型先在低分辨率（1/8）的特征上构建稠密匹配。具有高可信度的匹配被挑选出来并在亚像素级别上通过correlation-based方法进行调整。Transformer的全局感受野和位置编码使得处理后的特征（transformed feature）表征同时具有内容和位置信息。通过多次交替使用self-attention和cross-attention层，LoFTR从真值中学到了稠密排列的全局一致的匹配先验知识。线性transformer被用于减少计算复杂度。

### Methods

![](<../../.gitbook/assets/image (150).png>)

#### Local Feature Extraction

作者使用了一个标准的带有FPN结构的卷积结构（CNN）来对两张图像提取多尺度特征。作者用$$\tilde{F}^{A}$$和$$\tilde{F}^{B}$$来表示1/8分辨率的coarse-level特征，而$$\hat{F}^{A}$$和$$\hat{F}^{B}$$表示1/2分辨率的fine-level特征。CNN具有平移等值和局部性的归纳偏差，非常适合局部特征的提取。CNN引入的下采样也减少了LoFTR模块的输入长度，这对于控制计算消耗至关重要。

#### Local Feature Transformer (LoFTR) Module

提取完局部特征后，$$\tilde{F}^{A}$$和$$\tilde{F}^{B}$$被输入到LoFTR模块中来提取包含位置信息和内容信息的局部特征。直观地讲，LoFTR模块将特征转变为更易于匹配的特征表征。转变后的特征被记为$$\tilde{F}^{A}_{tr}$$和$$\tilde{F}^{B}_{tr}$$

**Preliminaries： Transformer**

![](<../../.gitbook/assets/image (863).png>)

Transformer Encoder是由顺序连接的encoder层构成的。如图3(a)所示，encoder层中的关键元素是注意力层。注意力层的输入一般被命名为query、key和value。与信息检索类似，query向量Q根据每个value V对应的key向量K与Q的点积计算出的注意力权值，从value向量V中检索信息。注意力层的计算图如图3(b)所示。用公式可以写为：&#x20;

![](<../../.gitbook/assets/image (1055).png>)

直观地讲，注意力机制通过测量query元素与每个key元素之间的相似度来选择相关信息。输出的信息是用相似性加权的value向量。如果相似性高，则vector中相关信息被提取出。这一过程，在GNN里也被称为“信息传递”。

**Linear Transformer**

假设Q和K的长度为N，它们的特征维度为D，那么Transformer中Q和K之间的点乘的计算复杂度是二次方增长的$$(O(N^2))$$。在特征匹配任务中，即使输入长度被CNN缩短，使用初始版本的Transformer也是不可取的。线性Transformer可以将Transformer的计算复杂度减为$$(O(N))$$，它利用一个核函数$$sim(Q,K)=\phi(Q)\cdot {\phi(K)}^T$$来代替传统Transformer中使用的exponential kernel，其中$$\phi(\cdot)=elu(\cdot)+1$$，计算图如图3(c)所示。利用矩阵乘法的结合性，可以先进行$${\phi(K)}^T$$与V之间的乘法。因为$$D\ll N$$，所以计算消耗被减小为$$(O(N))$$.

**Positional Encoding**

作者参考了DETR中的标准位置编码的2D拓展形式的，用于transformer中。与DETR不同，作者只把它们降到backbone输入一次。作者使用了absolute sinusoidal位置编码的2D拓展形式：&#x20;

![](<../../.gitbook/assets/image (1050).png>)

其中$$\omega_k=\frac{1}{10000^{2k/d}}$$，d是特征的通道数，i是特征通道的索引值。直观地讲，位置编码以正弦形式为每个元素提供唯一的位置信息。通过对$$\tilde{F}^{A}$$和$$\tilde{F}^{B}$$加入位置编码，transformed feature因此可以包含位置信息，这可以帮助LoFTR在indistinctive区域中产生匹配。如图4(c)所示，虽然白墙上的像素具有相似的RGB信息，但是每个位置上的transformed features $$\tilde{F}^{A}_{tr}$$和$$\tilde{F}^{B}_{tr}$$都是唯一的。&#x20;

![](<../../.gitbook/assets/image (888).png>)

**Self-attention and Cross-attention layers**

对于每个self-attention层，输入的特征$$f_i$$和$$f_j$$是一样的（**就是intra-image的吧**）。对于cross-attention层，输入的特征$$f_i$$和$$f_j$$是( $$\tilde{F}^{A}$$和$$\tilde{F}^{B}$$)或( $$\tilde{F}^{B}$$和$$\tilde{F}^{A}$$)，取决于cross-attention的方向。与SuperGlue相似，作者交替使用self-attention和cross-attention $$N_c$$次。

#### Establishing Coarse-level Matches

在LoFTR中使用了两种可微的匹配层，一种是optimal transport层（OT），另一种是dual-softmax操作。transformed features之间的分数矩阵S由$$S(i,j)=\frac{1}{\tau}\cdot <\tilde{F}^{A}_{tr},\tilde{F}^{B}_{tr}>$$来计算。当用OT来进行匹配时，S被用作partial assignment的损失矩阵，与SuperGlue中一样。我们也可以对S的两个维度上都进行softmax来获得双向最近邻匹配的概率，以下称为dual-softmax。当使用dual-softmax时，匹配概率$$P_c$$计算如下：&#x20;

![](<../../.gitbook/assets/image (558).png>)

**Matching**

基于可信度矩阵$$P_c$$，我们挑选出可信度高于$$\theta_c$$的匹配，然后加入互为最近邻匹配的限制，剔除了可能的外点。记coarse-level匹配为：&#x20;

![](<../../.gitbook/assets/image (196).png>)

#### Coarse-to-Fine Module

当建立了粗略匹配后，这些匹配通过coarse-to-fine模块调整到原图像分辨率。作者用了一个基于correlation的方法来实现这一点。对于每对粗略检索$$(\tilde{i},\tilde{j})$$，首先在fine-level特征图中找到对应的位置$$(\hat{i},\hat{j})$$，然后裁剪两块$$w \times w$$的局部滑窗。一个较小的LoFTR模块将滑窗内的特征转化$$N_f$$次，得到中心点在$$\hat{i}$$和$$\hat{j}$$的两个transformed feature maps $$\hat{F}^{A}_{tr}(\hat{i})$$和$$\hat{F}^{B}_{tr}(\hat{j})$$。然后作者将$$\hat{F}^{A}_{tr}(\hat{i})$$的中心向量与$$\hat{F}^{B}_{tr}(\hat{j})$$中的所有向量相关联，产生一个热力图，表示$$\hat{i}$$与$$\hat{j}$$邻域内点匹配的可能性。通过计算概率分布的期望，可以获得$$I^B$$上亚像素精度的最终位置$$\hat{j'}$$。最后的fine-level匹配$$M_f$$就是$$\{(\hat{i},\hat{j'})\}$$.

#### Supervision

最后的loss包括coarse-level和fine-level的loss：$$L=L_c+L_f$$.

**Coarse-level Supervision**

coarse-level的损失是可信度矩阵$$P_c$$上去负对数似然值。类似于SuperGlue，作者在训练时利用相机位姿和深度图来计算可信度矩阵的真值标签。定义真值粗略匹配$$M^{gt}_c$$为两组1/8分辨率栅格的双向最近邻匹配。两个栅格间的距离通过他们中心位置的重投影误差来定义。由于这部分的匹配是在1/8分辨率的栅格上获取的，可能会出现一对多的匹配。在实践中，作者选取左图中一个1/8栅格的中心位置，将其投影到相同尺度的深度图中，获得它的深度。基于深度和已知的相机位姿，将这个栅格中心warp到右图中，选择最近的1/8栅格中心作为一个候选匹配。然后保留彼此为最近邻匹配的匹配作为最后的真值。在右图到左图的过程中也采用相同的操作。当使用OT层时，作者使用了与SuperGlue一样的损失函数。当使用dual-softmax时，作者对$$M^{gt}_c$$中的栅格计算负梯度损失：&#x20;

![](<../../.gitbook/assets/image (156).png>)

**Fine-level Supervision**

作者使用L2损失进行fine-level微调。对于每个query $$\hat{i}$$，通过计算对应热力图的整体方差$$\sigma^2(\hat{i})$$来度量它的不确定性。优化的目标是让微调后的位置具有较低的不确定性：&#x20;

![](<../../.gitbook/assets/image (850).png>)

其中$$\hat{j'}_{gt}$$是将每个$$\hat{i}$$利用真值位姿和深度从$$\hat{F}^{A}_{tr}(\hat{i})$$warp到$$\hat{F}^{B}_{tr}(\hat{j})$$得到的。在计算$$L_f$$时，如果$$\hat{i}$$warp后的位置超出了$$\hat{F}^{B}_{tr}(\hat{j})$$滑窗的范围，就丢弃这对匹配$$(\hat{i},\hat{j'})$$. The gradient is not backpropagated through $$\sigma^2(\hat{i})$$during training.

#### Implementation Details

和SuperGlue一样，作者在MegaDepth上训练了室外的模型，在ScanNet上训练了室内的模型。$$N_c=4，N_f=1，\theta_c=0.2，w=5$$。在传入fine-level LoFTR之前，上采样$$\tilde{F}^{A}_{tr}$$和$$\tilde{F}^{B}_{tr}$$并与$$\hat{F}^{A}$$和$$\hat{F}^{B}$$拼接在一起。

#### Experiments

**Timing**

![](<../../.gitbook/assets/image (330).png>)

**Homography Estimation**

图像缩放到480，LoFTR选取top-1K特征。&#x20;

![](<../../.gitbook/assets/image (859).png>)

**Relative Pose Estimation**

![](<../../.gitbook/assets/image (565).png>)

![](<../../.gitbook/assets/image (818).png>)

**Visual Localization**

![](<../../.gitbook/assets/image (879).png>)

![](<../../.gitbook/assets/image (551).png>)

**Ablation Study**

![](<../../.gitbook/assets/image (164).png>)
