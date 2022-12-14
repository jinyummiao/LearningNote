---
description: Unifying Deep Local and Global Features for Image Search
---

# \[ECCV 2020] DELG

{% embed url="https://link.springer.com/chapter/10.1007/978-3-030-58565-5_43" %}

{% embed url="https://github.com/feymanpriv/DELG" %}

{% embed url="https://github.com/tensorflow/models/tree/master/research/delf" %}

## Abstract

In this work, our key contribution is to unify global and local features into a single deep model, enabling accurate retrieval with efficient feature extraction. We refer to the new model as DELG, standing for DEep Local and Global features.We leverage lessons from recent feature learning work and propose a model that combines generalized mean pooling for global features and attentive selection for local features. The entire network can be learned end-to-end by carefully balancing the gradient flow between two heads – requiring only image-level labels. We also introduce an autoencoder-based dimensionality reduction technique for local features, which is integrated into the model, improving training efficiency and matching performance.

## Introduction

本文的贡献如下：（1）提出了一个统一的模型，可以用一个CNN提取局部特征和全局特征，称为DELG。通过该模型，可以用单一模型实现全局特征提取、关键点提取和局部描述子计算，非常高效。作者通过generalized mean pooling和attentive local feature detection，让该模型可以很好地利用CNN自带的层次化图像表征能力；（2）作者用一个卷积autoencoder模块来提取低维局部特征，这样的处理易于嵌入统一模型中，不需要如PCA等后处理步骤；（3）作者提出一种只需要图像级别监督的端到端训练过程，这需要在反向传播过程中很仔细地控制全局网络和局部网络间的梯度流。

## DELG

### Design Consideration

为了最优的表现，图像检索算法应该能够理解目标的类型，分辨出用户可能感兴趣的目标，这样系统能够区分相关目标和背景。局部特征和全局特征都应该关注于图像中的有区分性的信息。但是两种特征存在着本质上的差异，给联合训练带来了挑战。

描述具有相同感兴趣目标的图像的全局特征应该是相似的，否则全局特征不相似。这需要高层次的、抽象的表征，对视角和光度变化不敏感；局部特征需要对图像的特定区域进行编码，关键点应当对视角不敏感，描述子应当对视觉信息进行编码，这些性质对使用局部特征进行几何一致性验证非常重要。

此外，作者希望模型可以端到端训练，无需要额外的训练过程。

### Model

![](<../../.gitbook/assets/image (264).png>)

模型结构如图1。全局特征来自较深层特征，具备高层次视觉信息；局部特征来自中间层特征，对局部区域的信息进行编码。

输入一张图像，通过一个CNN得到两个特征图：$$\mathcal{S} \in R^{H_S \times W_S \times C_S}$$​和$$\mathcal{D} \in R^{H_D \times W_D \times C_D}$$，分别为浅层和深层的特征。一般而言，$$H_D \le H_S$$,$$W_D \le W_S$$,$$C_D \ge C_S$$。这些特征经过ReLU处理，都是非负的。​

为了聚合深层特征得到全局特征，作者用了generalized mean pooling （GeM），对各个特征进行加权。学习全局特征的另一个关键点在于whiten聚合特征，作者用一个带有可学习偏差$$b_F$$的全连接层$$F \in R^{C_F \times C_D}$$​来实现这一点。经过这两步，得到了一个全局特征$$g \in R^{C_F}$$​:

![](<../../.gitbook/assets/image (229).png>)

其中p为generalized mean power参数。

对于局部特征来说，选取匹配相关区域中的特征是很重要的。这可以通过注意力模块M来实现。A=M(S)，其中M是一个小的卷积神经网络，$$A \in R^{H_s \times W_s}$$表示与S相关的注意力图。​

然后作者用了一个小的卷积自编码模块AE来给局部特征降维，得到局部特征L=T(S)，其中$$L_S \in R^{H_s \times W_s \times C_T}$$。L不需要非负。

全局和局部特征都需经过L2正则化。

### Training

![](<../../.gitbook/assets/image (273).png>)

**全局特征**训练中，we adopt a suitable loss function with L2-normalized classifier weights $$\hat{W}$$ , followed by scaled softmax normalization and cross-entropy loss (也称为cosine classifier)。此外，作者还用了ArcFace margin来引入较小的类内方差。具体的来说，对于全局描述子$$\hat{g}$$，首先计算其与$$\hat{W}$$​的余弦相似度，用ArcFace margin进行调节：

![](<../../.gitbook/assets/image (270).png>)

其中u是余弦相似度，m是ArcFace margin，c是判断是否分类正确的二进制标签。而交叉熵损失可以用softmax计算：

![](<../../.gitbook/assets/image (269).png>)

其中$$\gamma$$​是可学习的标量。

**局部特征**训练中，作者用了两个损失。首先，用均方差回归损失来度量AE是否能够恢复出S。记S'=T'(L)为恢复出的S，其中T'是一个有$$C_s$$​个卷积核的1x1卷积层加ReLU，该部分损失函数为：

![](<../../.gitbook/assets/image (237).png>)

其次，用交叉熵损失来激励注意力模块挑选局部特征。首先用注意力权重来pooling恢复后的S’：

![](<../../.gitbook/assets/image (290).png>)

然后采用标准的交叉熵损失：

![](<../../.gitbook/assets/image (245).png>)

其中和$$b_i$$都是类别i的分类器权重和偏差。这可以让具有区分性的特征的注意力权重变大。

总的损失为：$$L_g+\lambda L_r + \beta L_a$$

**梯度控制**：Naively optimizing the above-mentioned total loss experimentally leads to suboptimal results, because the reconstruction and attention loss terms significantly disturb the hierarchical feature representation which is usually obtained when training deep models. In particular, both tend to induce the shallower features S to be more semantic and less localizable, which end up being sparser. Sparser features can more easily optimize $$L_r$$, and more semantic features may help optimizing $$L_a$$; this, as a result, leads to underperforming local features.

为了解决这一问题，作者停止$$L_r$$​和$$L_a$$​的梯度传播到backbone网络S，即只用$$L_g$$​来优化backbone。

## Experiments

推理过程：For global features, we use 3 scales, $$\{ \frac{1}{\sqrt{2}}, 1, \sqrt{2}\}$$; L2 normalization is applied for each scale independently, then the three global features are average-pooled, followed by another L2 normalization step. For local features, we experiment with the same 3 scales, but also with the more expensive setting from DELF using 7 image scales in total, with range from 0:25 to 2:0 (this latter setting is used unless otherwise noted). Local features are selected based on their attention scores A; a maximum of 1k local features are allowed, with a minimum attention score $$\tau$$ , where we set  to the median attention score in the last iteration of training, unless otherwise noted. For local feature matching, we use RANSAC with an affine model. When re-ranking global feature retrieval results with local feature-based matching, the top 100 ranked images from the first stage are considered. For retrieval datasets, the final ranking is based on the number of inliers, then breaking ties using the global feature distance. For the recognition dataset, we follow a protocol to produce class predictions by aggregating scores of top retrieved images, where the scores of top images are given by $$\frac{min(70,i)}{70}+\alpha c$$ (here, i is the number of inliers, c the global descriptor cosine similarity and $$\alpha=0.25$$).

局部特征消融实验：首先证明了AE降维的有效性，其次证明了需要控制梯度传播。

![](<../../.gitbook/assets/image (279).png>)

作者分析了不控制梯度时局部特征表现下降的原因：The poor performance of the naive jointly trained model is due to the degradation of the hierarchical feature representation. This can be assessed by observing the evolution of activation sparsity in S (conv4) and D (conv5), as shown in Fig. 3. Generally, layers representing more abstract and high-level semantic properties (usually deeper layers) have high levels of sparsity, while shallower layers representing low-level and more localizable patterns are dense. As a reference, the ImageNet pre-trained model presents on average 45% and 82% sparsity for these two feature maps, respectively, when run over GLD images. For the naive joint training case, the activations of both layers quickly become much sparser, reaching 80% and 97% at the end of training; in comparison, our proposed training scheme preserves similar sparsity as the ImageNet model: 45% and 88%. This suggests that the conv4 features in the naive case degrade for the purposes of local feature matching; controlling the gradient effectively resolves this issue.

![](<../../.gitbook/assets/image (238).png>)

全局特征消融

![](<../../.gitbook/assets/image (257).png>)

与先进算法比较：

![](<../../.gitbook/assets/image (295).png>)

![](<../../.gitbook/assets/image (223).png>)

![](<../../.gitbook/assets/image (216).png>)

![](<../../.gitbook/assets/image (287).png>)
