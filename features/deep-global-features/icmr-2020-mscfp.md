---
description: Image Retrieval using Multi-scale CNN Features Pooling
---

# \[ICMR 2020] MSCFP

{% embed url="https://arxiv.org/abs/2004.09695" %}

### Abstract

作者基于NetVLAD进行了多尺度特征的pooling，得到一个高效的图像表征，用于基于内容的图像检索。

### Introduction

在这篇论文中，作者提出了一种多尺度CNN regions pooling方法，在NetVLAD aggregation之前对特征进行聚合。该方法与3-stream siamese network一起利用end-to-end方法进行训练，来优化图像表征。这篇论文还提出了一种triplet mining过程，来提供不同的semi-hard和hard triplets，避免hard triplets阻碍学习。

### The Proposed Method

本篇论文与现有算法的主要区别在于：1）特征利用两种不同的方法聚合，首先对两个尺度的特征进行max-pooling聚合，然后利用VLAD聚合；2）提出triplet mining方法来训练3-stream siamese，选取semi-hard和hard triplets，避免采集到那些由于重叠部分少、剧烈缩放造成的视觉相似度非常低的extremely hard样本，这些样本可能导致过拟合和丢掉泛化性。

#### Pooling of local CNN features

![](<../../.gitbook/assets/image (516).png>)

分别用2x2 max-pooling和3x3 max-pooling（步长都为1）来对卷积层得到的feature map进行pooling，得到精细的和更大感受野的细节信息。每一点上，得到f个activation maps，视为一个1x1xf的column feature。这个过程如图2所示，类似于稠密的grid-based采样描述子。column feature被拼接在一起，来获得图像的multi-scale描述子。&#x20;

![](<../../.gitbook/assets/image (697).png>)

所有局部CNN特征用NetVLAD层聚合，得到图像内容的全局描述子。NetVLAD层用K-Means聚类初始化。在NetVLAD中，设置K=64，得到32k维的描述子。&#x20;

![](<../../.gitbook/assets/image (813).png>)

#### Training and Triplet Mining

这篇论文中，作者使用基于图像三元组的ranking loss。这一方法的基本意图为学习一个让相关图像的表征比不相关图像的表征更相似的描述子。使用平方欧式距离来度量描述子间的相似度，triplet loss表述为：$$L=max(\alpha+d(q,p)-d(q,n),0)$$，其中$$\alpha=0.1$$。&#x20;

在该方法中一个很重要的问题是如何选择triplets，随机的选择可能会导致triplet不产生任何的loss，因此无法优化模型。作者提出triplet会根据它们自身的复杂度来对训练过程产生不同的影响。因此，作者将triplet划分为以下几类：&#x20;

**easy triplets**：$$d(q,p)<d(q,p)+\alpha<d(q,n)$$对模型没有帮助；&#x20;

**semi-hard triplets**：$$d(q,p)<d(q,n)$$但是$$d(q,p)+\alpha>d(q,n)$$，这些triplet会提供一些有用信息，但是不够多&#x20;

**hard triplets**：$$d(q,n)<d(q,p)$$会产生较大的loss。&#x20;

![](<../../.gitbook/assets/image (837).png>)

使用算法1(**这个伪代码感觉有问题吧...**)可以用如下逻辑规则以50%的概率产生semi-hard triplets或hard triplets：&#x20;

**case A**：关于query搜索第一个negative的索引j。如果j>2，那么设置positive的索引为j-1（第15行），产生一个semi-hard triplet；&#x20;

**case B**：否则，在第一个negative之后搜索第一个positive的索引值，产生一个hard triplet（第18-19行）；&#x20;

**case C**：处理非常困难的triplets，这时positive和query相距很远（第21行），例如第一个positive的索引值在阈值t之外，这种情况下triplets被丢弃，因为可能会引起过拟合和较差的泛化性。 算法1中，类别数量k被视为512，mining\_batch\_size设为2048，每个类别只挑选一个triplet（第28行），平均产生250个triplet。&#x20;

作者用Google Landmark V2数据集进行训练，用cleaned版本中的train数据集，包含1580470张图像和81313个标签，每8次迭代进行一次mining。

### Experiments

![](<../../.gitbook/assets/image (25).png>)

![](<../../.gitbook/assets/image (845).png>)

![](<../../.gitbook/assets/image (529).png>)
