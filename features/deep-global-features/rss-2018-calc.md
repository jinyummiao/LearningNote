---
description: Lightweight Unsupervised Deep Loop Closure
---

# \[RSS 2018] CALC

{% embed url="https://arxiv.org/abs/1805.07703.pdf" %}

{% embed url="https://github.com/rpng/calc" %}

### Abstract

这篇论文提出了一种无监督的神经网络CALC，采用autoencoder的结构，但是重建的不是原始图像，而是图像的HoG描述子。

### Method

这篇论文作者设计了一个可以将高维的原始图像信息映射到低维特征空间的网络，对场景变化不敏感，训练方法不需要标注图像。 训练pipeline如下：&#x20;

![](<../../.gitbook/assets/image (377).png>)

训练集中的每张图像被缩小到120x160，灰度图。通过projective transformations获得匹配图像。 HOG特征对网络提供了一个先验的几何约束，网络可以获得光照不变性，通过projective transformations来获得HOG所不具备的视角不变性。 获得训练的方法-projective transformations过程如下：&#x20;

![](<../../.gitbook/assets/image (331).png>)

这个过程的目的是：根据一张真实图像I，通过随机的2D projective transformation获得一系列描述同场景、但视角不同的图像。 对于每张图像，从其四角的某一区域内各随机选一个点，作为生成图像的四个角点，获得四个点后，就可以获得从原图到生成图像的homograph，矫正之后就生成了新图像。&#x20;

![](<../../.gitbook/assets/image (814).png>)

### Performance

<img src="../../.gitbook/assets/image (138).png" alt="" data-size="original"><img src="../../.gitbook/assets/image (336).png" alt="" data-size="original">

![](<../../.gitbook/assets/image (998).png>)

![](<../../.gitbook/assets/image (169).png>)
