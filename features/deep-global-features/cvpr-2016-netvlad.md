---
description: 'NetVLAD: CNN architecture for weakly supervised place recognition'
---

# \[CVPR 2016] NetVLAD

{% embed url="https://www.di.ens.fr/willow/research/netvlad" %}

### Abstract

很经典的论文，主要贡献是传统全局描述子VLAD的启发，设计了一个generalized VLAD层，可以实现end-to-end训练，这一结构可以很容易地嵌入任何CNN结构，通过反向传播进行训练，得到的表征通过PCA处理进行压缩，得到最后的描述子；采用弱监督的方法进行训练，通过Google Street View Time Machine中采集的不同时间同一场景的图像来进行训练，算是比较早期使用metric learning进行训练的工作了。

### Deep architecture for place recognition

大多数图像检索流程都是(1)先从图像中提取局部特征, (2)然后以一种无序的方法将其pool为一个向量或矩阵。这样做，可以提供较好的平移和遮挡不变性。而对于光照和视角变化的不变性则来自于描述子自己，尺度不变性来自于multi-scale提取描述子。&#x20;

作者设计了一个end-to-end的表征方法，针对(1)，从CNN中得到$$H \times W \times D$$输出特征图，视作在$$H \times W$$的空间中提取的D维描述子。针对(2)，作者设计了一个新的pooling层，可以将描述子pool为固定尺寸的图像表征，并可以通过BP优化参数。

#### NetVLAD： A Generalized VLAD layer

BoW保留的是视觉单词的数量，而VLAD保留的是每个视觉单词的残差（描述子与对应cluster之间的距离）和。当给定N个D维局部描述子$$x_i$$，和K个聚类中心（视觉单词）$$c_k$$。VLAD图像描述子V是$$K \times D$$维的(**看后文的表述，应该是**$$D \times K$$**吧，可能写错了**）。为了方便，这里将V写为一个$$K \times D$$矩阵，但是在normalization后将矩阵转换为一个向量，作为图像的表征。V矩阵中(j,k)的元素为：&#x20;

![](<../../.gitbook/assets/image (1045).png>)

其中$$x_i(j)$$和$$c_k(j)$$分别为第i个描述子和第k个聚类中心的第j维元素。$$a_k(x_i)$$为描述子$$x_i$$是否属于第k个描述子的条件变量，是则为1，否则为0。直观来讲，V矩阵中每个D维的列k记录了属于聚类$$c_k$$的所有描述子到聚类中心距离$$(x_i-c_k)$$的和。然后，矩阵V被以列为单位的L2-normalization(intra-normalization)，并转换成一个向量，最后整体L2-normalization。&#x20;

为了让整个过程实现end-to-end训练，需要让VLAD pooling变得可微。而VLAD中不连续的地方在于描述子$$x_i$$到聚类中心$$c_k$$的hard assignment $$a_k(x_i)$$，所以作者用下式代替：&#x20;

![](<../../.gitbook/assets/image (8).png>)

其中描述子$$x_k$$到聚类中心$$c_k$$分配的权重不光受它们之间距离的影响，也受描述子到其他聚类中心距离的影响。$$\overline{a_k}(x_i)$$取值在0到1之间，最近的聚类中心被分配为最高的权重。$$\alpha$$是一个正常数。当$$\alpha \rightarrow \infty$$时，soft assignment就是hard assignment。 观察上式，可以化简：&#x20;

![](<../../.gitbook/assets/image (36).png>)

其中向量$$w_k=2\alpha c_k$$，标量$$b_k=-\alpha {||c_k||}^{2}$$. 最后得到的形式为：&#x20;

![](<../../.gitbook/assets/image (338).png>)

其中$$w_k, b_k, c_k$$是每个聚类k需要学习的参数。&#x20;

![](<../../.gitbook/assets/image (378).png>)

与传统的VLAD相比，NetVLAD的参数是靠学习得到的，具有更强的适应性。 网络结构图如下：&#x20;

![](<../../.gitbook/assets/image (306).png>)

### Learning from Time Machine data

![](<../../.gitbook/assets/image (210).png>)

作者利用谷歌街景来获得训练数据，数据只具有粗糙的GPS信息。对于每个query，作者利用GPS信息获得可能的positive $$p^{q}_{i}$$ (可能地理距离很近，但是不同朝向)，和获得确定的negatives $$n^{q}_{j}$$.

#### Weakly supervised triplet ranking loss

由于无法确定可能的positive是否是正确的，所以先挑选最匹配的可能的positive：&#x20;

![](<../../.gitbook/assets/image (872).png>)

简单来说，triplet loss的作用就是让网络学习一种映射，使得query和positive之间的距离要小于query和negatives之间的距离，因此，总的loss为：&#x20;

![](<../../.gitbook/assets/image (480).png>)

其中$$l(x)=max(x,0)$$，m是一个超参。

### Experiments

评价规则：图像检索，判断top-N的candidate中是否有正确的，用recall来评价。与baselines的比较，经过PCA降维到4096维后的描述子(-\*-)和全维度的描述子(-o-)效果差不多：&#x20;

![](<../../.gitbook/assets/image (809).png>)

![](<../../.gitbook/assets/image (688).png>)

作者还做了实验，来看应该训网络的那些部分，发现VLAD层带来的提升最多，不需要完全训练整个网络：&#x20;

![](<../../.gitbook/assets/image (690).png>)
