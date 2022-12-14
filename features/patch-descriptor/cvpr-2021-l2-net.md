---
description: 'L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space'
---

# \[CVPR 2021] L2-Net

{% embed url="https://ieeexplore.ieee.org/document/8100132" %}

{% embed url="https://github.com/yuruntian/L2-Net" %}

### Abstract

这篇论文提出了一种在欧拉空间中表现很好的CNN特征，突出点有四点：（1）提出了一种渐进的采样策略，使得网络可以在很少epoch内获得数以亿计的训练样本；（2）重视descriptor之间的相对距离；（3）对中间的feature map有格外的监督；（4）考虑descriptor的compactness。

### Architecture

![](<../../.gitbook/assets/image (512).png>)

网络结构如上图，在每个卷积层后加入BN层，作者设置BN层的权重为1，方差为0，不进行更新。在输出层，作者使用了local response normalization layer（LRN）来输出单位des。L2-Net将32 x 32的patch转变为128维的des。作者使用了central-surround（CS） L2-Net，将两个L2-Net并联，左边的网络输入的是原始输入，右边的网络输入的是原本patch经过crop和resize后的中心部分（据说是可以处理scale不变性）。

### Training data

L2-Net用Hpatches和Brown数据集进行训练，这两个数据集都提供了matched patches。每个patch都有唯一的3D点索引，而具有相同3D点索引的patches是相互匹配的。&#x20;

在加载训练数据时，从P个3D点中挑选p1个点，然后从P-p1中挑选p2个点，得到了p(p1+p2)个点，对每个p，随机获取一个匹配的patch，这样就获得了2p个训练数据作为网络的输入，记为X，X是32 x 32 x 2p维的。网路的输出记为Y，Y是128 x 2p维的。由于网络的输出经过了normalization，所以可以定义距离矩阵$$D=\sqrt{2(1 - Y_1^T Y_2)}$$。&#x20;

D中包含了p x p个pairs，对角线上的p个pair是positive matched pair，非对角线上的pair是negative pairs。**(Q: 点都是随机取得，那在原本的p点中就可能会出现相互匹配的点，那么非对角线上的pair也不一定都是negative吧)**

### Loss function

损失函数由三部分构成，第一部分，作者利用相对距离去约束匹配和非匹配的pair；第二部分，作者强调descriptor的紧凑性，即descriptor的各维信息之间应该没有相互关系；第三部分，作者对中间的feature map进行了约束。

#### descriptor similarity

descriptor之间的相互距离在pair是匹配的时候最小，所以体现在D中，就是对角线上的元素应当是行、列中最小的。定义行相似矩阵和列相似矩阵：&#x20;

![](<../../.gitbook/assets/image (204).png>)

$$S^c$$可以理解为$$y_2$$匹配到$$y_1$$的概率，$$S^r$$可以理解维$$y_1$$匹配到$$y_2$$的概率。为了让匹配的descriptor之间距离减小，loss设计为：&#x20;

![](<../../.gitbook/assets/image (1060).png>)

#### descriptor compactness

作者发现过拟合问题是由于descriptor各维度之间的correlation（我理解的是des出现了冗余）。所以作者加入了对des compactness的考虑。&#x20;

作者设计了一个correlation matrix R：&#x20;

![](<../../.gitbook/assets/image (50).png>)

其中$$b_i$$表示q个patch的des中第i维元素的集合，是个行向量。（我理解是对des的每一维减掉均值后，计算cosine相似度，r=0，说明计算的两维des间正交，不相关）所以R的非对角线位置的元素需要靠近0。所以des compactness的loss为：&#x20;

![](<../../.gitbook/assets/image (319).png>)

如果加入了LRN后，每维数据的均值为0，所以$$R_s$$的计算可以简化为：&#x20;

![](<../../.gitbook/assets/image (673).png>)

**Intermediate feature maps**

用$$E_1$$的loss去计算网络中间的feature map上的similarity matrix G，然后构建了loss：&#x20;

![](<../../.gitbook/assets/image (144).png>)

作者称之为Discriminative Intermediate Feature maps (DIF)，并且发现在BN层后面加入DIF会提升效果，所以作者在第一个和最后一个BN层后计算了DIF。

### Performance

![](<../../.gitbook/assets/image (997).png>)

![](<../../.gitbook/assets/image (1059).png>)

![](<../../.gitbook/assets/image (541).png>)
