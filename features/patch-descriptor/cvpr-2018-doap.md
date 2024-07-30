---
description: Local Descriptors Optimized for Average Precision
---

# \[CVPR 2018] DOAP

{% embed url="https://openaccess.thecvf.com/content_cvpr_2018/papers/He_Local_Descriptors_Optimized_CVPR_2018_paper.pdf" %}

{% embed url="https://kunhe.github.io/papers/DOAP/index.html" %}

### Abstract

特征匹配，本质上就是特征描述子的最近邻检索。在这篇工作中，作者用神经网络直接优化ranking-based retrieval的指标，Average Precision。这一通用的方法可以视作比local ranking方法更先进的listwise learning方法。

### Introduction

观察特征匹配任务，作者发现这其实是一个最近邻检索问题。因此，作者基于对ranking-based retreval评价指标Average Precision（AP）的直接优化提出一种listwise learning方法，通过训练特征描述子来学习对特征进行排序（匹配）。本文的特征有二进制和浮点数两种形式的描述子。&#x20;

本文工作的一大特点是其通用性，即模型直接优化的是与任务无关的最近邻匹配阶段。虽然如此，为了获得用于特征匹配的描述子，作者还是加入了一些专用于某些任务的改进。首先，作者利用Spatial Transformer Module来在不需要额外监督的情况下，有效地处理几何噪声，提升匹配的鲁棒性。在有挑战性的HPatches上，作者设计了一种基于聚类的方法，来挖掘patch-level的监督，以此提升描述子在图像匹配任务中的表现。

### Optimizing Descriptors for Matching

这一节作者先指出描述子匹配就是最近邻检索，然后讨论了一种learning to rank方法来优化ranking-based检索表现。

#### Nearest Neighbor Matching

![](<../../.gitbook/assets/image (355).png>)

图1展示了由两幅图像计算F矩阵的流程，其中特征匹配过程可以描述为：每幅图中有M个特征，计算两幅图像的pairwise距离矩阵，矩阵共$$M^2$$个元素，对于图1中的每个特征，在图2中搜索其最近邻。相互为最近邻的匹配被当作候选匹配，输入之后的流程（如RANSAC）。&#x20;

作者指出，这一匹配流程就是最近邻检索：图1的每个特征被用于检索数据库，数据库是由图2中特征构成的。为了获得好的表现，正确的匹配应该作为top retrieval输出，错误的匹配则应该尽可能“排名较低”。匹配的表现直接反映了描述子的优劣。为了评估最近邻匹配的表现，作者采用了Average Precision（AP)来作为评价指标。AP是基于二值化相关性假设来评估检索表现的，即检索结果与query相关或不相关。这与特征匹配是相适应的。

#### Optimizing Average Precision

令$$\mathcal{X}$$为图像块空间，而$$S \subset \mathcal{X}$$为database。对于每个query patch $$q \in \mathcal{X}$$，令$$S^+_q$$为它在S匹配的patches，$$S^-_q$$为非匹配的。给定距离度量方式D，令$$(x_1,x_2,...,x_n)$$为$$S^+_q \bigcup S^-_q$$中所有元素依照与q的距离从小到大的排列。给定这一排序，AP是在不同位置测得准确度($$Prec@K$$)的平均值：&#x20;

![](<../../.gitbook/assets/image (379).png>)

AP只有当所有来自$$S^+_q$$的patch都排在$$S^-_q$$之前时，会达到最优值。&#x20;

AP的优化可以当作一个metric learning问题，其目标是学习一个可以让AP在retrieval时达到最优的距离度量D。理想的来说，如果上述过程都可以用可微的方式表示出来，那么AP就可以利用链式法则来优化。然而，排序过程是不可微的，连续变化的输入会引起AP值不连续的跳变。因此，appropriate smoothing对于引出可微的近似AP很重要。

**Binary Descriptor**

二进制描述子在应用中所需内存较少，匹配更快。 在这里，一个神经网络F被用于模拟一种映射，将patches映射到一个低维的hamming空间中，$$F: \mathcal{X} \rightarrow {\{-1,1\}}^b$$，对于hamming距离，它取{0, 1, ..., b}间的整数，AP可以通过直方图$$h^+=(h^+_0,...,h^+_b)$$的元素来闭环计算，其中$$h^+k=\sum{x \in S^+_q} \textbf{1}[D(q,x)=k]$$。闭式（解析）的AP可以进一步被连续化放宽并对于$$h^+$$可微。&#x20;

链式法则的下一步是要让$$h^+$$中的项关于网络F可微，直方图合并操作可以近似为：&#x20;

![](<../../.gitbook/assets/image (47).png>)

利用一个可微函数$$\delta$$（在D(q,x)=k时出现峰值）来代替二进制标志函数。这样，可以得到近似的梯度：

![](<../../.gitbook/assets/image (343).png>)

注意到hamming距离的可微形式如下：&#x20;

![](<../../.gitbook/assets/image (324).png>)

最后，可以用tanh近似进行阈值处理，获得二进制比特：&#x20;

![](<../../.gitbook/assets/image (564).png>)

其中，$$f_i$$为浮点数神经网络激活函数。由以上近似过程，网络可以端到端训练。

**Real-Valued Descriptor**

作者也提供了真值描述子的推导过程。作者将描述子定义为一个真值神经网络响应的向量，并用L2 normalization进行处理。在这种情况下，欧拉距离D可以由此给出：&#x20;

![](<../../.gitbook/assets/image (485).png>)

优化浮点数描述子的AP最大的挑战在于不可微的排序过程，但是浮点数排序没有一个简单的替代形式。但是，直方图合并可以作为一个近似：作者利用直方图合并量化了浮点数距离，获得直方图$$h^+$$，然后将优化问题化为之前的二进制描述子的优化问题。对于L2-normalized向量，量化过程可以简单的根据\[0,2]间欧拉距离来处理：作者将\[0,2]间均匀的划分为b+1个bin。在应用链式法则时，只有公式4和5需要修改。 与二进制描述子的优化不同，这里的b成为一个可调的参数。较大的b可以减少量化的误差，但是计算梯度时其计算复杂度与b呈线性。因此，在这篇论文里，$$b\leq 25$$.

#### Comparison with Other Ranking Approaches

triplets定义了local pairwise ranking losses，而本文的方法属于listwise，因为优化的评估指标AP是在一个ranked list上定义的。 作者指出，triplets是难以训练的，所以需要hardest negative mining，anchor swap， distance-weighted sampling等操作。如图2所示，listwise指标对于位置是敏感的，而local loss是不敏感的；在一个triplet中出现的错误，如果它出现在list的顶部，那么会对整个结果产生很大影响。因此，需要启发式的方法来减少high-rank errors。相反的，本文的方法直接优化了listwise评估指标，AP，所以不需要这些启发式方法。listwise优化已经隐性的包含了hard negative mining：因为它强制匹配的patches排在非匹配的之前，这一过程自动要求了hardest triplet的正确分类，而无需显性的使用它。

![](<../../.gitbook/assets/image (166).png>)

### Task-Specific Improvements

#### Handling Geometric Noise

为了提升特征在匹配时的鲁棒性，保证描述子对几何噪声的不敏感型很重要。作者使用了Spatial Transformer，通过预测6-DOF 仿射变化来对齐patches，无需额外的监督。在本文的实验中，他被用于矫正几何畸变，提升效果。

#### Label Mining for Image Matching

前面的优化过程是根据patch retrieval任务设计的，但是它也适用于更高层的任务。作者在HPatches上测试了算法，来验证算法在image matching任务上的表现。HPatches数据集的image matching任务与patch retrieval任务类似，都是从一些干扰项中检索匹配的图像（块）。但是，在patch retrieval任务中，不会挑选与query位于同一图像序列的图像块作为干扰项，以防止图像具有重复的纹理（干扰项看起来确实与query一样）。在image matching任务中，图像与同一序列的其他图像相匹配，所有干扰项都来自于同一序列。因此，图像匹配的表现可以用将同一序列干扰项加入优化patch retrieval来优化。&#x20;

作者在优化HPatches上的patch retrieval时用label mining来增强干扰项。为了避免重复结构带来的有噪声的标签，作者用了一个简单的启发式方法：聚类。对于每个图像序列，作者根据视觉外观将所有patches聚类，然后，具有较高类内距离的patches被记为彼此的干扰项（要经过3D的验证）。要注意，label mining与hard negative mining无关，它的目的是增加额外的监督。

### Experiments

作者使用了L2Net的结构，网络的输入是32x32的灰度图，当加入Spatial Transformer module时，输入的尺寸被增加到42x42，用3个卷积层来预测一个6-DOF仿射变换，这一变换被用于采样32x32的patch。

![](<../../.gitbook/assets/image (272).png>)

![](<../../.gitbook/assets/image (838).png>)

![](<../../.gitbook/assets/image (482).png>)

image matching的baseline与VL-Banchmarks一致，用Harris-Affine detector检测点，相对与检测的特征帧，用边缘因子为3提取patches。&#x20;

![](<../../.gitbook/assets/image (21) (1).png>)
