---
description: >-
  Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place
  Recognition
---

# \[CVPR 2021]Patch-NetVLAD

{% embed url="https://arxiv.org/abs/2103.01486" %}

{% embed url="https://github.com/QVPR/Patch-NetVLAD" %}

### Abstract

这篇论文中，作者提出Patch-NetVLAD算法，结合了由NetVLAD residuals中的patch-level特征衍生出的局部和全局特征。与现有局部关键特征中固定的空间邻域的机制不同，我们的方法允许在特征空间grid上定义深度学习特征的聚合和匹配。作者进一步提出了一种通过积分特征空间获得互补的尺度(即patch大小)的多尺度patch特征融合方法，并证明融合后的特征具有条件(季节、结构和光照)不变性和视角(平移和旋转)不变性。

### Introduction

![](<../../.gitbook/assets/image (24).png>)

作者结合全局特征和局部特征的优势，并且尽可能规避它们的缺点。为了实现这一点，作者作出如下贡献：

1. 作者提出一种先进的VPR算法，通过对一组图像中locally-global descriptors的穷举匹配，获得一个相似分数。这些descriptors是对于密集采样的局部patches，利用一种VPR-optimized聚合技术（像NetVLAD那种）从特征空间中提取出来的。
2. 作者提出一种多尺度融合技术，来产生和结合不同尺度的混合特征，获得比单一尺度方法更好的表现。为了减少单一尺度到多尺度方法时增长的计算消耗，作者使用了一个积分特征空间（类似于积分图像）来产生不同patch尺寸的局部特征。
3. 结合以上贡献，该算法有很好的应对不同任务需求的灵活性：作者提供了许多易于使用的系统设置，达到不同的表现和计算消耗之间的平衡。

### Methodology

![](<../../.gitbook/assets/image (843).png>)

Patch-NetVLAD计算每对图像间的相似分数，度量了图像间的空间和外观一致性。当给定一张query图像，该层次化系统首先利用original NetVLAD描述子检索出top-k（在实验中设定k=100）最相似的匹配。然后，作者利用一种新的patch descriptor代替NetVLAD中的VLAD层，来对patch-level descriptors进行局部匹配，对最初的匹配列表进行重新排列和调整，得到最后的图像检索。这种结合的方法最小化了patch features间相互匹配带来的额外计算负担，并且不损失最后图像检索阶段的recall表现。

#### Original NetVLAD Architecture

定义NetVLAD的basenet为函数$$f_{\theta}: I\rightarrow \mathbb{R}^{H \times W \times D}$$，给定输入图像I，得到一个$$H \times W \times D$$维的特征图F（比如VGG的conv5层）。传统的NetVLAD结构通过对每个特征$$x_i\in \mathbb{R}^D$$与K个学到的聚类中心求加权和，将这些D维的特征聚合到一个$$K \times D$$维的矩阵。可以表述为，给定$$N \times D$$维特征，VLAD聚合层$$f_{VLAD}：\mathbb{R}^{N\times D}\rightarrow \mathbb{R}^{K\times D}$$可以由下式计算出：&#x20;

![](<../../.gitbook/assets/image (701).png>)

其中$$x_i(j)$$是第i个描述子的第j维元素，$$\overline{a}_k$$是soft-assignment函数，$$c_k$$是第k个聚类中心。在VLAD聚合后，产生的矩阵通过一个映射层$$f_{proj}: \mathbb{R}^{K\times D}\rightarrow \mathbb{R}^{D_{proj}}$$被映射到一个降维的向量。 我们在local patches上应用这种特征图聚合方法来提取描述子($$N \ll H \times W$$)然后对这些不同尺度的patches进行cross-matching，来产生最后用于图像检索的相似分数。这与传统NetVLAD设置$$N=H \times W$$并且聚合图中所有特征不一样。

#### Patch-level Global Features

该系统的核心部分是在完整的特征图中为密集采样的sub-regions(以patches的形式)提取全局描述符。作者从特征图$$F\in \mathbb{R}^{H \times W \times D}$$中以步长$$s_p$$提取一系列$$d_x \times d_y$$大小的patches $${\{P_i, x_i, y_i\}}^{n_p}_{i=1}$$，其中patches的总数量为：&#x20;

![](<../../.gitbook/assets/image (684).png>)

并且$$P_i \in \mathbb{R}^{(d_x \times d_y) \times D}$$和$$x_i, y_i$$是patch特征和特征图中patch的中心。实验证明正方形的square在不同环境中泛化性最好。&#x20;

对于每个patch，作者随后利用NetVLAD对patch上的特征提取一个描述子，构成patch descriptor集合$${\{f_i\}}^{n_p}_{i=1}$$，其中$$f_i=f_{proj}(f_{VLAD}(P_i)) \in \mathbb{R}^{D_{proj}}$$。&#x20;

与传统的基于局部特征的匹配相比，传统局部特征是从比较小的区域中提取出来的，而Patch-NetVLAD是从较大的区域中提取的，具有隐含的语义信息（比如门、窗、树等）。

#### Mutual Nearest Neighbours

当给定一组reference和query特征$${\{f^{r}_{i}\}}^{n_p}_{i=1}$$和$${\{f^{q}_{i}\}}^{n_p}_{i=1}$$(为了方便，假设所有图像都是相同分辨率的)，作者对两张图像的特征描述子进行匹配。得到相互最近邻匹配：&#x20;

![](<../../.gitbook/assets/image (11) (1).png>)

#### Spatial Scoring

作者提供了两种spatial scoring方法来计算query和reference之间的图像相似度，一种是基于RANSAC的方法，需要更多地计算时间，检索表现更好；另一种更快，但是会损失一些检索表现。

**RANSAC Scoring**

sptial scoring基于两幅图像中mutual nearest neighboring的patch特征计算，由符合两幅图像的homography的内点数决定。作者假设每个patch对应一个2D的图像点，即patch的中心点。设置内点的误差阈值为$$s_p$$，然后用patches的数量来标准化一致性分数，这是为了后续结合多尺度上的spatial score。

**Rapid Spatial Scoring**

为了计算rapid spatial scoring，令$$x_d={\{x^{r}_{i}-x^{q}_{j}\}}_{(i,j)\in P}$$为相匹配的patches间patch位置的水平偏差，并且$$y_d$$为垂直偏差。除此之外，令$$\overline{x}d=\frac{1}{|x_d|}\sum_{x_{d,i}\in x_d}x_{d,i}$$和$$\overline{y}_d$$为平均偏差。因此，spatial score可以定义为：&#x20;

![](<../../.gitbook/assets/image (349).png>)

其中，分数包括相对最大空间偏移量的，与均值相对位移的总和。spatial score可以惩罚与平均偏移相比较大的patch位置偏移，有效地度量了在视角变化下场景中各物体的运动一致性。

#### Multiple Patch Size

该相似分数计量方法可以很容易拓展到多尺度上，带来更好的表现。对于$$n_s$$个不同的patch尺寸，作者将不同patch尺寸下的spatial matching scores取凸包集合，得到最终的匹配分数：&#x20;

![](<../../.gitbook/assets/image (847).png>)

其中$$s_{i, spatial}$$是第i个patch尺寸的spatial score。而$$\sum_{i}w_i=1, w_i \le 0$$, for all i.

#### IntegralVLAD

为了协助在多尺度上计算提取patch descriptors，作者提出了integralVLAD的概念，类似于integral images。一个patch的聚合VLAD描述子（在projection之前）可以通过对所有$$1 \times 1$$的patch descriptors求和得到。即计算多尺度patch descriptor存在大量冗余处理，计算较大patch的描述子包含了计算较小patch的描述子。这允许我们去提前计算一个积分的patch feature map，用于计算多尺度融合时的patch descriptors。令integral feature map $$\mathcal{I}$$为：&#x20;

![](<../../.gitbook/assets/image (540).png>)

其中，$$f^{1}_{i'.j'}$$代表特征空间中(i',j')处patch size为1的VLAD聚合patch descriptor（在projection之前）。现在我们可以通过一些在积分特征图上的四则运算，来恢复任意尺度的patch descriptors。在实践中，这是通过2D depth-wise dilated卷积实现的，卷积K为：&#x20;

![](<../../.gitbook/assets/image (208).png>)

而dilation设置为所需的patch size。

### Experimental Results

#### Inplementation

作者在RobotCar Seasons v2训练集上挑选了参数，使用单尺度时，令$$d_x=d_y=5$$，stride $$s_p=1$$；使用多尺度融合时，patch size为2,5,8，融合时的权重分别为0.45,0.15,0.4

#### Results

判断依据遵循场景识别的一般规则，就是Recall@N，判断top-N张图像中是否有一张是正确的。在Nordlan数据集上真值的前后10帧图像被认为是正确的，Pittsburgh和Tokyo24/7允许平移误差为25m，Mapillary允许平移误差为25m且朝向误差为40°。而在RobotCar Seasons v2和Extended CMU Seasons数据集上，使用默认的误差阈值，平移误差0.25、0.5、5m，对应的朝向误差为2、5、10°。该算法没有进行query图像的位姿估计，所以所给出的pose是由最相似的reference图像的位姿给出的。&#x20;

定量对比实验：&#x20;

![](<../../.gitbook/assets/image (153).png>)

![](<../../.gitbook/assets/image (1046).png>)

消融实验（单一尺度v.s.多尺度，RANSAC v.s. spatial scoring）&#x20;

![](<../../.gitbook/assets/image (1061).png>)

计算消耗时间：&#x20;

![](<../../.gitbook/assets/image (17).png>)
