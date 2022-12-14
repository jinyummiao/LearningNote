---
description: 'AdaLAM: Revisiting Handcrafted Outlier Detection'
---

# \[ECCV 2020] AdaLAM

{% embed url="https://arxiv.org/abs/2006.04250" %}

{% embed url="https://github.com/cavalli1234/AdaLAM" %}

### Abstract

这篇论文中，作者设计了一个handcrafted hierarchical pipeline来有效地检测匹配中的外点。AdaLAM可以有效地利用现代并行硬件，从而非常快并且非常准确地滤除外点。

### Introduction

在这篇文章中，作者回顾了人工设计的的外点滤除方法。根据这一研究领域的最佳实践，作者提出了一种基于样本自适应阈值的局部仿射运动验证的外点滤除方法，命名为Adaptive Locally-Affine Matching (AdaLAM)。此外，AdaLAM可以有效地利用现代并行硬件，在一块现代GPU上，每对图像只需不到20毫秒的时间在8000个关键点中过滤匹配。本文的贡献如下：

1. 提出了AdaLAM，将过去很多在空间匹配上的方法统一、鲁棒、高并行化地结合在一起，来实现快速的外点滤除；
2. 由于该模型基于几何假设，在不同的场景下具有不同的分辨能力，我们提出了一种新的方法，自适应地弱化我们的假设，以实现更好的泛化到不同的领域，同时仍然从每个图像领域中挖掘尽可能多的信息。
3. 实验表明，自适应方法改进了泛化能力，AdaLAM优于当前最先进的方法。

### Hierarchical Adaptive Affine Verification

给定图$$I_1$$和$$I_2$$中关键点$$K_1$$和$$K_2$$，M是从$$K_1$$到$$K_2$$的最近邻匹配。在实际中，由于描述子的限制，M中有大量不正确的匹配，因此，本文的目标即是找到一个子集$$M'\subset M$$，这个子集是全部正确内点的几何$$M^*\subset M$$的近似。&#x20;

![](<../../.gitbook/assets/image (1).png>)

该算法的主要流程为：

1. 选择有限个可信的、分布较好的匹配，作为seed points；
2. 对于每个seed point，选择邻近区域内可以相容的correspondences；
3. 通过高度并行的RANSACs方法来验证每个seed points的邻域内的局部affine consistency，该RANSAC方法使用sample-adaptive inlier threshold。输出seed points中有足够强支持（每个seed point都有自己的inlier threshold）的内点作为匹配M'。

#### Preliminaries and core assumptions

从两个视角看3D空间中的平面具有单应性，这一性质可以用图像空间中的affine transformation A来局部近似。这种仿射变换对于正确的关键点correspondence有很强的空间一致性约束，可以作为一个非常可靠的过滤器。但是，这种关于平面的、局部的和正确的映射的假设在真实图像中往往无法满足：

1. 3D点所处的表面可能不是平面。某一点处的三维切平面与实际曲面之间的偏移会引起不位于切平面上的三维点的投影的非线性误差，这种偏差随着曲面曲率的增大而越来越显著。
2. 检测出的点可能彼此不靠近，这会对仿射模型产生畸变，使得其无法再很好地近似隐含单应性。这种误差随着关键点的相对距离和切平面的斜率的增大而增大。
3. 匹配的关键点可能不是完全相同的3D点的投影点，这是宽基线视觉变化中的常见问题，因为光照的微小变化和遮挡可能会在关键点定位时移动显著性热力图的峰值。
4. 不正确的非线性镜头畸变模型会进一步在两个视角中引入关键点运动的非线性。

为了解决以上问题，作者针对核心假设提出了adaptive relaxation。

#### Seed points selection

因为仿射变换A是3D点P周围局部变化的近似，所以作者采用了最近邻匹配来指导搜索可能的3D平面上的点。更具体的说，作者希望找到可靠的、分布良好的correspondences的受限集合作为P的假设，在其周围可以搜索一致的点匹配关系。作者将这些假设称为seed points。为了选择seed points，作者对每个关键点分配一个可信度分数，并将半径R内最大分数的关键点提升为seed point。这可以通过局部NMS来高效的实现。对于每个关键点，作者用最近邻匹配的ratio test分数作为可信度分数。通过这种方式，作者保证了seed points的distinctiveness和coverage，并且避免了网格化，同时让挑选过程完全并行化，可以在GPU上高效计算，因为在seed point挑选中每个匹配可以单独保存和与最近邻比较，和其他选择相互独立。

#### Local neighborhood selection and filtering

对seed points分配correspondences是算法中很重要的一步，因为它在P的每个假设周围构建了搜索空间来寻找仿射变换A。更大邻域可以更容易包含符合A的正确匹配，同时它们隐式地放宽了仿射约束，因为它们违背了局部性假设。&#x20;

令$$S_i=(x_1^{S_i},x_2^{S_i})$$为一个seed point匹配，从它的局部特征引出了一个相似变换（旋转+缩放）$$(\alpha^{S_i}=\alpha_2^{S_i}-\alpha_1^{S_i},\sigma^{S_i}=\sigma_2^{S_i}/\sigma_1^{S_i})$$，其中旋转为$$\alpha^{S_i}$$，缩放为$$\sigma^{S_i}$$。$$N_i\subseteq M$$是分配给$$S_i$$的匹配，其符合了仿射一致性。令$$t_\alpha$$和$$t_\sigma$$是候选匹配与seed匹配$$S_i$$之间旋转和缩放的agreement阈值。

当符合以下条件时，引出变换($$\alpha^{p}=\alpha_2-\alpha_1,\sigma^p=\sigma_2/\sigma_1$$)的匹配$$(p_1,p_2)=((x_1,d_1,\sigma_1,\alpha_1),(x_2,d_2,\sigma_2,\alpha_2))\in M$$加入$$N_i$$:&#x20;

![](<../../.gitbook/assets/image (1053).png>)

其中，$$R_1,R_2$$是图1和图2中seed point散布的半径，$$\lambda$$是用于调整邻域重叠程度的超参数。注意角度$$\alpha$$是在$$(-\pi,\pi]$$之间的。不同的半径$$R_1,R_2$$是根据图像面积来等比选择的，不受图像缩放影响。&#x20;

简单来说，就是在seed point邻域（公式1）内符合一定旋转角度和缩放尺度要求（公式2）的点被加入匹配集合中，用于估计局部的仿射变换。在这一步中，每个集合$$N_i$$可以独立处理进行仿射验证。

#### Adaptive Affine Verification

在这一节，作者描述如何从集合$$N_i$$内挑选内点。作者用经典的RANSAC框架来处理这一方法，用最小样本迭代固定次数。由于中心仿射变换的最小样本可以只包含两个匹配，所以在实践中无需迭代很多次。采样的策略是受PROSAC启发。作者用ratio-test分数来让RANSAC采样的分布偏向于优先选取更可能为内点的样本。但是当保证最优模型存在时，PROSAC旨在寻找可以解释数据的这个最优模型，而在作者的设置中，作者不希望在集合$$N_i$$上花费过多计算能力，因为它们本身可能是外点。所以，对于所有集合$$N_i$$作者只迭代固定次数，与PROSAC只要不触发提前终止就逐渐退化到均匀采样的采样策略不同。除此之外，作者确切的选择要处理的样本，以便在固定次数迭代预算允许的范围内尽可能地探索最小样本（原文：Moreover, we deterministically select the samples to be drawn so that we exhaustively explore the most likely minimal samples as much as the fixed iterations budget allows.）。&#x20;

在第$j$次迭代中，作者在每个集合$$N_i$$中采集一个最小集合，并拟合仿射变换$$A^j_i$$。用一个对应于点$$x^k_i,x^k_2$$的匹配k的残差$$r_k$$来描述它与$$A^j_i$$的偏差：&#x20;

![](<../../.gitbook/assets/image (162).png>)

如\[Preliminaries and core assumptions]节所期望的，仿射模型部分地解释了我们想要识别出的图像移动，并且仿射模型的误差没有清晰的界限。因此，我们无法设置一个固定的阈值来通过$$r_k$$判断k是否是一个内点。相比均匀分散的外点这一零假设，作者在内点集合的统计学显著性上设置阈值，而非直接对误差分数设置阈值。（原文：Instead of thresholding directly on the error score, we threshold on the statistical significance of an inlier set against the null hypothesis of uniformly scattered outliers.）这一映射更好的解释了作者的目标：作者不对仿射模型相对真实移动模型的偏差设限，而是对观察的似然设限。&#x20;

特别是，当$$H_o$$是具有均匀分散外点匹配的假设，作者将残差集合$$\mathcal{R}$$中的残差值$$r_k$$映射到一个可信度分数$$c_k$$:&#x20;

![](<../../.gitbook/assets/image (56).png>)

其中，正样本数$$P=|l:r_l\le r_k|$$为假设匹配k是最差内点时的内点数量。Notice that sorting R by residual yields P = k + 1 (assuming 0-indexing).所提出的可信度c有效地度量了实际找到的内点数和在只有外点的假设$$H_o$$下找到的内点数的比例。它可以被解释为一个ratio-test，比较外点假设和采样得到的事实。$$\mathbb{E}_{H_o}[P]=|\mathcal{R}|\frac{t^2}{R^2_2}$$是基于假设：在第二幅图像中，外点均匀分布在整个采样半径$R\_2$上。在均匀性假设失败的情况下，所提出的度量会随着外点局部密度的增加而呈线性偏离。在实践中，这种情况经常发生，因为匹配集中在高度纹理区域，只覆盖采样半径中的一小部分。然而，对c的值使用保守的高阈值$$t_c$$，数据中的较为显著的模式仍然可以以非常高的置信度出现。$$c_k>t_c$$的采样k被记为内点。这个方法是基于显著性的自适应阈值方法，可信度是可以用并行硬件进行高速计算的。&#x20;

在每一次并行迭代中作者计算所有样本的可信度，并在可信度上选择一个固定的阈值$$t_c$$。通过内点集合估计仿射模型，再通过仿射模型挑选内点。最后，对于每个集合$$N_i$$，我们只输出内点数最高的那次迭代的内点，并过滤出最佳迭代时内点数少于$$t_n$$的i。

#### Inplementation details

在实验中，作者设置seed point提取中的R，让其为NMS区域的面积$$R^2\pi$$和图像面积wh之间的比值。特别地，设置$$R=\sqrt{\frac{wh}{\pi r_a}},r_a=100$$。为了对每个seed point $i$选取近邻$$N_i$$，作者采用了一个比R大$$\lambda$$倍的半径，$$\lambda=4$$。这确保了相邻区域之间有足够的但受控的重叠，对seed匹配中的误差具有鲁棒性。作者发现当增加RANSAC的迭代次数时，算法很快饱和，因此固定为128次迭代。当与SIFT相结合进行实验时，$$t_\sigma=1.5,t_\alpha=30°$$。最后，作者观察到，在$$t_c$$非常保守的情况下，可以获得最佳性能，因此，作者设置$$t_c=200$$并要求近邻集合中至少有6个内点。

### Experiments

用OpenCV SIFT作为特征，每张图最多提8k个特征点。通过得到的匹配计算Essential矩阵，并将E矩阵分解为旋转和平移，以角度来度量旋转和平移的误差，取两者中较大者，计算阈值为5°，10°，20度时的AUC值。  &#x20;

![](<../../.gitbook/assets/image (142).png>)

![](<../../.gitbook/assets/image (679).png>)

![](<../../.gitbook/assets/image (22).png>)

![](<../../.gitbook/assets/image (870).png>)

