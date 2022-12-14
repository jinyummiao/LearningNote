---
description: >-
  LiPo-LCD: Combining Lines and Points for Appearance-based Loop Closure
  Detection
---

# \[BMVC 2020] LiPo-LCD

{% embed url="https://emiliofidalgo.github.io/publication/company-2020-lipo" %}

## Introduction

in this paper, we introduce Lines and Points Loop Closure Detection (LiPo-LCD), a novel appearance-based loop closure detection approach which combines points and lines. For a start, both features are described using binary descriptors. Next, an incremental BoW scheme is used for feature indexing. Lines and points are maintained separately into two incremental visual vocabularies and employed in parallel to obtain loop closure candidates efficiently. To combine the information provided by the two vocabularies, we propose a late fusion method based on a ranked voting system. Finally, to discard false positives, we improve the typical spatial verification step integrating lines into the procedure through: (1) a line matching strategy which includes structural information to achieve a higher number of matching candidates; and (2) a line representation tolerant to occlusions, which is combined with points into an epipolarity analysis step.

## Overview of the Loop-Closure Detection Approach

![](<../../../.gitbook/assets/image (235).png>)

### Image Description

本文使用点特征和线特征描述图像，对于$$I_t$$，提取特征$$\phi(I_t)=\{P_t,L_t\}$$,其中$$P_t$$为局部关键点描述子，$$L_t$$为线描述子。这两种特征可以互补地描述图像，增强回环检测的鲁棒性。

#### Point Description

作者使用ORB特征从图像中提取m个点特征。

#### Line Description

作者用LSD检测线，用二进制LBD描述线，从图像中提取n个256维线描述子。

In the original implementation, a rectangular region centred on each line is considered. Such region is divided into a set of bands $$B_i$$, from which a descriptor $${BD}_i$$ is computed contrasting Bi with its neighbouring bands. On the other hand, the binary descriptor is finally obtained considering 32 possible pairs of band descriptors $${BD}_i$$ within the support region. Each pair is compared bit by bit, generating an 8-bit string per pair. A final 256-bit descriptor is generated concatenating the resulting strings for all pairs.

### Searching for Loop Closure Candidates

作者根据OBIndex2来检索回环候选帧。在检索时，算法维护了两个OBIndex2的实例，分别对应于点特征和线特征。给定某图像，分别并行地根据点特征和线特征来检索相似的图像，根据点得到m个最相似的图像<img src="../../../.gitbook/assets/image (226).png" alt="" data-size="original">，根据线找到n个最相似的图像![](<../../../.gitbook/assets/image (292).png>)。两个列表都依据其相似度分数进行排序。由于分数的范围会随着词典中视觉单词的分布而变化，所以利用min-max normalization将分数映射到\[0,1]间：

![](<../../../.gitbook/assets/image (253).png>)

相似度分数小于阈值的图像被剔除掉。此外，当前图像的特征被用于更新词典。

### Merging Lists of Candidates

作者基于Borda count方法来进行排序投票，将两种特征检索到的结果融合起来。在LiPo-LCD中，两种特征作为两个投票人，候选人数量c被定义为前文检索到的候选帧列表的最小数量，然后每个列表中top-c张图像被分配了分数：

![](<../../../.gitbook/assets/image (225).png>)

对于在两个列表中都出现的图像，其分数为：

![](<../../../.gitbook/assets/image (278).png>)

然后，根据此分数将两列表整合在一起，并进行排序。得到的列表$$C^t_{pl}$$融合了两种特征信息，并且与特征数量无关。最后为了解决场景中主要存在一种特征的情况，只在一个列表中出现的图像也被加入最终的列表$$C^t_{pl}$$。

### Dynamic Islands Computation

作者考虑了时间一致性，采用iBoW-LCD中的dynamic islands理论来将图像先聚类成islands。将$$C^t_{pl}$$中的图像关联到其island上，然后计算island的分数：

![](<../../../.gitbook/assets/image (242).png>)

然后将islands根据其分数排序，对于最相似的island，判断是否前一帧图像的最相似island与当前图像的最相似island在时间上有重叠。一旦找到最相似的island，该island中具有最高Borda分数的图像被作为候选帧，进行下一步几何验证。

### Spatial Verification

对于ORB点特征匹配，作者使用汉明距离，并使用最近邻ratio test。

#### Line Feature Matching

对于当前图像中的每个线描述子$$l^t_i$$，从候选帧中检索最相似的线描述子，构成一个有序列表。为了解决相机旋转问题，计算了两针间的旋转$$\theta_g$$，然后计算每对线之间的相对旋转$$\alpha^j_i$$

![](<../../../.gitbook/assets/image (291).png>)

其中$$\theta^t_i$$为线在图像中的朝向。对于有序列表，剔除具有较大$$\alpha^j_i$$的线匹配。为了获得最后的线匹配，从每个有序列表中挑选最相似的线特征，并进行最近邻ratio test。

#### Epipolar Geometry Analysis Combining Points and Lines

In this work, line segments are represented by their endpoints. On the other hand, endpoints are first matched between matching lines and next regarded as additional point correspondences for F computation. To associate segment endpoints (taking into account that a starting point of a line might correspond to the end point of the line in the other image), we select that pair that minimizes the rotation between lines using lines orientation and the global rotation $$\theta_g$$, We consider a candidate line matching as an inlier if at least one endpoint pair supports the geometric model。

## Experimental Results

准确率-召回率曲线：

![](<../../../.gitbook/assets/image (218).png>)

线匹配准确率：

![](<../../../.gitbook/assets/image (283).png>)

实时性表现：

![](<../../../.gitbook/assets/image (250).png>)

和SOTA算法对比：

![](<../../../.gitbook/assets/image (267).png>)
