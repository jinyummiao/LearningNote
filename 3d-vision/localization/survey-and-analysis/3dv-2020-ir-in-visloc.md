---
description: Benchmarking Image Retrieval for Visual Localization
---

# \[3DV 2020] IR in VisLoc

{% embed url="https://ieeexplore.ieee.org/abstract/document/9320415" %}

{% embed url="https://github.com/naver/kapture-localization" %}

### Abstract

定位算法一般依靠图像检索算法来实现两个任务：1）给定一个大约的位姿估计；2）判断场景中的那一部分在图像中是可见的。图像检索任务一般是通过检索大视角变化的显著地标来训练的。但是，在视觉定位任务中，对视角的鲁棒性不是必需的。本文旨在探索不同视觉定位任务中图像检索的作用。作者引入一个benchmark设置，在不同数据集上进行检索算法的比较。研究表明，经典地标检索/识别任务的检索性能只与定位性能部分相关。这说明需要专门为了定位任务设计相应的检索算法。

### Introduction

Visual localization is the problem of estimating the exact camera pose for a given image in a known scene, i.e., the exact position and orientation from which the image was taken.

![](<../../../.gitbook/assets/image (875).png>)

图像检索在视觉定位中有很多不同的作用。Efficient pose approximation by representing the pose of a query image by a (linear) combination of the poses of retrieved database images (Task 1). Accurate pose estimation without a global 3D map by computing the pose of the query image relative to the known poses of retrieved database images (Task 2a). Accurate pose estimation with a global 3D map by estimating 2D-3D matches between features in a query image and the 3D points visible in the retrieved images (Task 2b).

这三个任务对检索有不同的要求：任务1要求检索阶段找到与query图像位姿尽可能相近的图像，即图像表征不应当对视角变化具备鲁棒性和不变形；任务2a和2b要求检索阶段找到与query图像描述相同场景部分的图像，但是检索到的图像不需要拍摄于相近的位姿，只要局部特征匹配能够成功。实际上，任务2a需要检索到与query图像位姿不同的多张图像，任务2b检索到的图像最好与query图像有较大的重叠视野，理论上只需要一张相关的reference图像。

作者通过实验证明了：(1) there is no correlation between landmark retrieval/ place recognition performance and Task 1. (2) Similarly, retrieval/recognition performance is not a good indicator for performance on Task 2a. (3) Task 2b correlates with the classical place recognition task. (4) Our results clearly show that there is a need to design image retrieval approaches specifically tailored to the requirements of some of the localization tasks.

### The proposed benchmark

本文希望探索地标检索/场景识别和视觉定位表现之间的关联。具体而言，希望调研当前最好的检索/识别表征是否已经足够好，是否需要设计任务相关的特定表征。

本文通过统一定位pipeline的其他部分，只改变三个定位任务中不同的检索方法，来公平对比检索方法的表现。这一评估框架包含两个部分，其一评测了三个定位任务的表现，其二评估了地表检索/场景识别的表现。通过在这些任务上的表现对比，可知定位任务中图像检索与地标检索/场景识别之间的关系。

#### Visual localization tasks

图像检索在视觉定位任务中一般有两个作用。其一为识别与query图像具有相近位姿的reference图像（任务1）；其二为检索到与query图像描述相同部分场景的图像，不需要位姿相近（任务2a和2b）。

**Task 1: Pose approximation** 该类方法受到场景识别的启发，根据top k张检索图像的位姿来有效地估计query图像的位姿。

定义相机位姿为元组$$P=(c,q)$$，其中$$c \in R^3$$为相机在全局坐标系下的位置，$$q \in R^4$$为相机的姿态（用四元数表示）。query图像的位姿$$P_q$$用reference图像位姿$$P_i$$的加权和来计算$$P_q=\sum^k_{i=1}w_iP_i$$，$$P_i$$需要归一化为四元数。

作者考虑了三种情况：**equal weighted barycenter**令所有权重相等，即$$w_i=1/k$$**（EWB）**；**barycentric descriptor interpolation（BDI）**：

![](<../../../.gitbook/assets/image (361).png>)

第三种方法中，$$w_i$$基于L2正则化后的描述子之间的**cosine similarity（CSI）**

![](<../../../.gitbook/assets/image (566).png>)

当$$\alpha=0$$时，该方法等同于EWB。在本文中，作者令$$\alpha=8.$$

**Task 2a: Pose estimation without a global map** 理论上，如果相对位姿估计准确（包括平移的尺度），检测到一张图像就足够。实际上，检索多于一张图像会提升准确率，因为考虑了多个相对位姿。一旦估计出query图像和database图像间的相对位姿，可以用三角化去计算绝对位姿。However, pose triangulation fails if the query pose is colinear with the poses of the database images, which is often the case in autonomous driving scenarios.（没搞懂如何计算的...）

因此，作者用带有位姿的检索到的database图像在线构建了场景的3D地图，然后利用PNP来注册query图像。Similar to pose triangulation, this local SFM approach fails if (i) less than two images among the top k database images depict the same place as the query image, (ii) the viewpoint change among the retrieved images and/or between the query and the retrieved images is too large (to be handled by local feature matching) or (iii) the baseline between the retrieved database images is not large enough to allow stable triangulation of enough 3D points. 因此，该方法需要检索出一组与query图像描述同一场景、但是视角不同的database图像。所以，Task 2a需要鲁棒但是不具备视角不变性的图像表征。

**Task 2b: Pose estimation with a global map** 与任务2a不同，该任务先构建了场景的全局3D模型。场景的SFM模型能够提供database图像中局部特征和地图中3D点之间的对应关系。构建query图像和database图像间2D-2D匹配会产生2D-3D匹配，可以用于利用PNP和RANSAC估计位姿。

理论上，只要局部特征可以获得query和检索到的database图像间的匹配，只要检索到一张database图像就足够了。检索到更多的相关图像能够增加准确位姿估计的机会。For efficiency, k should be as small as possible as local feature matching is often the bottleneck in terms of processing time.

**Visual localization metric** We follow common practices to measure localization performance by computing the position and rotation errors between an estimated and a reference pose. For evaluation, we use the percentage of query images localized within a given pair of (position, rotation) error thresholds $$(X m, Y\degree)$$.

#### Landmark retrieval and place recognition tasks

**Landmark retrieval** 该任务旨在找到与query图像具备相同主要目标（地标）的图像。因此，图像表征应该具备视觉和视觉条件不变性，来识别出所有相关图像。

为了评估检索图像与query图像是否相关，作者采用了基于3D模型的定义：两图的相似度用其可见的3D点间的IoU来度量。为了计算query图像的可见3D点，作者用R2D2特征和COLMAP算法对数据库图像和query图像构建了SFM模型。

**Landmark retrieval metric** The classical mean Average Precision (mAP) metric, most commonly used in the literature to measure landmark retrieval performance, reports a single number integrating over different numbers of retrieved images. We use the related _mean Precision@k (P@k)_ measure to determine the link between number of retrieved images and localization performance.

**Place recognition** 该任务旨在估计query图像拍摄的大致地点。地点使用被检索到的图像来定义的，因此需要在top-k个被检索图像中至少有一个相关的reference图像。如果database图像拍摄于query图像的附近，则认为该图像是相关于query图像的。当相机姿态已知，则相机姿态间的夹角也应当被考虑。所以，上述IoU相似度也可以用于图像是否描述了同一地点。

&#x20;**Place recognition metric** We follow the standard protocol and measure place recognition performance via Recall@k (R@k). R@k measures the percentage of query images with at least one relevant database image amongst the top k retrieved ones.

### Experimental evaluation

**Experimental setup** We use DenseVLAD and three popular deep image representations, NetVLAD, APGeM, and DELG, for image retrieval. DenseVLAD pools densely extracted SIFT descriptors through the VLAD representation, resulting in a compact image-level descriptor. We use two variants: DenseVLAD extracts descriptors at multiple scales, while DenseVLAD-mono uses only a single scale. NetVLAD uses CNN features instead of SIFT features and was trained on the Pitts30k dataset. Both DenseVLAD and NetVLAD have been used for visual localization and place recognition before. AP-GeM and DELG represent state-of-the-art representations for landmark retrieval, while AP-GeM was recently used for visual localization as well. Both models were trained on the Google Landmarks dataset (GLD), where each training image has a class label based on the landmark visible in the image. Relevance between images is established based on these labels. Hence, two images can be relevant to each other without showing the same part of a landmark. We use the best pre-trained models released by the authors for all experiments.

For Tasks 2a and 2b, i.e., pose estimation without and with a global map, we use R2D2 to extract local image features and COLMAP for SFM.

For all three datasets we use three threshold pairs for evaluating localization for low $$(5 m, 10\degree)$$, medium $$(0.5 m, 5\degree)$$, and high $$(0.25 m, 2\degree)$$ accuracy.

#### Landmark retrieval and place recognition

![](<../../../.gitbook/assets/image (192).png>)

As expected, the learned descriptors (NetVLAD, AP-GeM, and DELG) typically outperform the SIFT-based DenseVLAD. There are two interesting observations: (1) NetVLAD outperforms both AP-GeM and DELG under theR@k measure for small k on the day-time RobotCar queries. This can be attributed to the fact that NetVLAD was trained on street-view images captured at daytime from a vehicle while AP-GeM and DELG were trained with a large variety of landmark images taken from very different viewpoints. (2) On R@k for the RobotCar night-time queries, DELG performs significantly worse than the others. We attribute this to the low-quality nighttime images, which often exhibit strong motion blur and color artifacts which are not reflected in the training set of DELG. AP-GeM, trained on the same data, avoids this problem through adequate data augmentation.

DELG和AP-GeM描述子是场景检索或识别任务中最好的方案。



![](<../../../.gitbook/assets/image (816).png>)

在Oxford和Paris数据集上，DELG和AP-GeM描述子表现比DenseVLAD和NetVLAD好很多。但是如果目标是场景识别，即需要找到至少一张相关图像，这种差距会小很多，而且有时排名会改变。

#### Task 1: Pose approximation

![](<../../../.gitbook/assets/image (51).png>)

We show the percentage of query images localized within a given error threshold w.r.t. the ground truth poses as a function of the number k of retrieved images used for pose approximation. We only report results for the low $$(5 m, 10\degree)$$ thresholds.

EWB对top-k张检索图像赋予相同的权重。相反，BDI和CSI给排名靠前的图像赋予更高权重。因此，它们假设描述子和位姿的相似度之间有关联。如图3，只有在Aachen数据集上，检索多于一张图像会提升表现，这是因为query图像和reference图像之间的位姿差异较大。因此，检索图像的位姿插值会得到更好的估计位姿。CSI表现最好，因为它给不相关图像的权重很低。

根据图2和图3，可以发现定位表现和P@K相关，但是与R@K不相关，只有在Aachen上是例外。在Aachen上，DenseVLAD在地标检索/场景识别任务上表现较差，但是在位姿估计上表现很好。其原因可能是：DenseVLAD缺乏视角不变性，这有助于位姿估计，因为位姿更相近的图像有更相似的描述子。但是，DenseVLAD在光照变化方面的鲁棒性比AP-GeM和DELG要差，这解释了为什么后者在RobotCar和Aachen的night-time序列上表现更好。

AP-GeM和DELG在Aachen上检索和识别表现差不多，但是位姿估计任务上DELG更好些。This suggests that DELG retrieves reference images that are better spread through the scene, which is beneficial for pose interpolation。

pose approximation和图像检索相关，与place recognition不相关。These suggest that learning scene representations tailored to pose approximation, instead of using off-the-shelf methods trained for landmark retrieval, is an interesting direction for future work.

The best results for RobotCar and Baidu are obtained for k = 1, in which case all three methods perform the same. For RobotCar, this result is surprising as interpolating between two consecutive images should give finer approximation. This indicates that there is potential for improvement by designing image retrieval representations specifically suited for pose interpolation.

#### Task 2：Accurate pose estimation

![](<../../../.gitbook/assets/image (376).png>)

**Task 2a** 所有描述子在室外day-time和室内图像中都表现很好，表现接近，DenseVLAD-Mono是个例外。在day-night情况下，学习的描述子比DenseVLAD表现更好，AP-Gem表现最好。

根据图2和图4，检索/识别任务与任务2a没有明显关联。所有方法在Aachen和RobotCar day-time序列上都表现良好。AP-GeM在RobotCar night-time序列上具备更好的检索/识别表现，相应地也获得了更好的任务2a表现。DELG和AP-GeM在Aachen night-time序列上也具备更好的P@K表现，但是任务2a的表现没有更好。In fact, there is no correlation between P@k, which decreases with increasing k (c.f . Fig. 2), and Task 2a performance, which remains the same or increases with increasing k.

To achieve good pose accuracy using a local SFM model for a given set of retrieved images, a high R@k score is not sufficient. This is due to the fact that more than one relevant image is needed to build the local map. At the same time, not all of the top k retrieved images need to be relevant, i.e., a high P@k score is not needed.&#x20;

Overall, retrieval/recognition performance is not a good indicator for Task 2a performance. This indicates that better retrieval representations can be learned that are tailored to the task of pose estimation without a global map, e.g., by designing a loss that optimizes pose accuracy.

![](<../../../.gitbook/assets/image (856).png>)

**Task 2b** 所有方法在day-time和室内场景中表现都不错，学习的方法在Aachen night-time序列上更好。

在RobotCar night-time和Baidu序列上表现较差的原因可以用较低的R@K来解释，尤其是当K<10时。如果在top-k个检索图像中没有至少一张相关图像，任务2b会失败。

在粗略的位姿阈值时，R@K和定位表现有明确的关联。较高的R@K会提升图像被成功定位的几率。这证明了一般会在定位任务中使用当前表现最好的检索/识别算法。但是，较高的R@K并不一定会带来高的位姿准确性。One explanation is that the retrieved images are relevant but share little visual overlap with the query image. In this case, all matches will be found in a small area of the query image resulting in an unstable (and thus likely inaccurate) pose estimate.

### Conclusion

Retrieval techniques are often used to efficiently approximate the pose of the query image or as an intermediate step towards obtaining a more accurate pose estimate.

Our results show that state-of-the-art image-level descriptors for place recognition are a good choice when localizing an image against a pre-built map as performance on both tasks is correlated. We can see that on the night images as well as on Baidu, AP-GeM often outperforms the other features. One of the reason might be that AP-GeM is the only feature that was trained not only with geometric data augmentation but also with color jittering. This might explain why it better handles day-night variations.

We also show that the tasks of pose approximation and localization without a pre-built map (local SFM) are not directly correlated with landmark retrieval/place recognition. In the case of pose approximation, representations that reflect pose similarity in their descriptor similarities, i.e., exhibit robustness only to illumination changes, are preferable as they tend to retrieve closer images. For local SFM, there is a complex relationship between the retrieved images that is not captured by the classical Precision@k and Recall@k measures used for retrieval and recognition.&#x20;

Our results suggest that developing suitable representations tailored to these tasks are interesting directions for future work. Our code and evaluation protocols are publicly available to support such research.
