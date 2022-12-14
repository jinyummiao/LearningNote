# \[RAL 2020] CoHOG

{% embed url="https://ieeexplore.ieee.org/document/8972582" %}

{% embed url="https://github.com/MubarizZaffar/CoHOG_Results_RAL2019" %}

### Abstract

这篇论文，作者利用图像熵来提取ROI，利用regional-convolution进行特征匹配，在低内存需求的情况下获得很好的VPR表现，不需要训练过程。

### Introduction

作者提出了一种基于人工设计特征的VPR算法，无需训练阶段。算法提取特征的时间很少，所需内存也很小。本篇论文受到以下启发：1）CNN能够扫描整张图像来获得特定特征，不管这个特征在图像中的哪个位置，相同的CNN滤波器将被激活；2）CNN可以提取图像中具有信息和有区分力的ROI；3）在条件多变的VPR数据集上训练过的CNN可以学校到对季节和光照变化不敏感的图像表征。 受到CNN的启发，作者首先计算图像的熵图，然后据此提取信息丰富的区域。每个区域由HOG描述子局部描述。然后，作者用convolutional matching匹配HOG描述子，来获得视角不变性。这种regional-convolutional匹配是基于标准的矩阵乘法，因此计算很高效。为了获得光照不变性，作者对HOG描述子使用了block normalization。整个图像检索流程如图2所示：&#x20;

![](<../../.gitbook/assets/image (20).png>)

### Methodology

![](<../../.gitbook/assets/image (557).png>)

所提出方法可以分为7个主要模块，如图3所示。输入图像为RGB图像，被转换为灰度图，缩放为$$H_1\times W_1$$。机器人地图包括参考图像的HOG描述子。在这篇论文中，作者使用原始版本的HOG特征，但是作者是在局部区域内计算HOG而非全局区域。

#### ROI Extraction

![](<../../.gitbook/assets/image (165).png>)

首先作者计算图像的熵图。熵图的大小与原图一样，为$$H_1\times W_1$$。定义图像中的一个区域为一个$$W_2 \times H_2$$的图像patch，则一张大小为$$H_1\times W_1$$的图像包括N个大小为$$W_2 \times H_2$$的patch，其中$$N=(H_1/H_2) \times (W_1 / W_2)$$，质量由一个矩阵来表示：&#x20;

![](<../../.gitbook/assets/image (1033).png>)

为了评估R中每个区域的质量$$r_{xy}$$，熵图用一个矩阵E来表示。熵矩阵大小为$$H_1\times W_1$$，元素的值在0-1之间：&#x20;

![](<../../.gitbook/assets/image (492).png>)

每个区域的质量$$r_{xy}$$通过对大小为$$(W_2 \times 2) \times(H_2 \times 2)$$的平均熵值阈值化来计算，即每个块包含四个大小为$$W_2 \times H_2$$的区域，其中这四个区域有一个共同的角点。这样的block-level评估提供了HOG描述子计算时的一致性。block-level质量评估是以$$stride=W_2=H_2$$为步长的，因此评估的区域块的数量为$$M=(n-1)\times(m-1)$$。所有大于质量阈值GT的G个区域都被挑选出来进行匹配。这样用阈值挑选比传统的挑选top-G个区域更能适应低纹理区域，并且计算效率更高。&#x20;

![](<../../.gitbook/assets/image (1030).png>)

**感觉原文写得很混乱……个人理解是先按照Algorithm 1中的算法求出每个pixel的entropy，即矩阵E，然后通过对2x2像素的block（一个region）的平均entropy进行阈值化，阈值为GT，得到二值化矩阵R，挑选region**

![](<../../.gitbook/assets/image (515).png>)

#### HOG-Descriptor Computation

端到端的HOG描述子计算流程为：

1. 计算大小为$$H_1\times W_1$$的灰度图的梯度图；
2. 对图像中N个大小为$$W_2 \times H_2$$的区域计算HOG。每个区域的直方图都有L个bin，可以通过分配给bin的梯度角范围来识别bin；
3. 在大小为$$(W_2 \times 2) \times(H_2 \times 2)$$的block级别上对HOG特征进行L2-normalization。这样生成了一个深度为4xL的描述子，并给block-level的HOG描述子数量为M个。即每个ROI由一个对应的深度为4xL的HOG描述子，用于检索。

#### Regions based Convolutional Matching

经过HOG描述子计算后，一张图像被转换为M个region，每个region由一个长度为4xL的特征向量描述。根据阈值，可以挑选其中M个region，这样一张检索图像可以由一个二维矩阵A来表示，A的维度为\[G, 4xL]。参考图像则被转换为二维矩阵B，维度为\[M，4xL]。通过矩阵乘法，得到矩阵C，维度为\[G, M]。对矩阵C的每一行进行max-pooling，找到每个检索图像区域最相似的候选参考图像区域，得到了一个长度为G的向量D。取D的算术平均，来得到检索图像和参考图像间的相似度。具有最高相似度的参考图像被认为是匹配图像。

### Results and Analysis

作者用了一个结合准确率为1时召回率和特征提取时间的评价指标，Performance-per-Compute-Unit（PCU）：&#x20;

![](<../../.gitbook/assets/image (1019).png>)

参数设置：$$GW_1=H_1=512, W_2=H_2=16, L=8, GT=0.5$$**.**

![](<../../.gitbook/assets/image (676).png>)

![](<../../.gitbook/assets/image (549).png>)

![](<../../.gitbook/assets/image (548).png>)

![](<../../.gitbook/assets/image (1010).png>)
