---
description: Toward Geometric Deep SLAM
---

# \[arxiv 2017] MagicPoint

{% embed url="https://arxiv.org/abs/1707.07410.pdf" %}

### Abstract

论文提出了一种基于两个DCN的point tracking system。第一个DCN就是MagicPoint，提取图像的显著二维坐标点（只有detector）；第二个DCN是MagicWarp，输入利用MagicPoint得到的一对图像中的二维坐标点信息，直接预测homography（不需要descriptor信息）。

### Introduction

作者先抛出了一个问题

> what would it take to build an ImageNet for SLAM?&#x20;
>
> What would it take to build DeepSLAM?

由于SLAM领域的真实数据往往无法获得很好的标注，而仿真数据无法囊括现实中的所有变化，所以可能引起domain adaptation issues和过拟合。所以用data-driven的深度学习方法去解决SLAM问题尚未解决。 作者提到了两个点，首先利用预测图像的DCN去估计ego-motion是可能的，作者没有使用直接用图像估计6DoF位姿的监督方法，而是更关注geometric consistency；其次作者发现对于SLAM系统来说，预测和对齐关键点已经足够去解算pose，那么就不用去预测整幅图像了，直接估计homography足以满足需求。

### Method

#### Overview

![](<../../.gitbook/assets/image (1069).png>)

#### MagicPoint

作者设计MagicPoint的motivation就是认为hand-crafted detector需要过多的经验和技巧，往往无法cover所有的干扰，所以就直接用DCN去估计pixel-level的显著性，提取图像关键点。&#x20;

![](<../../.gitbook/assets/image (1031).png>)

结构类似于VGG。输入一个图像，得到一个同等分辨率的point response image，输出的每个pixel的值代表原图中这个位置是角点的概率。但是直接用encoder下采样-decoder上采样的结构恢复分辨率很耗算力，所以作者用网络得到了1/8大小的feature map，维度是65维（65个通道），这65个通道对应原图中不重叠的8x8的区域和一个dustbin通道（用于表示该8x8区域内无关键点），最后reshape到原本分辨率，这样decoder就没有参数了。&#x20;

![](../../.gitbook/assets/magicpoint\_3.png)

训练时使用OpenCV作了一批虚拟的几何体，几何体的角点可以直接得到，然后加入噪声、光照变化等进行数据增强。训练时对feature map上15x20个grid中的每个cell进行softmax处理，然后在每个cell中计算cross-entropy loss。

#### MagicWarp

MagicWarp输入一对图像的关键点，然后估计homography。比如两幅120x160的图像输入MagicPoint，分别得到65x15x20的feature map。输入MagicWarp后，先从channel维度上进行concatenation，得到130x15x20的feature map，然后经过一个VGG型的encoder，再通过全连接层降维，得到一个9-d的向量，恢复成3x3的homography矩阵。&#x20;

![](<../../.gitbook/assets/image (305).png>)

训练时，将仿真得到的点云投影到虚拟视角下的相机上，得到两个视角下相互匹配的点，作为训练数据，计算loss时，用估计的homography将图1的点投影到图2，然后计算投影误差，用L2-distance作为loss。&#x20;

![](<../../.gitbook/assets/image (880).png>)

### MagicPoint Evaluation

#### 评估指标

**Corner Detection Average Precision** 对于角点检测，约定如果检测到点的位置与最近的真值角点之间的距离小于阈值$$\varepsilon=4$$，则该点检测正确，即定义correctness为：&#x20;

![](../../.gitbook/assets/magicpoint\_0.png)

通过改变检测可信度来得到准确率-召回率曲线，并用对应的Area Under Curve（也称Average Precision）来综合评估。&#x20;

**Corner Localization Error** 定义Localization error为：&#x20;

![](<../../.gitbook/assets/image (1009).png>)

localization error在$$(0,\varepsilon)$$之间，越小越好。

**Repeatability** 作者还定义了repeatability，即一个点在下一帧中被检测到的概率。作者只计算了sequential repeatability（在第t帧和第t+1帧之间）。这里作者定义correctness的阈值为$$\varepsilon=2$$，假设图1中有$$N_1$$个点，图2中有$$N_2$$个点，定义repeatability实验中的correctness为：

![](<../../.gitbook/assets/image (886).png>)

而repeatability定义为一个点在另一张图像中出现的概率：&#x20;

![](../../.gitbook/assets/magicpoint\_2.png)

对于每个图像序列，作者先计算了repeatability与检测点数量的相关曲线，然后找到最大repeatability的点，并以repeatability@N的形式作为评估结果，其中N为最大repeatability时的平均监测点数量。

#### 在Synthetic Shapes数据集上的评估

![](../../.gitbook/assets/magicpoint\_4.png)

![](<../../.gitbook/assets/image (342).png>)

作者研究了噪声程度对检测的影响&#x20;

![](<../../.gitbook/assets/image (365).png>)

![](<../../.gitbook/assets/image (329).png>)

#### 在30 Static Corners数据集上的评估

这个是真实数据，包含30个序列，相机是静止不动的，只有光照在变化，人为标注第一帧的点。&#x20;

![](<../../.gitbook/assets/image (1002).png>)

图像尺寸对模型的影响&#x20;

![](<../../.gitbook/assets/image (203).png>)

> For an input image size 160x120, the average forward pass times on a single CPU for MagicPointS and MagicPointL are 5.3ms and 19.4ms respectively. For an input image size of 320x240, the average forward pass times on a single CPU for MagicPointS and MagicPointL are 38.1ms and 150.9ms respectively. The times were computed with BatchNorm layers folded into the convolutional layers.

### MagicWarp Evaluation

在四种变换下测试magicwarp，图像中的点是随机放置的。其中Translation是简单的右移，Rotation是平面内旋转，Scale是缩放，Random H是一个随机的单应性变换。Nearest Neighbor方法采用3x3单位矩阵作为H的估计，MagicWarp采用模型的输出作为H的估计。 定义Match Correctness为：&#x20;

![](<../../.gitbook/assets/image (681).png>)

其中H是预测的变换，$$\hat{H}$$是真值变换。 Match Repeatability为正确匹配的百分比：&#x20;

![](<../../.gitbook/assets/image (490).png>)

![](<../../.gitbook/assets/image (309).png>)

![](<../../.gitbook/assets/image (1057).png>)

![](../../.gitbook/assets/magicpoint\_15.png)

magicwarp可以应对更大的图像变换。

> MagicWarp is very efficient. For an input size of 20x15x130 (corresponding to an image size of 160x120), the average forward pass time on a single CPU is 2.3 ms. For an input size of 40x30x130 (corresponding to an image size of 320x240), the average forward pass time on a single CPU is 6.1ms. The times were computed with BatchNorm layers folded into the convolutional layers.
