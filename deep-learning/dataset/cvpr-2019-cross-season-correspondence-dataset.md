---
description: A Cross-Season Correspondence Dataset for Robust Semantic Segmentation
---

# \[CVPR 2019] Cross-Season Correspondence Dataset

{% embed url="https://arxiv.org/abs/1903.06916v2" %}

{% embed url="https://visuallocalization.net" %}

### Abstract

这篇论文中，作者提出一种利用不同视觉条件下图像的2D-2D点匹配来训练语义分割网络的方法。通过要求匹配点的语义一致性，来让语义分割网络在不同视觉条件下更鲁棒。

### Introduction

论文的主要贡献为：

1.不同视觉条件下图像之间的点匹配为语义分割网络的训练提供了新的约束，即匹配点应当有一致的语义，作者以此作为一个损失函数；

2.作者获得点对应关系的方法不需要真值，只需少量的人工干预；

3.本篇论文得到的模型在多变视觉条件下表现有显著提升。

### Semantic Correspondence Loss

作者将从数据集中获得一个样本记为$$(I^r,I^t,x^r,x^t)$$，其中$$I^r$$是reference traversal中的一张图像，$$I^t$$是target traversal中的一张图像，x分别是两张图像中匹配的关键点位置。reference每次取同一次traversal，而target从其他traversal随机选取。因此，匹配损失函数$$L_{corr}$$可以记为：&#x20;

![](<../../.gitbook/assets/image (511).png>)

其中，l为hinge loss或cross-entropy loss。 令$$d_x \in R^F$$为语义分割网络在点x处获得的长度为F的特征向量。则correspondence hinge loss被定义为：&#x20;

![](<../../.gitbook/assets/image (35).png>)

而对于correspondence cross-entropy loss，作者首先从reference image的最后特征图中得到最可能的语义类别，用一个one-hot编码向量$$c_{x_i}$$来表示$$x_i$$位置点的最可能的类别，损失函数可以写为：&#x20;

![](<../../.gitbook/assets/image (494).png>)

在训练过程中，作者在标准的cross-entropy loss之外加入correspondence loss进行训练。

### A Cross-Season Correspondence Dataset

数据集中每个样本包含了不同季节或天气条件下采集的两张邻近图像，并且包含图像间2D-2D的点对应关系。点对应关系是自动通过两点间3D几何一致性来得到的。在不同视觉条件下，几何关系要比光度信息更稳定。该数据集是基于CMU和RobotCar建立的。&#x20;

![](<../../.gitbook/assets/image (705).png>)

该数据集的建立可以分为四步：1.计算每张图像的位姿；2.对每个条件或traversal下的环境建立一个稠密的3D点云；3.不同条件下的3D点云进行匹配，由于所有的点云是在同一坐标系下，所以可以通过位置来匹配，不需要特征描述子，避免了视觉条件的干扰；最后，根据相机位姿，基于3D点云的匹配关系，获得2D-2D的匹配关系。

#### CMU Seasons Correspondence Dataset

![](<../../.gitbook/assets/image (1038).png>)

#### Oxford RoBotCar Correspondence Dataset

由于视觉条件较差，所以用RGB图像去进行MVS点云中的点太少了，所以作者用了Lidar点云去建图，利用真值的pose和时间戳，去获得点云到图像的映射。&#x20;

![](<../../.gitbook/assets/image (824).png>)

### Inplementation Details

作者使用了在Cityscapes上预训练过的PSPNet作为模型。除了Cityscapes训练图像外，作者来添加了一个CMU和RobotCar粗略标注的图像。这一过程，是为了避免模型将所有像素预测为同一类别，让模型在Cityscapes上依然具有好的表现。&#x20;

作者还添加了一个运行过程中进行的correspondence refinement阶段，即那些在reference image中被分类为不稳定类别点的匹配被删除。被纳入考虑的类别有person,rider,car,truck,bus,train,motorcycle和bicycle。在加入correspondence loss前，模型先经过了500代warm-up，来保证模型对于reference images有较好的分割效果。
