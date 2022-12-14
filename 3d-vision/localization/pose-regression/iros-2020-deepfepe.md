---
description: Deep Keypoint-Based Camera Pose Estimation with Geometric Constraints
---

# \[IROS 2020] deepFEPE

{% embed url="https://ieeexplore.ieee.org/document/9341229/" %}

{% embed url="https://github.com/eric-yyjau/pytorch-deepFEPE" %}

## Abstract

In this paper, we design an end-to-end trainable framework consisting of learnable modules for detection, feature extraction, matching and outlier rejection, while directly optimizing for the geometric pose objective.

## Introduction

基于学习的位姿估计系统表现尚不如传统的基于SIFT的算法，其原因可能来自于：1.这些学习的系统是为不同目的独立训练的，没有端到端地为位姿估计任务进行训练和优化。几何约束和位姿估计的目标没有充分地结合到pipeline中。2.基于学习的方法存在过拟合问题，泛化性差；3.现有基于学习的特征detector比SIFT detector要差，这是因为很难获得带有准确关键点和correspondence真值的训练样本。

为此，作者设计了一种端到端对的框架，用于相对位姿估计。受传统基于几何方法的启发，该框架包含可训练的关键点检测、特征描述、外点剔除模块。整个框架可以端到端地利用真值相机位姿来训练。此外，为了获得足够准确的关键点，作者在pipeline中引入了softargmax detector，因此最后的相机位姿估计误差可以反向传播，提供亚像素级的监督。

本文贡献总结如下：

1. We propose the keypoint-based camera pose estimation pipeline, which is end-to-end trainable with better robustness and generalizability than the learning-based baselines.
2. The pipeline is connected with the novel Softargmax bridge, and optimized with geometry-based objectives obtained from correspondences.
3. The thorough study on cross-dataset setting is done to evaluate generalization ability, which is critical but not much discussed in the existing works.The thorough study on cross-dataset setting is done to evaluate generalization ability, which is critical but not much discussed in the existing works.

![](<../../../.gitbook/assets/image (464).png>)

## Method

### Overview

本文提出一种基于深度学习得到位姿估计pipeline，称为Deep learning-based Feature Extraction and Pose Estimation (DeepFEPE)，该pipeline输入两帧图像，估计相对位姿。如图1所示，该pipeline主要包含两个基于学习的模型，分别用于提取特征和位姿估计。&#x20;

![](<../../../.gitbook/assets/image (467).png>)

softargmax使得关键点检测达到亚像素精度，并且使得梯度能够回传到点坐标。在损失函数设计中，作者不仅拟合F矩阵，而且对分解后的位姿进行了约束。

**Notation** 定义一对图像$$I,I'\in R^{H \times W}$$​，第i帧到第j帧的变换矩阵为$$T_{ij}=[R|t]$$​，其中R是旋转矩阵，t是平移向量。2D图像坐标系内的点为p=\[u,v].

### Feature Extraction (FE)

作者使用了SuperPoint作为基本组件，SuperPoint输出灰度图，输出关键点heatmap $$H_{det}$$​和描述子$$H_{desc}$$​.

**softargmax detector head** 为了实现端到端训练，作者提出用2D softargmax来作为detector head。SuperPoint原本是在$$H_{det}$$​上进行NMS来提取稀疏的关键点，但是NMS只有像素级精度，并且不可微。因此，作者在每个NMS提取到的关键点附近5x5邻域内使用softargmax，最后每个关键点的坐标为：

![](<../../../.gitbook/assets/image (433).png>)

在每个2D区域内：

![](<../../../.gitbook/assets/image (479).png>)

f(u,v)表示(u,v)处的heatmap值，i,j表示相对中心点$$(u_0,v_0)$$​的相对坐标。

To pre-train the FE module with Softargmax, we convolve the ground truth 2D detection heatmap with a Gaussian kernel $$\sigma_{fe}$$. The label of each keypoint is represented as a discrete Gaussian distribution on a 2D image.To pre-train the FE module with Softargmax, we convolve the ground truth 2D detection heatmap with a Gaussian kernel $$\sigma_{fe}$$. The label of each keypoint is represented as a discrete Gaussian distribution on a 2D image.

**descriptor sparse loss** 为了预训练FE，作者使用了稀疏的描述子损失。作者在每对匹配图像间稀疏地采样了N对正样本和M对负样本，构成MxN对correspondence。损失函数依然采用SuperPoint中的平均constrastive loss。

**output of feature extractor** 根据稀疏的特征得到correspondence。为了获得关键点，在heatmap上使用NMS，并设定一个阈值去剔除冗余。描述子从$$H_{desc}$$中用双线性插值获得。得到两组特征后，用双向最近邻来获得N对correspondence，构成一个Nx2矩阵，作为位姿估计的输入。​

### Pose Estimation (PE)

作者基于Deep Fundamental Matrix Estimation （DeepF）从带有噪声的correspondence中构建可微的位姿估计pipeline，来替代RANSAC，并设计了基于几何的损失函数来训练DeepFEPE。

**existing objective for learning fundamental matrix** DeepF将F矩阵估计视为一个加权的最小二乘问题。correspondence的权重代表了匹配的可信度，使用一个PointNet型的神经网络预测得到的。然后，用这一权重和点去求解F矩阵。用平均Sampson距离来计算预测的残差。correspondence、权重和残差被循环输入模型来调整权重。具体而言，残差被定义为：

![](<../../../.gitbook/assets/image (399).png>)

其中p和p’为归一化平面上的一对correspondence。这一损失函数定义了点到其匹配点的投影极线之间的距离，在本文中，被称为**F-loss**。

**geometry-based pose loss** 由于极线空间的良好估计并不意味着良好的位姿估计。作者提出基于几何的损失函数来约束估计位姿和真值位姿。估计的F矩阵被转换为E矩阵，然后分解为两组旋转矩阵和两组平移向量。从四组可能的解中选取所有点位于相机前方的一组解，获得了四元数表示的旋转向量和平移向量，计算基于几何的损失函数：

![](<../../../.gitbook/assets/image (403).png>)

最后的损失函数为：

![](<../../../.gitbook/assets/image (451).png>)

这一损失函数被称为**pose-loss**。

### Training Process

FE使用SuperPoint的训练方法预训练的。由预训练PE获得correspondence后，用F-loss初始化PE。这个网络是一个迭代5次的RNN结构。当使用pose-loss训练时，$$c_r=0.1,c_t=0.5,\lambda_{rt}=0.1$$.

When connecting the entire pipeline, the gradients from pose-loss flow back through the Pose Estimation(PE) module to update the prediction of weights, as well as the Feature Extraction(FE) module to update the locations of keypoints.

## Experiments&#x20;

作者是连续的相邻帧来训练和测试的。在KITTI00-08（16k个样本）上进行训练，在KITTI09-10（2710个样本）上进行测试. 在ApolloScape Road 11（5.8k个样本）上进行测试。

![](<../../../.gitbook/assets/image (435).png>)

### Relative Pose Estimation

旋转误差计算如下：

![](<../../../.gitbook/assets/image (443).png>)

其中Rog()将旋转矩阵转换为Rodrigues旋转向量，向量的长度指示了角度误差。由于尺度不明确，将平移向量固定为单位向量：

![](<../../../.gitbook/assets/image (454).png>)

得到了平均向量间的角度误差。With the rotation and translation error for each pair of images throughout the sequence, we compute the inlier ratio, from 0% to 100%, under different thresholds. Mean and median of the error are also computed in degrees.

![](<../../../.gitbook/assets/image (429).png>)

![](<../../../.gitbook/assets/image (424).png>)

![](<../../../.gitbook/assets/image (438).png>)

![](<../../../.gitbook/assets/image (455).png>)

### Visual Odometry Evaluation

![](<../../../.gitbook/assets/image (457).png>)

### SuperPoint Correspondence Estimation

![](<../../../.gitbook/assets/image (418).png>)
