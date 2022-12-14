---
description: >-
  DASGIL: Domain Adaptation for Semantic and Geometric-Aware Image-Based
  Localization
---

# \[TIP 2021] DASGIL

{% embed url="https://github.com/HanjiangHu/DASGIL" %}

{% embed url="https://ieeexplore.ieee.org/document/9296559" %}

## Abstract

Image retrieval for localization is an efficient and effective solution to the problem. In this paper, we propose a novel multi-task architecture to fuse the geometric and semantic information into the multi-scale latent embedding representation for visual place recognition. To use the high-quality ground truths without any human effort, the effective multi-scale feature discriminator is proposed for adversarial training to achieve the domain adaptation from synthetic virtual KITTI dataset to real-world KITTI dataset.

## Introduction

Imagebased localization is an efficient way to retrieve target image from database given the query image across different environments. Once the place recognition presents the coarse results, the high-precision 6-DoF camera pose could be regressed with the retrieved initial value.

CNN深层特征具备更多语义信息，因此更加鲁棒。有定位算法引入了语义分割和深度估计，但是这些算法需要人工标注语义或深度数据，耗时耗力过多。虚拟数据更易标注，但是在虚拟数据和真实数据间有较大的的domain gap。

为了在视觉定位任务中使用几何和视觉信息，并且使用较少的标注成本，作者提出DASGIL，a novel domain adaptation with semantic and geometric information for image-based localization. 所提出的方法使用一个encoder、两个decoder的多任务结构，通过共享的Fusion Feature Extractor将几何和语义信息融合到多尺度的隐层特征。The fused features of both virtual and real images follow the same distribution in multi-layer-feature adversarial training through the novel Flatten and Cascade Discriminator, adapting from synthetic images to real-world images. Based on the fused multi-scale features, metric learning for place recognition is accomplished through multi-scale triplet loss for metric learning.&#x20;

For the experiments, we train the model on Virtual KITTI 2 dataset but test it on Extended CMU-Seasons dataset and Oxford RobotCar dataset for retrieval-based localization and place recognition, and our results are better than state-of-the-art methods under various regional environments, vegetation conditions and weather conditions.

本文贡献如下：

* We propose a novel and state-of-the-art approach, DASGIL, fusing semantic and geometric information into latent features through a multi-task architecture of depth prediction and semantic segmentation；
* A novel domain adaptation framework is introduced using multi-scale feature discriminator through adversarial training from synthetic to real-world dataset for representation learning；
* Multi-scale metric learning for place recognition is adopted through multi-layer triplet loss and features from different scales are applied in retrieval process as well；
* A series of comparison experiments have been conducted to validate the effectiveness of every proposed module in DASGIL. And our approach outperforms state-of-the-art image-based localization baselines on the Extended CMU-Seasons dataset and Oxford RobotCar dataset though only supervisely trained on Virtual KITTI 2 dataset.

## DASGIL Architecture

![](<../../../.gitbook/assets/image (249).png>)

### Architecture Overview

DASGIL结果包含一个公用的融合特征提取层E，和两个生成层Gs、Gd用于语义分割和深度估计。当给定虚拟图像Iv，E提取出的多尺度特征用于生成深度图和语义图。为了减少虚拟图像和真实图像间的domain gap，通过多尺度特征分辨器D来进行对抗训练。让虚拟图像Iv和真实图像Ir产生相同的多尺度特征分布。

### Fusion Feature Extractor

E提取出的特征融合了几何信息和语义信息。该网络像UNet一样，将encoder中各层卷积特征（本文为8层）通过skip connection传递给decoder。This structure instructs the model to obtain and use different levels of features containing geometric and semantic information, which assists the the generation of depth map and segmentation map.

### Depth Map and Segmentation Map Generator

![](<../../../.gitbook/assets/image (265).png>)

Gd和Gs结构相同，decoder接收了来自encoder各层特征，两个decoder的输入相同。

### Multi-Scale Discriminator

由于E是在虚拟数据上训练的，在真实数据上测试，因此需要保证融合了几何和语义信息的多尺度特征在虚拟数据和真实数据上分布是一致的。为了实现从虚拟数据到真实数据的domain adaptation，作者在多尺度特征空间中进行了对抗训练。

![](<../../../.gitbook/assets/image (221).png>)

在输入特征分辨器D之前，将多尺度特征拼接起来，并经过batchnorm层。所提出的特征分辨器包含三个全连接层，作为一个二分类器，来判断特征时来自真实数据还是虚拟数据。该分辨器为Flatten Discriminator （FD）。

但是特征是从不同层提取出的，直接分辨所有拼接在一起的特征可能是无效的。为了在不同domain间区分浅层特征和深层特征，作者提出了Cascade Discriminator （CD）。对于某一特征图，浅层特征图被输入一个CNN，输出与下一更深层的特征图相连接，最后生成二分类结果。

![](<../../../.gitbook/assets/image (260).png>)

These two multi-layer discriminator structures allow the model to recognize R and V from multiple levels, enabling the model to better utilize the fusion information extracted.

## DASGIL Pipeline for Image-based Localization

### Domain Adaptation for Multi-task Training

#### Multi-scale Depth Reconstruction Loss

作者对Gd各层得到的深度分割结果进行了监督，将其与深度真值计算L1损失：

![](<../../../.gitbook/assets/image (246).png>)

#### Cross Entropy Segmentation Loss

作者只对Gs最后一层的语义分割结果进行了监督，使用交叉熵损失：

![](<../../../.gitbook/assets/image (224).png>)

#### Multi-scale Feature Adversarial Loss

记虚拟图像为$$I_V \in p_V(I)$$，真实图像为$$I_V \in p_V(I)$$，作者采用了LSGAN的形式：

![](<../../../.gitbook/assets/image (247).png>)

在对抗训练过程中，$$L_{Dis}$$和$$L_{Gen}$$被分别优化。

### Representation Learning for Place Recognition

#### Multi-scale Fusion Triplet Loss

作者对特征编码器E中的特征计算了triplet loss。这些特征隐式地包含了图像信息、深度信息和语义信息。

![](<../../../.gitbook/assets/image (256).png>)

A triplet loss involves an anchor virtual image $$q_i \in q_V(I)$$, a positive sample $$q_{i+}$$ representing the same scene as the anchor and a negative sample $$q_{i-}$$ which is unrelated to the anchor image. The triplet loss $$L_t(q_i,q_{i+},q_{i-},m)$$ is shown in the formula below:

![](<../../../.gitbook/assets/image (266).png>)

作者还将上式扩展为多尺度形式：

![](<../../../.gitbook/assets/image (271).png>)

注意不同层的margin可能不同。

#### Total Adversarial Training Loss

网络训练总的loss为：

![](<../../../.gitbook/assets/image (248).png>)

在generation optimizing process中使用公式8进行优化，在discrimination optimizing process中使用公式7优化。

### Image Retrieval for Localization

![](<../../../.gitbook/assets/image (289).png>)

Lower-level feature suffers from the change of view point during the retrieval, while higher-level feature is sensitive to the environmental variance, i.e. seasonal change, illumination change, etc. Therefore, the middle-layer features are chosen to calculate the similarity finally.

在图像检索时，只利用了特征编码器E，提取图像的多尺度特征，计算L1距离或cosine相似度。

## Experiments

模型在KITTI和Virtual KITTI 2数据集上训练。

For the model with the flatten discriminator, the total epoch is set to be 5 to avoid the collapse of adversarial training. While the total training epoch is 40 for the model with the cascade discriminator.

在Extended CMU Seasons Dataset上使用L1距离；在场景识别数据集上，Recall@N is the average ratio of one of the top N retrieved candidates lies within 25m around the groundtruth. Top-1 Recall@D is the percentage that the top 1 retrieval lies within distance threshold D form 15m to 50m with respect to the groundtruth position. The retrieval metric in this experiment is cosin distance.

Since there are different variations on camera angles and environments for any image sequence in the virtual KITTI 2 dataset, the positively paired images are within 5-image-interval along a sequence but from different environments while the negatively paired ones are randomly chosen and flipped. The negative pairs could come from all the environments with different scenes. All the images are randomly horizontally flipped for data augmentation while keeping the same transformation for positive pairs. The features from the middle-four layers are used to construct the multi-scale triplet loss. For image retrieval, features from the fifth and sixth layers are used as representation with FD while only the fifth layer feature is used for CD model. The margins are set to be 1 for the features at the third, fourth, fifth and sixth layer in the multi-scale triplet loss.

### Extended CMU-Seasons Dataset Results

![](<../../../.gitbook/assets/image (282).png>)

![](<../../../.gitbook/assets/image (222).png>)

![](<../../../.gitbook/assets/image (252).png>)

### Oxford RobotCar Dataset Results

![](<../../../.gitbook/assets/image (285).png>)

DASGIL-CD performs better than DASGIL-FD for Long-term and Snow place recognition, which shows that the cascade discriminator is beneficial to the recall with more candidates under challenging environments.

### Ablation Study

#### Training Modules in DASGIL

![](<../../../.gitbook/assets/image (234).png>)

It could be concluded that all of the Depth generation module and Segmentation generation module are effective and indispensable in the proposed DASGIL framework. The geometric information from the depth map generation is more important to the place recognition than the semantic information form segmentation map.

While for different types of discriminator, Single FD represents that only single-layer feature is used when training flatten discriminator, where the 5th layer feature is the best among all the 8 layers in the experiment and the retrieval is based on the 5th layer as well. Multiple FD and Multiple CD use features from all the layers as input. It could be seen that without any discriminator the performance is the worst and the results of multi-layer discriminator are better than single flatten discriminator. Also, Multiple Triplet Loss (3rd to 6th layers) gives better the results compared to the best Single Triplet Loss (5th layer) for triplet loss.

#### Feature Representation for Retrieval

![](<../../../.gitbook/assets/image (232).png>)

The model used in this ablation study is with multi-layer triplet loss from the 3rd to 6th layers. The Single-layer indicates that the 5th layer feature is used for retrieval which is the best among all layers. The Multi-layer uses the features from 5th and 6th layers for best retrieval results. NetVLAD layer is fine-tuned on the fixed pretrained DASGIL models of both FD and CD models for 5 epoch. R-MAC is implemented directly to the features through pretrained models when testing. For the multi-layer feature, we adopt the L2 distance to sum up the output of NetVLAD or R-MAC for image retrieval.

直接对特征求L1距离的方法最好。

For the model with Flatten Discriminator, the retrieval with multi-layer feature is better than the single-layer feature. While the model with Cascade Discriminator obtains better results with single-layer feature retrieval in park and urban scenarios, indicating that the Cascade Discriminator module could make the feature from each layer follow the same distribution more strictly so that single-layer representation could be used for image retrieval and localization.
