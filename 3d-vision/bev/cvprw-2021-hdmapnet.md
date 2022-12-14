---
description: 'HDMapNet: An Online HD Map Construction and Evaluation Framework'
---

# \[CVPRW 2021] HDMapNet

## Abstract

HDMapNet encodes image features from surrounding cameras and/or point clouds from LiDAR, and predicts vectorized map elements in the bird’s-eye view.

## Introduction

<figure><img src="../../.gitbook/assets/image (620).png" alt=""><figcaption></figcaption></figure>

HDMapNet不是为了构建全局高精度地图，而是提供了一种简单的方法来构建局部语义地图。本文贡献在于：

* We propose a novel online framework to construct HD semantic maps from the sensory observations, and together with a method named HDMapNet.&#x20;
* We come up with a novel feature projection module from perspective view to bird’s-eye view. This module models 3D environments implicitly and considers the camera extrinsic explicitly.&#x20;
* We develop comprehensive evaluation protocols and metrics to facilitate future research.

## Semantic Map Learning

<figure><img src="../../.gitbook/assets/image (654).png" alt=""><figcaption></figcaption></figure>

### HDMapNet

HDMapNet通过神经网络以单帧图像I和点云P为输入，预测地图要素M。模型包括四个部分：一个perspective view image encoder $$\phi_I$$，一个图像分支的neural view transformer $$\phi_v$$，一个pillar-based point cloud encoder $$\phi_P$$和一个地图要素decoder $$\phi_M$$.

HDMapNet(Surr)以环视相机图像为输入，HDMapNet(LiDAR)以LiDAR为输入，HDMapNet(Fusion)将两者都作为输入。

#### Image encoder

image encoder包含perspective view image encoder和neural view transformer两部分。

**Perspective view image encoder.** 图像分支从$$N_m$$​个环视相机中获取perspective view输入，获得了场景的全景内容。对每个图像$$I_i$$​用一个公用的网络$$\phi_I$$处理，获得perspective view特征图$$F^{pv}_{I_i}$$.

**Neural view transformer.** 如图3，首先将图像特征从perspective view转换到相机坐标系中，然后转换到BEV下。perspective view和相机坐标系中任意两个像素之间的关联可以用一个MLP $$\phi_{V_i}$$来表示：

<figure><img src="../../.gitbook/assets/image (658).png" alt=""><figcaption></figcaption></figure>

其中$$\phi^{hw}_{V_i}$$​表示相机坐标系下(h,w)处的特征向量与perspective view特征图中所有像素之间的关联。BEV（ego coordinate system）特征$$F^{bev}_{I_i}$$​是将$$F^c_{I_i}$$根据相机外参投影得到的。最后的图像特征$$F^{bev}_I$$是$$N_m$$个相机特征的均值。

#### Point cloud encoder

​point cloud encoder $$\phi_P$$是带有dynamic voxelization的PointPillar的变种，它将3D空间划分为多个pillars，并从pillar-wise点云的pillar-wise特征中学习到特征图。输入是点云中的N个Lidar points。对每个点p，它有三维坐标和额外的K维特征，记为$$f_p \in R^{K+3}$$.

当将特征投影到BEV时，可以由多个点落到同个pillar中。作者定义pillar j中的点集为$$P_j$$​，用PointNet将一个pillar中点的特征聚合在一起：

<figure><img src="../../.gitbook/assets/image (616).png" alt=""><figcaption></figcaption></figure>

然后pillar-wise特征通过一个CNN $$\phi_{pillar}$$​进行编码，记BEV下的特征图为$$F^{bev}_P$$.

#### Bird's-eye view decoder

地图是一个复杂的图网络，包含车道线和道路边缘的实例级和方向信息。车道线需要被矢量化，以便自动驾驶车辆使用。因此，BEV decoder $$\phi_M$$不仅输出语义分割结果，还预测实例编码和车道线方向。用一个后处理过程来从编码中聚合实例，并对其进行矢量化。

**Overall architecture.** BEV decoder采用三分支的FCN结构，即语义分割分支、实例编码分支和方向预测分支。BEV decoder的输入是图像特征图$$F^{bev}_I$$​或点云特征图$$F^{bev}_P$$，如果两者都存在，则将他们拼接起来。

**Semantic prediction.** 语义预测模块是一个FCN，用交叉熵损失来训练这一模块。

**Instance embedding.** 该模块用于将每个BEV embedding进行聚类。记C为真值聚类数，$$N_c$$​是聚类c中element的数量，$$\mu_c$$是聚类c的embedding均值，$${[x]}_+=max(0,x)$$，$$\delta$$是损失函数中的margin，则聚类损失为：

<figure><img src="../../.gitbook/assets/image (614).png" alt=""><figcaption></figcaption></figure>

**Direction prediction.** 这一模块用于预测每个pixel C中车道线的方向。方向被离散化为$$N_d$$​个均匀分布的类别。通过对当前像素$$C_{now}$$​的方向D进行分类，下一个车道线像素$$C_{next}$$​可以获得：$$C_{next}=C_{now}+\triangle_{step}\cdot D$$，其中$$\triangle_{step}$$是预设的步长。由于我们不知道车道线的方向，我们无法辨别每个node的前向和后向方向，因此作者将它们都视为positive。具体的来说，就是每个车道线node的标签是一个$$N_d$$​维向量，其中两个值是1，其他是0. Note that most of the pixels on the topdown map don’t lie on the lanes, which means they don’t have directions. The direction vector of those pixels is a zero vector and we never do backpropagation for those pixels during training. 作者用softmax作为分类的激活函数。

**Vectorization.** 在inference时，首先用Density-Based Spatial Clustering of Applications with Noise (DBSCAN)将cluster instance embeddings进行聚类，用NMS去避免冗余，Finally, the vector representations are obtained by greedily connecting the pixels with the help of the predicted direction.

### Evaluation

作者提出了语义地图学习的一些评价指标，包括语义指标和实例指标。

#### Semantic metrics

模型的语义预测可以用Eulerian和Lagrangian两种形式去评估。Eulerian指标在一个dense grid上计算，评估像素值的差异；Lagrangian指标偏重于形状，度量形状的空间分布。

**Eulerian metrics.** 用IoU作为Eulerian metrics：

<figure><img src="../../.gitbook/assets/image (585).png" alt=""><figcaption></figcaption></figure>

其中$$D_1,D_2 \in R^{H ]\times W \times D}$$​是形状的稠密表征（grid中光栅化的曲线）,H和W是grid的高和宽，D是类别数量。$$|\cdot|$$​是集合的大小。

**Lagrangian metrics.** 我们还对结构化输出感兴趣，即由连接点组成的曲线。为了度量预测曲线和真值曲线间的空间距离，作者用两个曲线的采样点集间的Chamfer distance (CD)：

<figure><img src="../../.gitbook/assets/image (615).png" alt=""><figcaption></figcaption></figure>

其中$$CD_{dir}$$​是有向的Charmfer distance。

#### Instance metrics

进一步评估模型的实例检测能力。用average precision (AP)：

<figure><img src="../../.gitbook/assets/image (592).png" alt=""><figcaption></figcaption></figure>

We collect all predictions and rank them in descending order according to the semantic confidences. Then, we classify each prediction based on the CD threshold. For example, if the CD is lower than a predefined threshold, it is considered true positive, otherwise false positive. Finally, we obtain all precisionrecall pairs and compute APs accordingly.

## Experiments

### Inplementation details

**Tasks & Metrics.** Due to the limited types of map elements in the nuScenes dataset, we consider three static map elements: lane boundary, lane divider, and pedestrian crossing.

**Architecture.** For the perspective view image encoder, we adopt EfficientNet-B0 pre-trained on ImageNet. Then, we use a multi-layer perceptron (MLP) to convert the perspective view features to bird’s-eye view features in the camera coordinate system. The MLP is shared channel-wisely and does not change the feature dimension. For point clouds, we use a variant of PointPillars with dynamic voxelization. We use a PointNet with a 64dimensional layer to aggregate points in a pillar. ResNet with three blocks is used as the BEV decoder.

**Training details.** We use the cross-entropy loss for the semantic segmentation, and use the discriminative loss (Equation 5) for the instance embedding where we set $$\alpha=\beta=1, \delta_v=0.5, \delta_d=3.0$$. We use Adam for model training, witfh a learning rate of 1e−3.

### Results

<figure><img src="../../.gitbook/assets/image (611).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (668).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (667).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (603).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (579).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (650).png" alt=""><figcaption></figcaption></figure>

Temporal fusion：We first conduct short-term temporal fusion by pasting feature maps of previous frames into current’s according to ego poses. The feature maps are fused by max pooling and then fed into decoder.
