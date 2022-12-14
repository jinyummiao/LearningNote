---
description: 'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis'
---

# \[ECCV 2020] NeRF

## Abstract

We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (nonconvolutional) deep network, whose input is a single continuous 5D coordinate (spatial location (x, y, z) and viewing direction ($$\theta,\phi$$)) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis.

## Introduction

![](<../../.gitbook/assets/image (631).png>)

* An approach for representing continuous scenes with complex geometry and materials as 5D neural radiance fields, parameterized as basic MLP networks.
* A differentiable rendering procedure based on classical volume rendering techniques, which we use to optimize these representations from standard RGB images. This includes a hierarchical sampling strategy to allocate the MLP’s capacity towards space with visible scene content.
* A positional encoding to map each input 5D coordinate into a higher dimensional space, which enables us to successfully optimize neural radiance fields to represent high-frequency scene content.

## Neural Raidance Field Scene Representation

用一个5D向量函数来表示连续场景，输入是3D坐标$$\mathbf{x}=(x,y,z)$$和2D视角方向$$(\theta,\phi)$$​，输出是颜色$$\mathbf{c}=(r,g,b)$$​和volume density $$\sigma$$​.实际上，作者用3D笛卡尔单位向量d来表示方向，用一个MLP网络来近似这种连续的5D场景表征$$F_\Theta:(\textbf{x,d}\rightarrow(\textbf{c},\sigma))$$，优化它的权重$$\Theta$$​，来将每个输入的5D坐标映射到它对应的volume density和这个方向的颜色。

这种场景表征需要满足各视角的一致性，因此需要将volume density $$\sigma$$​定义为一个只和位置有关的函数，将RGB颜色$$\textbf{c}$$​定义为和位置和视角都有关的函数。为了实现这一点，MLP $$F_\Theta$$先用8层全连接层（+ReLU，每层有256个神经元）处理输入的3D坐标**x**，输出$$\sigma$$和一个256维的特征向量。该向量然后与相机射线的视角拼接在一起，输出额外的一个全连接层（+ReLU，128个神经元），输出和视角相关的RGB颜色。

![](<../../.gitbook/assets/image (623).png>)

## Volume Rendering with Radiance Fields

Our 5D neural radiance field represents a scene as the volume density and directional emitted radiance at any point in space.

volume density $$\sigma(\textbf{x})$$可以解释为一条射线在位置​**x**的无穷小粒子处终止的概率，即密度场。射线$$r(t)=o+td$$（近处和远处的边界分别为$$t_n,t_f$$）的预期颜色$$C(r)$$为:

![](<../../.gitbook/assets/image (619).png>)

函数T(t)是射线从$$t_n$$​到_t_的透射率，即射线从$$t_n$$​到_t_移动且不碰到任何粒子的概率。从连续神经辐射场渲染一个视图需要估计由虚拟相机的每个像素发射的相机射线的积分C(r)。

针对上式的推导：

给定射线$$r=(o,d)$$​，沿射线的任意点x都可以写作$$r(t)=o+td$$。密度与透射率函数T(t)密切相关，表示光线在区间\[0,t)上传播而没有击中任何粒子的概率。则从t到t+dt过程中没有击中任何粒子的概率为：

$$
T(t+dt)=T(t)\cdot (1-dt \cdot \sigma(t))
$$

$$
\frac{T(t+dt)-T(t)}{dt}=T'(t)=-T(t)\cdot \sigma(t)
$$

​求解得到：

$$
T(a\rightarrow b)=exp(-\int^b_a \sigma(t) dt)
$$

$$T(a\rightarrow b)$$定义为光线从a到b而没有碰到任何粒子的概率。



We numerically estimate this continuous integral using quadrature, which is typically used for rendering discretized voxel grids, would effectively limit our representation’s resolution because the MLP would only be queried at a fixed discrete set of locations. 因此，作者采用分层采样方法，将$$[t_n,t_f]$$划分为N个均匀分布的bins，然后从每个bin中随机抽取一个样本：

![](<../../.gitbook/assets/image (588).png>)

Although we use a discrete set of samples to estimate the integral, stratified sampling enables us to represent a continuous scene representation because it results in the MLP being evaluated at continuous positions over the course of optimization. 作者用这些样本去估计C(r)：

![](<../../.gitbook/assets/image (608).png>)

其中$$\delta_i=t_{i+1}-t_i$$是相邻两个样本之间的距离。

## Optimizing a Neural Radiance Field

为了让NeRF能够表征高分辨率的复杂场景，作者引入两个改进策略，其一为输入坐标的位置编码，帮助MLP表征高频函数，其二是分层采样过程，能够有效采样高频表征。

### Positional encoding

重新定义连续的场景表征为$$F_\Theta=F_\Theta' \circ \gamma$$​，其中$$F_\Theta'$$是上文所述可学习的MLP，$$\gamma$$无需训练：

![](<../../.gitbook/assets/image (577).png>)

对坐标**x**的三个量（归一化到\[-1,1]间）和笛卡尔方向单位向量d的三个量（构建到\[-1,1]间）分别用$$\gamma(\cdot)$$进行处理。在实验中，$$\gamma(\textbf{x})$$中的L=10，$$\gamma(d)$$中L=4。

### Hierarchical volume sampling

Our rendering strategy of densely evaluating the neural radiance field network at N query points along each camera ray is inefficient: free space and occluded regions that do not contribute to the rendered image are still sampled repeatedly.

![](<../../.gitbook/assets/image (646).png>)

### Implementation details

At each optimization iteration, we randomly sample a batch of camera rays from the set of all pixels in the dataset, and then follow the hierarchical sampling to query Nc samples from the coarse network and Nc + Nf samples from the fine network. We then use the volume rendering procedure to render the color of each ray from both sets of samples. Our loss is simply the total squared error between the rendered and true pixel colors for both the coarse and fine renderings:

![](<../../.gitbook/assets/image (582).png>)

![](<../../.gitbook/assets/image (613).png>)

## Results

![](<../../.gitbook/assets/image (665).png>)

![](<../../.gitbook/assets/image (661).png>)
