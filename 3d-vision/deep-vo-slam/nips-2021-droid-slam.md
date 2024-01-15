---
description: 'DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras'
---

# \[NIPS 2021] DROID-SLAM

{% embed url="https://github.com/princeton-vl/DROID-SLAM" %}

{% embed url="https://arxiv.org/abs/2108.10869" %}

## Abstract

We introduce DROID-SLAM, a new deep learning based SLAM system. DROID-SLAM consists of recurrent iterative updates of camera pose and pixelwise depth through a Dense Bundle Adjustment layer. DROID-SLAM is accurate, achieving large improvements over prior work, and robust, suffering from substantially fewer catastrophic failures. Despite training on monocular video, it can leverage stereo or RGB-D video to achieve improved performance at test time.

## Introduction

SLAM问题可以从多个角度研究。早期的工作基于概率和滤波方法，对地图和相机位姿进行交替优化。近期的SLAM系统使用最小二乘优化，其影响准确性的关键因素在于BA，BA可以同时优化地图和相机位姿。

基于优化的SLAM系统容易迁移到其他传感器上。

当前的SLAM系统在真实世界的应用中缺乏鲁棒性，出现失败情况，如遭遇局部特征跟丢、优化发散和累积漂移。

本文提出了DROID-SLAM，一种新的基于深度学习的SLAM系统，它具有精度高、鲁棒性好、泛化性好（用单目进行模型训练，可以直接迁移到双目、RGB-D，无需重新训练）的优势。

DROID-SLAM的优异表现源自“Differentiable Recurrent Optimization-Inspired Design”，这是一种结合了传统方法和深度神经网络的端到端的可微结构。它包含循环迭代优化，基于RAFT光流，还引入了两个关键性创新。

首先，RAFT是迭代优化光流，而本文算法是迭代优化相机位姿和深度。RAFT需要输入两张图像，而本文可以在任意张图像上进行更新，实现对所有相机位姿和深度图的全局调整，这对于回环检测和减少长期累积误差很重要。

其次，在DROID-SLAM中，通过可微的Dense Bundle Adjustment（DBA）层来每次产生相机位姿和深度的更新，BDA层计算相机位姿和稠密像素级深度的Gauss-Newton更新，用光流的当前估计来最大化他们的相容性。DBA层能够根据几何约束，提升准确率和鲁棒性，并且使得DROID-SLAM无需重新训练就能迁移到双目和RGB-D输入。

## Approach

该算法以视频为输入，旨在同步估计相机运动的轨迹和构建环境的3D地图。

算法的输入是一组有序图像$${\{I_t\}}^N_{t=0}$$，对于每个图像t，保留两个状态变量：相机位姿$$G_t \in SE(3)$$和拟深度$$d_t \in R^{H \times W}_+$$。位姿和拟深度是未知的状态变量，当新的一帧输入后，对其进行迭代更新。本文的深度都是指拟深度​。

作者采用frame-graph $$(\mathcal{V}, \mathcal{E})$$来表示帧之间的共视关系。边$$(i,j) \in \mathcal{E}$$表示图$$I_i$$和图$$I_j$$之间有重叠视野，frame graph在训练和推理过程中动态构建，每次位姿和深度更新后，我们可以重新计算共视比例来更新frame graph。如果相机回到之前建图过的区域，在frame graph中加入长期连接来进行回环检测。

### Feature Extraction and Correlation

**Feature Extraction** 新输入的图像经过特征提取网络的处理，该网络包含6个残差模块和3个下采样层，得到1/8分辨率大小的稠密特征图。和RAFT一样，作者采用两个独立的网络：一个特征提取网络和一个内容网络。特征网络用于构建correlation volumes，而内容网络得到的特征用于每次更新处理。

**Correlation Pyramid** 对于frame graph中的每个边$$(i,j) \in \mathcal{E}$$​，计算所有特征向量对之间的点乘，得到4D correlation volume:

![](<../../.gitbook/assets/image (263).png>)

然后，我们对correlation volume的后两维进行average pooling，来获得一个4层的correlation pyramid。（对$$u_2,v_2$$以2为倍数进行pooling，得到不同大小的correlation volume）

**Correlation Lookup** 定义了一个lookup算子，来用一个半径为r的网格来检索correlation volume，

![](<../../.gitbook/assets/image (268).png>)

lookup算子以一个$$H \times W$$的坐标​网格为输入，根据双线性插值从correlation volume中检索值，对pyramid中的每个correlation volume进行查找，将每层的结构拼接在一起，得到最后的特征向量。

### Update Operator

![](<../../.gitbook/assets/image (293).png>)

本文SLAM系统的核心部分在于图2所示的可学习的更新算子。该算子是一个带有隐藏状态h的3x3卷积GRU。该算子每次更新隐藏状态，并得到位姿更新量$$\triangle \xi^{(k)}$$​和深度更新量$$\triangle d^{(k)}$$：

![](<../../.gitbook/assets/image (244).png>)

迭代更新产生了一组位姿和深度，寄希望于位姿和深度可以收敛到定点$$\{G^{(K)}\}\rightarrow G^*, \{d^{(k)}\}\rightarrow d^*$$.

**Correspondence** 在每次更新的开始，用当前位姿和深度的估计量来预测correspondence。给定第i帧的像素坐标网格，$$p_i \in R^{H \times W \times 2}$$，对frame graph中的每条边$$(i,j) \in \mathcal{E}$$计算稠密的correspondence field $$p_{ij}$$:

<div align="center">

<img src="../../.gitbook/assets/image (391) (2).png" alt="">

</div>

其中$$\Pi_c$$​为相机模型，将3D点投影到图像上，而$$\Pi^{-1}_c$$​是拟投影函数，将像素坐标$$p_i$$​和拟深度d投影到3D点云上。$$p_{ij}$$​表示像素坐标$$p_i$$投影到第j帧上的坐标。

**Inputs** 根据correspondence field 来检索correlation volumes。对于frame graph中的每条边$$(i,j) \in \mathcal{E}$$，用$$p_{ij}$$​从correlation volume $$C_{ij}$$中查找，检索correlation features。​此外，我们还用correspondence field来获得光流 $$p_{ij} - p_j$$.​

The correlation features provide information about visual similarity in the neighbourhood of $$p_{ij}$$allowing the network to learn to align visually similar image regions. However, correspondence is sometimes ambiguous. The flow provides an complementary source of information allowing the network to exploit smoothness in the motion fields to gain robustness.

**Update** 在输入GRU之前，correlation features和flow features分别通过两个卷积层进行处理，此外，作者还将context feature通过element-wise addition引入GRU。

ConvGRU是一个感受野很小的局部计算。作者在图像空间维度上对隐藏状态求平均来提取全局信息，并以此作为GRU的额外输入。全局信息对于SLAM很重要，因为不正确的correspondence会影响系统的精度。网络需要识别并剔除错误的correspondence。

GRU得到一个更新后的隐藏状态$$h^{(k+1)}$$​，作者没有直接预测位姿和深度的更新量，而是预测稠密光流的更新量。作者将隐藏状态输入两个额外的卷积层，得到（1）修改后的flow field $$r_{ij} \in R^{H \times W \times 2}$$​和（2）关联的可信度图$$w_{ij} \in R^{H \times W \times 2}_+$$​. $$r_{ij}$$​是网络预测的修正项，用于矫正稠密corrsepondence field中的误差。矫正后的correspondence记为$$p^*_{ij}=r_{ij}+p_{ij}$$​.

然后，作者对所有具有共享视野i的特征的隐藏状态进行池化，预测一个pixel-wise阻尼因子$$\lambda$$​，用softplus来确保阻尼因子是正值。此外，还用池化后的特征预测一个8x8的mask，用于对拟深度估计值进行上采样。

**Dense Bundle Adjustment Layer (DBA)** DBA将光流映射到一个位姿和pixel-wise深度更新量。在整个frame graph上定义损失函数：

![](<../../.gitbook/assets/image (416).png>)

其$$||\cdot||_\Sigma$$为Mahalanobis距离，根据可信度权重$$w_{ij}$$​对误差项加权。我们希望得到一个更新后的位姿$$G'$$​和深度$$d'$$，据此得到的重投影点符合修正后的correspondence​ $$p^*_{ij}$$.​

作者用局部参数化去线性化上式，并用Gauss-Newton法去求解更新量$$(\triangle \xi, \triangle d)$$​，由于上式中的每一项只包含一个单一的深度变量，Hessian矩阵具有块对角阵的结构。Separating pose and depth variables, the system can be solved efficiently using the Schur complement with the pixelwise damping factor $$\lambda$$ added to the depth block:

![](<../../.gitbook/assets/image (442).png>)

其中对角阵C是对角阵，$$C^{-1}=1/C$$.​ The DBA layer is implemented as part of the computation graph and backpropogation is performed through the layer during training.

### Training&#x20;

SLAM系统用PyTorch实现，并用LieTorch拓展库来在所有群元素的正切空间中进行反向传播，

**Removing gauge freedom** 在单目系统中，网络只能将相机轨迹恢复到相似变换。一个解决办法是定义一个对相似变换具有不变性的loss。但是，gauge-freedom在训练过程中依然存在，并影响了线性系统，影响梯度稳定性。作者通过在每个训练序列中，将前两个位姿固定为真值位姿来解决这一问题。固定第一个位姿解决了6-dof gauge freedom，固定第二个位姿解决了尺度不一致问题。

**Constructing training video** 每个训练样本包含7帧。对每个长度为$$N_i$$​的视频i，提前计算每两帧之间的平均光流幅度吗，得到一个$$N_i \times N_i$$​的距离矩阵。重叠部分小于50%的帧之间的距离被设为无穷，在训练过程中，作者根据距离矩阵动态采样训练样本，保证相邻帧之间的平均光流在8像素到96像素之间。

**Supervision** 网络用位姿损失和光流损失来监督。根据估计的深度和位姿计算光流，根据真实位姿和深度计算真值光流，光流损失为两光流间的L2距离。

给定一组真值位姿$${\{T\}}^N_i$$​和预测位姿$${\{G\}}^N_i$$​，位姿损失为两位姿间的距离：

![](<../../.gitbook/assets/image (398).png>)

We apply the losses to the output of every iteration with exponentially increasing weight using $$\gamma=0.9$$​.

### SLAM System

本文系统包含两个异步运行的线程。前端线程输入新的帧，提取特征，选择关键帧，进行局部优化。后端线程对全部关键帧进行全局优化。

**Initialization** 当输入12帧后，进行初始化。在累积帧时，只有当光流大于16像素（只进行一次更新）时才保留前一帧。当累积够12帧后，初始化一个frame graph，构建3步以内关键帧之间的边，然后进行10次更新。

**Frontend** 前端保留了一组关键帧和一个frame graph。关键帧的位姿和深度都经过优化。先从输入的帧中提取特征，然后根据平均光流得到3个最近邻，在frame graph中添加边。位姿用线性移动模块来初始化，然后进行多次更新来更新关键帧的位姿和深度。固定前两帧的位姿来避免gauge freedom，对深度不加限制。

当新的一帧被跟踪，要删除一个关键帧。计算帧间的平均光流幅度，删除冗余的帧。如果没有适合删除的帧，就删除最旧的关键帧。

**Backend** 后端对所有关键帧进行优化。在每次迭代后，根据每对关键帧间的光流来重构frame graph，得到NxN距离矩阵。首先对相邻帧之间加上边，然后根据光流的增加顺序从距离矩阵中采样新边。对每个边，作者限制相邻边之间的距离要小于2，边的距离采用Chebyshev距离：

![](<../../.gitbook/assets/image (390).png>)

然后对整个frame graph进行更新。

We only perform full bundle adjustment on keyframe images. In order to recover the poses of non-keyframes, we perform motion-only bundle adjustment by iteratively estimating flow between each keyframe and its neighboring non-keyframes. During testing, we evaluate on the full camera trajectory, not just keyframes

**Stereo and RGB-D** 对于RGB-D相机，将深度视为一个变量，在DBA的损失函数中加入关于深度的loss，即真值深度和预测深度间的均方差；对于双目相机，使用完全相同的系统，只是帧多了一倍，并且在DBA层中固定左右目相机间的相对位姿，frame graph中的双目相机之间的边能够有效利用双目信息。

## Experiments

Our network is trained entirely on monocular video from the synthetic TartanAir dataset. We train our network for 250k steps with a batch size of 4, resolution 384 x 512, and 7 frame clips, and unroll 15 update iterations. Training takes 1 week on 4 RTX-3090 GPUs.

![](<../../.gitbook/assets/image (411).png>)

![](<../../.gitbook/assets/image (400).png>)

![](<../../.gitbook/assets/image (388).png>)

![](<../../.gitbook/assets/image (419).png>)

![](<../../.gitbook/assets/image (417).png>)

![\\](<../../.gitbook/assets/image (387).png>)

![](<../../.gitbook/assets/image (460).png>)

![](<../../.gitbook/assets/image (437).png>)

![](<../../.gitbook/assets/image (415).png>)

**Timing and Memory** Our system can run in real-time with 2 3090 GPUs. Tracking and local BA is run on the first GPU, while global BA and loop closure is run on the second. On EuRoC, we average 20fps (camera hz) by downsampling to 320 x 512 resolution and skipping every other frame. Results in Tab. 4 were obtained in this setting. On TUM-RGBD, we average 30fps by downsampling to 240 x 320 and skipping every other frame, again the reported results where obtained in this setting. On TartanAir, due to much faster camera motion, we are unable to run in real-time, averaging 8fps. However, this is still a 16x speedup over the top 2 submissions to the TartanAir SLAM challenge, which rely on COLMAP.

The SLAM frontend can be run on GPUs with 8GB of memory. The backend, which requires storing feature maps from the full set of images, is more memory intensive. All results on TUM-RGBD can be produced on a single 1080Ti graphics card. Results on EuRoC, TartanAir and ETH-3D (where video can be up to 5000 frames) requires a GPU with 24GB memory. While memory and resource requirements are currently the biggest limitation of our system, we believe these can be drastically reduced by culling redundant computation and more efficient representations.

## Appendix&#x20;

### Camera Model and Jacobians

![](<../../.gitbook/assets/image (468).png>)

![](<../../.gitbook/assets/image (402).png>)

### Network Architecture

![](<../../.gitbook/assets/image (452).png>)

![](<../../.gitbook/assets/image (389).png>)
