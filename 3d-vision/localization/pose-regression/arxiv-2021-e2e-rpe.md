---
description: >-
  End-to-end learning of keypoint detection and matching for relative pose
  estimation
---

# \[arxiv 2021] E2E RPE

## Abstract

We propose a new method for estimating the relative pose between two images, where we jointly learn keypoint detection, description extraction, matching and robust pose estimation. While our architecture follows the traditional pipeline for pose estimation from geometric computer vision, all steps are learnt in an end-to-end fashion, including feature matching.

## Introduction

本文的贡献总结为：

1. End-to-end learning of all the traditional four steps of relative pose estimation （关键点检测、描述子计算、特征匹配和位姿估计）；
2. A novel trainable layer for feature matching；
3. A self-supervised data augmentation method to generate ground-truth matches from RGB-D images

## Proposed Method

### Architecture

本文所设计的结构旨在根据reference图像$$I_r$$​（已知位姿），恢复query图像的位姿$$I_q$$​。这一任务一般包含四个步骤：关键点检测、描述子计算、特征匹配和位姿估计。作者提出一种CNN结构来实现端到端的相对位姿估计，显示地实现以上四个步骤。

![](<../../../.gitbook/assets/image (383).png>)

利用SuperPoint提取两幅图像的特征，用correlation层来融合两组描述子。根据correlation层的输出，matching层能够得到点的correspondence，用于位姿估计。

#### Correlation layer

利用SuperPoint提取query图像和reference图像的特征，记为$$(K_q,D_q)$$​和$$(K_r,D_r)$$，计算两幅图描述子间的点乘，并用keypoint分数来加权$$K'(i,j)=1-K(i,j,65)$$，得到4D张量：

![](<../../../.gitbook/assets/image (436).png>)

#### Matching layer

作者希望通过一个CNN来学习4D张量$$C_{q,r}(i,j,i',j')$$​的结构，从而获得较好的匹配。因此matching层要通过一组卷积核对4D correspondence map进行处理，获得稀疏的map，即在构成匹配的(i,j),(i',j')对应的(i,j,i',j')处产生峰值。这需要全局信息，而局部描述子匹配只提供了局部信息。

![](<../../../.gitbook/assets/image (427).png>)

为了高效计算，将4D张量分解为3D张量，即将(i',j')用2D Hilbert's curve H转换为一个单一的索引值k，来更好地保留空间局部性。这样，作者就可以在大小为(H,W,HxW)的3D张量进行3D卷积计算，避免了计算量较大的4D卷积。对于没有匹配的点，在matching层的输出M中加入一个额外的dustbin通道来处理。因此，M的大小为（H,W,HxW+1）。

![](<../../../.gitbook/assets/image (405).png>)

在matching层中用V-Net结构，来保留细节信息，提升表现。

3D tensor中的峰值代表了关键点和匹配。为了实现端到端训练，需要将峰值转化为点坐标间的correpondence (kp(i,j),kp'(i,j))。这一步用softargmax来实现：

![](<../../../.gitbook/assets/image (393).png>)

其中K是SuperPoint提取出的keypoint map K，大小为(H/8 x W/8 \* 65).

![](<../../../.gitbook/assets/image (385).png>)

### Losses

在DSAC中，作者根据Weighted Direct Linear Transform出发设计了loss，包含三个部分：由真值相对位姿$$(R_{r,q}, T_{r,q})$$决定的$$\tilde{e}$$，由correspondence $$q_{i,j}=(kp(i,j),kp'_n(i,j))$$决定的X和每个correspondence对应的权重​。作者在本文中也使用了这个loss，将correspondence weight定义为$$M'(i,j)=1-M(i,j,H\times W+1)$$，并去掉正则项。

![](<../../../.gitbook/assets/image (441).png>)

为了避免权重全部收敛到0导致的trivial solution，作者用sigmoid函数$$\sigma$$​来获得可微形式的内点数。假设图像的深度和全局位姿已知，那么可以计算reference图像中点kp'(i,j)重投影到query图像中的位置。令$$C_{calib}$$为相机内参矩阵，$$P_{r,q}=(R_{r,q}|T_{r,q})$$为图像间的相对位姿。​重投影可以写作：

$$
\mathcal{R}(kp'(i,j)=C_{calib}P_{r,q}kp'_n=C_{calib}P_{r,q}\frac{C^{-1}kp'(i,j)}{||C^{-1}kp'(i,j)||}{depth}_r(i,j)
$$

![](<../../../.gitbook/assets/image (421).png>)

Eventually, we noticed that without any additional constraint keypoints tend to converge to a dense solution where there is a keypoint in each patch centered in the middle of the patch even if it means losing some accuracy in pose. To overcome that we added a unitary term that enforce keypoints to be close to the keypoint position detected by Superpoint. 所以作者在两张图上计算keypoint的cross-entropy loss，以SuperPoint检测出的关键点为真值，这一loss记为$$L_{keypoint}$$.

最后的loss为：

![](<../../../.gitbook/assets/image (459).png>)

其中$$\alpha=\beta=2, \kappa=0.1,\tau=16.$$

### Training

#### Paris generation

​使用Standard 2D-3D Semantic dataset进行训练，共70496张图像，46575张（area2,4,5）用于训练，其他的用于测试。作者提出了一个寻找图像对的方法：对每张训练图像$$I_t$$​，用在Pitts250k数据集上预训练过的NetVLAD寻找64张图像$$I_r$$，首先根据全局位姿提出与$$I_t$$不一致的图像（相对距离超过20m且没有视野重叠）​，最后删除靠的太近的图像（$$||T_{t,r}|| \le 0.5m,\angle R_{t,r} \le 5 \degree$$）.

#### Scene adaptation

Instead of using Superpoint’s keypoints as ground-truth for the keypoint unitary loss, we propose to aggregate points from reprojections of different viewpoints of the scene using ground-truth poses and depth rather than through random homography.

In practise we use the same generated training pairs for the different viewpoints. However it can happen that a keypoint visible in one image is not visible in the other because of occlusion. We discard those situations by comparing the depth of the reprojected keypoint depth with its z-value before reprojection.

these new keypoints are iteratively updated during training. It means that every 20 epochs, we recompute the keypoints ground-truth used in $$L_{keypoint}$$ by aggregating across viewpoints the keypoints extracted using the last learned state of the network.

#### Implementation details

首先固定特征提取部分的权重，只训练matching layer，然后再端到端训练。

### Pose refinement

To use our network on the problem of localising a query image in a database of images with known pose, as in most retrieval methods based on keypoints, we first find the closest N = 16 putative database images to the query using NetVLAD. We then predict the pose between the query and each putative image using our network, and keep the pose $$P_{max}$$with most inliers.

At this stage, we perform a further refinement step, where we add correspondences from the putative images whose predicted pose is not too far from $$P_{max}$$(i.e. $$||T_r - T_{max}|| \le 1m$$). Last, we recompute the pose on this larger set of matches using P3P-RANSAC and refine it by non-linear optimization on the inliers using Levenberg- Marquardt. This step improves the accuracy of our method, and has a low computational cost because our network extracts few keypoints with a high inlier ratio on which RANSAC is very efficient.

## Experiments

![](<../../../.gitbook/assets/image (414).png>)

![](<../../../.gitbook/assets/image (463).png>)
