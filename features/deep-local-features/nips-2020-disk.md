---
description: 'DISK: Learning local features with policy gradient'
---

# \[NIPS 2020] DISK

{% embed url="https://arxiv.org/abs/2006.13566" %}

{% embed url="https://github.com/cvlab-epfl/disk" %}

## Introduction

Moreover, the "quality" of a given feature depends on the rest, because a feature that is very similar to others is less distinctive, and therefore less useful. This is hard to account for during training.

Our backbone is a network that takes images as input and outputs keypoint 'heatmaps' and dense descriptors. Discrete keypoints are sampled from the heatmap, and the descriptors at those locations are used to build a distribution over feature matches across images. We then use geometric ground truth to assign positive or negative rewards to each match, and perform gradient descent to maximize the expected reward E P r negative rewards to each match, and perform gradient descent to maximize the expected reward $$\mathbb{E} \sum_{(i,j)\in M_{A\leftrightarrow B}}r(i\leftrightarrow j)$$, where $$M_{A\leftrightarrow B}$$ is the set of matches and r is per-match reward. In effect, this is a policy gradient method.

## Method

给定图A和图B，本文算法旨在提取一组局部特征$$F_A$$和$$F_B$$，匹配获得$$M_{A\leftrightarrow b}$$。记根据特征算法$$\theta_F$$从图$$I$$中提取出特征$$F_I$$的概率为$$P(F_I|I,\theta_F)$$，图A和图B之间的匹配分布为$$P(M_{A \leftrightarrow B}|F_A, F_B,\theta_M)$$，$$\theta_M$$为匹配算法的参数。

### Feature Distribution

网络结构采用U-Net，输出N+1维特征图，其中一维用于检测关键点，N维用于提取描述子，分别记为检测热力图K和描述子特征图D。本文中N=128.

K被划分为$$h \times h$$大小的单元格，每个单元格内最多提取一个特征。记该单元格$$u$$的检测热力图为$$K^u$$。

则单元格$$u$$内像素$$p$$被采样的概率为$$P_s(p|K^u) = softmax(K^u)$$，该值度量了在单元格内某像素的相对偏好，在inference时，softmax被替换为argmax；

但是被单元格内可能不存在关键点，所以定义像素的被接受概率为$$P_a({accept}_p|K^u)=sigmoid(K^u_p)$$，该值度量了像素点的绝对质量，在inference时，sigmoid被替换为sign。

因此，像素点p作为特征点的概率为$$P(p|K^u)=P_s(p|K^u) \cdot P_a({accept}_p|K^u)$$。根据采样概率，可以提取一系列关键点，从D中检索其对应描述子，进行L2正则化，得到图像的特征$$F_I=\{(p_1,D(p_1)),(p_2,D(p_2)),...\}$$.

### Match Distribution

当获得两图的特征$$F_A$$和$$F_B$$，可以计算描述子间的L2距离，获得距离矩阵d。作者选择放宽cycle-consistent matching策略，根据d的行分布获得从A到B的匹配，根据d的列分布获得从B到A的匹配，只有当两特征是相互匹配的，才接受这对匹配。

记从A到B的匹配为$$P_{A\rightarrow B}(j|d,i)={softmax(-\theta_M d(i,\cdot))}_j$$，其中$$\theta_M$$是匹配算法的参数。从B到A的匹配可以由$$d^T$$获得。

因此，任意一对匹配被接受的概率为$$P(i \leftrightarrow j)=P_{A\rightarrow B}(i|d,j) \cdot P_{B\rightarrow A}(j|d,i)$$。

如果当给定$$F_A$$,$$F_B$$时，回报可以分解到各个匹配上，那么就匹配过程就不会影响到梯度的分布，我们就可以获得确切的回报关于特征的提取，而不需要进行排序。这有助于网络的稳定收敛。

### Reward Function

作者采用了较为简单的回报策略。当两像素点的图像位姿和深度已知，那么可以获得点到点的对应关系，如果两点位于投影点的邻域内，则认为该匹配是正确的，赋予正值回报（奖励）$$\lambda_{tp}$$；如果两像素点的深度未知，那么只能获得点到极线的对应关系，如果点到极线的距离小于阈值，认为该匹配是未知的，既不奖励也不惩罚；对于其他情况，赋予负值回报（惩罚）$$\lambda_{fp}$$.

### Gradient Estimator

![](<../../.gitbook/assets/image (258).png>)

需要注意的是，DISK只是通过对参与匹配的特征质量进行监督，没有对特征网络进行任何监督，因此如果一个没有被匹配到的特征点被定义为中立的。This is a very useful property because keypoints may not be co-visible across two images, and should not be penalized for it as long as they do not create incorrect associations. 但是另一方面网络可能会提取过多无法匹配的特征，这会对特征匹配造成额外的负担。为此，作者对每个采样的关键点赋予了一个额外的小的惩罚$$\lambda_{kp}$$，可以视为加入一个正则项。

### Inference

当网络训练好后，用NMS代替单元格的检测策略。

## Experiments

DISK使用MegaDepth数据集进行训练。在训练时使用具有重叠视野的A,B,C三张图像进行训练，分别计算梯度并累加起来反向传播。匹配算法的参数$$\theta_M$$不加入训练，随训练过程逐渐增大。设置$$h$$为8，$$\lambda_{tp}=1$$,$$\lambda_{fp}=-0.25$$,$$\lambda_{kp}=-0.001$$.Since a randomly initialized network tends to generate very poor matches, the quality of keypoints is negative on average at first, and the network would cease to sample them at all, reaching a local maximum reward of 0. To avoid that, we anneal $$\lambda_{fp}$$ and $$\lambda_{kp}$$ over the first 5 epochs, starting with 0 and linearly increasing to their full value at the end.

### Evaluation on the Image Matching Challenge (IMC)

![](<../../.gitbook/assets/image (231).png>)

![](<../../.gitbook/assets/image (274).png>)

![](<../../.gitbook/assets/image (262).png>)

### Evaluation on HPatches

![](<../../.gitbook/assets/image (240).png>)

### Evaluation on the ETH-COLMAP Benchmark

![](<../../.gitbook/assets/image (254).png>)
