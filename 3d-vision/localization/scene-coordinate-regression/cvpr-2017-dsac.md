---
description: DSAC - Differentiable RANSAC for Camera Localization
---

# \[CVPR 2017] DSAC

{% embed url="https://arxiv.org/abs/1611.05705" %}

{% embed url="https://github.com/cvlab-dresden/DSAC" %}

{% embed url="https://github.com/vislearn/DSACLine" %}

## Abstract

In this work, we present two different ways to overcome this limitation. The most promising approach is inspired by reinforcement learning, namely to replace the deterministic hypothesis selection by a probabilistic selection for which we can derive the expected loss w.r.t. to all learnable parameters.

## Introduction

RANSAC类的算法遵循着一个策略，即Local predictions (e.g. feature matches) induce a global model (e.g. a homography).

In this work, we are interested in learning components of a computer vision pipeline that follows the principle: predict locally, fit globally. As explained earlier, RANSAC is an integral component of this wide-spread strategy. We ask the question, whether we can train such a pipeline end-toend. More specifically, we want to learn parameters of a convolutional neural network (CNN) such that models, fit robustly to its predictions via RANSAC, minimize a task specific loss function.

RANSAC先从一小部分随机选取的数据中估计模型，然后用所有数据的模型契合度来作为模型的分数。最后选取契合度分数最高的模型作为最后的输出。但是，选择模型这一过程是不可微的，无法用端到端的训练方法来优化。

作者先尝试将RANSAC中的argmax替换成soft argmax，这一方法其实是令网络学习到各假设模型的均值，而非选取一个最好的模型，容易过拟合。因此，作者选择保留了hard模型选择的过程，但是用概率过程来实现它，即DSAC。DSAC可以计算expected loss相对各网络参数的梯度。

To demonstrate the principle, we choose the problem of camera localization: From a single RGB image in a known static scene, we estimate the 6D camera pose (3D translation and 3D rotation) relative to the scene. We demonstrate an end-to-end trainable solution for this problem, building on the scene coordinate regression forest (SCoRF) approach. The original SCoRF approach uses a regression forest to predict the 3D location of each pixel in an observed image in terms of ‘scene coordinates’. A hypothesize-verify-refine RANSAC loop then randomly select scene coordinates of four pixel locations to generate an initial set of camera pose hypotheses, which is then iteratively pruned and refined until a single high-quality pose estimate remains. In contrast to previous SCoRF approaches, we adopt two CNNs for predicting scene coordinates and for scoring hypotheses. More importantly, the key novelty of this work is to replace RANSAC by our new, differentiable DSAC.

## Method

### Background

基本的RANSAC算法包含四个步骤：1.从数据中采样出一个子集，根据子集数据来估计模型；2.根据一些一致性度量标准（如内点数）来给模型打分；3.选择最好的假设模型；4.用额外的数据来优化所选的模型。步骤4是可选的，在实际中能够提升准确率。

令$$I$$​为图像，像素点索引为$$i$$。希望可以构建解释$$I$$​的模型，模型参数为$$\tilde{h}$$​。在定位问题中，这是6DoF相机位姿，即相机相对场景坐标系的3D旋转和3D平移。作者不直接从图像数据$$I$$拟合模型$$\tilde{h}$$，而是利用间接的、带噪声的、对每个像素预测的2D-3D匹配$$Y(I)=\{y(I,i)|\forall i\}$$，$$y_i=y(I,i)$$是像素i的场景坐标。为了从Y中估计出$$\tilde{h}$$​，作者采用RANSAC

**1.Generate a pool of hypotheses.** 从匹配的子集中估计模型。每个子集包含能够求解问题的最少数量的匹配。记子集为$$Y_J$$，J为匹配子集的索引$$J=\{j_1,...,j_n\}$$。为了获得这一集合，均匀地采集匹配索引$$j_m \in [1,...,|Y|]$$来获得$$Y_J:=\{y_{j_1},...,y_{j_n}\}$$。从$$Y_J$$​中利用函数H来估计模型$$h_J=H(Y_J)$$。在本算法中，H为PNP算法，n=4.

**2.Score hypotheses.** $$s(h_J,Y)$$​度量了模型$$h_J$$​的一致性/质量。定义场景坐标$$y_i$$​的重投影误差：

![](<../../../.gitbook/assets/image (397).png>)

其中$$p_i$$是像素i的2D位置，C是相机投影矩阵。当$$e_i<\tau$$时，$$y_i$$是一个内点。

**3.Select best hypothesis.**&#x20;

![](<../../../.gitbook/assets/image (598).png>)

**4.Refine hypothesis.** 用函数$$R(h_{AM},Y)$$来优化$$h_{AM}$$​。用全部匹配Y来进行优化。常规做法是在Y中挑选所有的内点，然后在内点上重新计算H。优化后的位姿是算法的输出：$$\tilde{h}_{AM}=R(h_{AM},Y)$$

### Learning in a RANSAC Pipeline

本文旨在同时学习场景坐标预测函数$$y(I,i;w)$$和评估函数$$s(h_J,Y;v)$$，并将其与RANSAC框架结合，实现端到端训练参数w和v。其中w影响所产生的位姿的质量，v影响着模型选择过程。$$Y^w$$​表示场景坐标预测受参数w影响，$$h^{w,v}_{AM}$$​表示所选的模型受参数w和v影响。优化的目标是寻找参数w和v，让训练图像集$$\mathcal{I}$$上优化后的模型的损失$$\mathcal{l}$$最小：

![](<../../../.gitbook/assets/image (606).png>)

其中h\*是图像I的真值模型参数。为了端到端训练，需要设计可微的损失函数l和可微的优化函数R。

![](<../../../.gitbook/assets/image (641).png>)

#### Soft argmax Selection (SoftAM）

用soft argmax来替换公式2中的argmax：

![](<../../../.gitbook/assets/image (624).png>)

![](<../../../.gitbook/assets/image (575).png>)

#### Probabilistic Selection (DSAC)

用概率化选择方法来替换公式2中的规则化选择方法，即根据概率来选择模型：

![](<../../../.gitbook/assets/image (596).png>)

求中P(J|v,w)是公式5中$$s(h^w_j,Y^w;v)$$估计分数的softmax分布。这一方法来源于强化学习的策略梯度方法。能够通过最小化随机过程（公式6）的loss的期望来学习参数w和v：

![](<../../../.gitbook/assets/image (605).png>)

## Differentiable Camera Localization

Brachmann et al. use an auto-context random forest to predict multi-modal scene coordinate distributions per image patch. After that, minimal sets of four scene coordinates are randomly sampled and the PNP algorithm is applied to create a pool of camera pose hypotheses. A preemptive RANSAC schema iteratively refines, re-scores and rejects hypotheses until only one remains. The preemptive RANSAC scores hypotheses by counting inlier scene coordinates, i.e. scene coordinates yi for which reprojection error $$e_i<\tau$$ . In a last step, the final, remaining hypothesis is further optimized using the uncertainty of the scene coordinate distributions.

本文所提出算法与Brachmann et al.算法的区别在于：

1.用CNN（称为Coordinate CNN）来预测场景坐标。对于每42x42大小的图像块，预测一个场景坐标。用13层的VGG（模型参数量33M）。为了减少测试时间，每张图只处理40x40个图像块。

2.用CNN评估模型（称为Score CNN），该CNN根据重投影误差来评估模型的一致性。对于某个场景坐标$$y_i$$，由公式1计算模型$$h_J$$​的重投影误差，得到一个40x40大小的重投影误差图像，然后输入Score CNN（一个13层的VGG，模型参数量为6M）。

3.不进行迭代的RANSAC，只使用SoftAM或DSAC评估一次模型并选择最后的位姿。

4.只优化最后的位姿。选择最多100个内点坐标（重投影误差小于阈值$$\tau$$），用PNP求解。重复多次。

采样256个模型，进行8次优化，内点阈值$$\tau=10px$$.

## Experiments

在训练中，使用如下损失函数：

![](<../../../.gitbook/assets/image (666).png>)

其中$$h={(\theta,t)}^{-1}$$，$$\theta$$​是角轴，t为相机平移。

### Componentwise Training

**Coordinate CNN**采用损失函数：$$l_{coord}(y,y^*)=||y-y^*||$$.每个batch有64个样本，用Adam优化器，学习率为$$10^{-4}$$，每50k步学习率减少一半，总共训300k代。

**Score CNN**采用损失函数：$$l_{score}(s,s^*)=|s-s^*|$$，其中$$s^*=-\beta l_{pose}(h,h^*)$$.$$\beta=10$$.每个batch有64个样本，用Adam优化器，总共训练2k代。

### End-to-End Training

![](<../../../.gitbook/assets/image (570).png>)

从零开始端到端训练容易收敛到局部最优。因此，作者先用componentwise training来初始化Coordinate CNN和Score CNN.

![](<../../../.gitbook/assets/image (644).png>)

### Results

![](<../../../.gitbook/assets/image (618).png>)

![](<../../../.gitbook/assets/image (647).png>)

![](<../../../.gitbook/assets/image (635).png>)

![](<../../../.gitbook/assets/image (634).png>)

**Restoring the argmax Selection.** After end-to-end training, one may restore the original RANSAC algorithm, e.g. selecting hypotheses w.r.t. scores via argmax. In this case, the average accuracy of DSAC stays at 62.4%, while the accuracy of SoftAM decreases to 57.2%.&#x20;

**Test Time.** The scene coordinate prediction takes ∼0.5s on a Tesla K80 GPU. Pose optimization takes ∼1s. The runtime of argmax hypothesis selection (RANSAC) or probabilistic selection (DSAC) is identical and negligible.
