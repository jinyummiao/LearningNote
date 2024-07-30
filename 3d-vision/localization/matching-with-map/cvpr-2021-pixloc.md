---
description: 'Back to the Feature: Learning Robust Camera Localization from Pixels to Pose'
---

# \[CVPR 2021] PixLoc

{% embed url="https://openaccess.thecvf.com/content/CVPR2021/papers/Sarlin_Back_to_the_Feature_Learning_Robust_Camera_Localization_From_Pixels_CVPR_2021_paper.pdf" %}

{% embed url="https://github.com/cvg/pixloc" %}

### Abstract

> In this paper, we go Back to the Feature: we argue that deep networks should focus on learning robust and invariant visual features, while the geometric estimation should be left to principled algorithms. We introduce PixLoc, a scene- agnostic neural network that estimates an accurate 6-DoF pose from an image and a 3D model. Our approach is based on the direct alignment of multiscale deep features, casting camera localization as metric learning. PixLoc learns strong data priors by end-to-end training from pixels to pose and ex- hibits exceptional generalization to new scenes by separating model parameters and scene geometry. The system can local- ize in large environments given coarse pose priors but also improve the accuracy of sparse feature matching by jointly refining keypoints and poses with little overhead.

### Introduction

现有的表现较好的定位系统依赖于2D-3D特征匹配，一般通过在RANSAC循环中使用PnP来求解位姿。为了获得更好的匹配表现，很多算法使用CNN来提取局部特征。但是用端到端的方法来实现这一类pipeline比较复杂，因为梯度不好计算（不可微）。因此有人直接从图像回归相机位姿或像素点的三维坐标。然而这一类方法是场景特定的，在新场景中必须重新训练。此外，绝对位姿或相对位姿估计的准确率不高，缺乏泛化能力。虽然回归相对位姿是和场景无关的，但是据作者所知，现有此类算法无法在保持精度的同时获得对不同场景的泛化能力。

作者认为应当利用图像的先验位姿和场景的3D模型（这些信息一般是可以通过对场景进行重建获得）来进行端到端的位姿回归。并且认为：与其致力于让模型学习基本的几何关系或编码3D地图，还不如让模型根据人们已经很熟悉的几何原理，学习对外观和结构变化的鲁棒性，利用表征学习来解决端到端的定位任务。

![](<../../../.gitbook/assets/image (2) (1) (1) (1).png>)

> In this paper, we introduce a trainable algorithm, PixLoc, that localizes an image by aligning it to an explicit 3D model of the scene based on dense features extracted by a CNN (Figure 1). By relying on classical geometric optimization, the network does not need to learn pose regression itself, but only to extract suitable features, making the algorithm accurate and scene-agnostic. We train PixLoc end-to-end, from pixels to pose, by unrolling the direct alignment and supervising only the pose. Given an initial pose obtained by image retrieval, our formulation results in a simple localization pipeline competitive with complex state-of-the-art approaches, even when the latter are trained specifically per scene. PixLoc can also refine poses estimated by any existing approach as a lightweight post-processing step

### Method

![](<../../../.gitbook/assets/image (302).png>)

**Overview** PixLoc根据已知的场景3D结构来对其query和reference图像，进而定位图像。对齐过程通过对齐由CNN提取的特征图来实现。CNN和优化参数利用真值位姿来端到端训练。

**Motivation** 在图像的绝对位姿和相对位姿任务中，网络学习：1）识别图像在场景中的大致位置，2）识别场景中特有的特征，3）回归如位姿和坐标等准确的几何属性。由于CNN可以学习到具备外观和几何泛化性的特征，1）和2）不需要和场景相关，并且1）已经可以通过图像检索来实现。另一方面，3）可以根据3D模型进行特征匹配和图像对齐来实现。因此，应当学习到鲁棒和通用的特征，让位姿估计和场景无关，与几何学紧密结合。这一问题的难点在于如何定义对定位有利的特征。作者通过可微化几何估计来解决这一问题，只监督最后估计的位姿。

不同于位姿回归和位置回归任务，本文假设场景的3D模型是已知的。

**Problem formulation** 本文旨在预测$$I_q$$的6DOF位姿$$(R,t)\in SE(3)$$，其中R是旋转矩阵，t是平移向量。给定场景的3D模型，如一个稀疏或稠密的3D点云$$\{P_i\}$$或标有位姿的图像$$\{I_k\}$$，记为参考数据。

#### Localization as image alignment

**Image represetation** 对于query图像$$I_q$$和reference图像$$I_k$$，用CNN在第$$l \in \{L, ...,1\}$$层提取$$D_l$$维特征图$$F^l \in R^{W_l \times H_l \times D_l}$$.这些特征图在通道维度进行L2正则化，来提升特征的鲁棒性和泛化性。

**Direct alignment** The goal of the geometric optimization is to find the pose $$(R,t)$$ which minimizes the difference in appearance between the query image and each reference image. 对于一个给定的特征层$$l$$和在reference图像$$k$$中可观测到的3D点$$i$$，可定义残差：

![](<../../../.gitbook/assets/image (189).png>)

其中<img src="../../../.gitbook/assets/image (140).png" alt="" data-size="original">是给定当前估计位姿时$$i$$的投影点，\[\*]为插值索引。N个观测上的总误差为：

![](<../../../.gitbook/assets/image (1034).png>)

其中$$\rho$$为鲁棒的损失函数，其导数为$$\rho'$$，$$w^i_k$$为权重。使用Levenberg-Marquardt (LM)算法从初始估计$$(R_0,t_0)$$开始迭代最小化这个非线性最小二乘误差。

为了使训练平稳，作者选择递进地优化每个特征层，从最粗略的特征层$$l=1$$开始，用前一层的结果初始化后一层。低分辨率的特征图提升了位姿估计的鲁棒性，更精细的特征用于增强准确率。每次位姿更新量$$\delta \in R^6$$用其李代数表示到SE(3)。所有残差被拼接为$$r \in R^{ND}$$，所有权重写为$$W={diag}_{i,k}(w^i_k \rho')$$，并定义Jacobian和Hessian矩阵为：

![](<../../../.gitbook/assets/image (503).png>)

由衰减Hessian矩阵和求解线性系统来计算更新：

![](<../../../.gitbook/assets/image (304).png>)

其中$$\lambda$$是衰减系数，在Gauss-Newton法($$\lambda=1$$)和梯度下降法($$\lambda \rightarrow \infin$$)之间进行插值，在每一次迭代中根据不同的启发式方法进行调整。最后，新的位姿根据左乘获得：

![](<../../../.gitbook/assets/image (835).png>)

当更新量足够小时，停止优化。

**Infusing visual priors** 以上步骤与经典的像素对齐是一样的。但是CNN能够学到复杂的视觉先验，因此作者希望能够赋予它向正确位姿引导优化的能力。为此，CNN预测了在每个特征层外，还预测了一个不确定性map $$U^l_k \in R^{W_l \times H_l}_{>0}$$。query图像和reference图像的pointwise不确定性用于生成权重：

![](<../../../.gitbook/assets/image (55).png>)

The weight is 1 if the 3D point projects into a location with low uncertainty in both the query and the reference images. It tends to 0 as either of the location is uncertain. 此处权重没有被显性地监督，但是学习去提升估计位姿的准确性。

![](<../../../.gitbook/assets/image (152).png>)

![](<../../../.gitbook/assets/image (685).png>)

**Fitting the optimizer to the data** LM算法需要选择鲁棒损失函数$$\rho$$和衰减系数$$\lambda$$，充满了技巧性。过去的工作用神经网络来根据残差和视觉特征来预测$$\rho'$$和$$\lambda$$，甚至位姿更新量。作者认为这会损伤模型对新数据分布的泛化能力，因为它会让优化器与训练数据的视觉语义信息相关联。作者希望优化器想关于姿态或残差的分布，而不是它们的语义内容。为此，作者提出令$$\lambda$$为固定的模型参数，用梯度下降法进行优化。

Importantly, we learn a different factor for each of the 6 pose parameters and for each feature level, replacing the scalar λ by $${\lambda}_l \in R^6$$, parametrized by $$\theta_l$$ as

![](<../../../.gitbook/assets/image (561).png>)

这将调整训练过程中各个姿势参数的曲率，根据数据直接学习运动先验。For example, when the camera is mounted on a car or a robot that is mostly upright, we expect the damping for the in-plane rotation to be large. In contrast, common heuristics treat all pose parameters equally and do not permit a per-parameter damping.

#### Learning from poses

PixLoc可以泛化于任何已知的3D结构，如点云、RGBD深度图、雷达数据等。

**Training** 梯度通过特征和不确定map和CNN从位姿流向像素。Thanks to the uncertainties and robust cost, PixLoc is robust to incorrect 3D geometry and works well with noisy reference data like sparse SfM models. During training, an imperfect 3D representation is sufficient – our approach does not require accurate or dense 3D models.

**Loss function** 通过比较每层预测的位姿和真值位姿来训练网络，作者最小化3D点的重投影误差：

![](<../../../.gitbook/assets/image (353).png>)

其中$$\gamma$$为Huber损失。This loss weights the supervision of the rotation and translation adaptively for each training example and is invariant to the scale of the scene, making it possible to train with data generated by SfM. To prevent hard examples from smoothing the fine features, we apply the loss at a given level only if the previous one succeeded in bringing the pose sufficiently close to the ground truth. Otherwise, the subsequent loss terms are ignored.

### Localization pipeline

**Initialization** We apply PixLoc to image pyramids, starting at the lowest resolution, yielding coarsest feature maps of size W=16. To keep the pipeline simple, we select the initial pose as the pose of the first reference image returned by image retrieval. This results in a good convergence in most scenarios.

**3D structure** For simplicity and unless mentioned, for both training and evaluation, we use sparse SfM models triangulated from posed reference images using hloc and COLMAP. Given a subset of reference images, e.g. top-5 retrieved, we gather all the 3D points that they observe, extract multilevel features at their 2D observations, and average them based on their confidence.

### Experiments

**Architecture** We employ a UNet feature extractor based on a VGG19 encoder pretrained on ImageNet, and extract L=3 feature maps with strides 1, 4, and 16, and dimensions Dl=32, 128, and 128, respectively. PixLoc is implemented in PyTorch, extracts features for an image in around 100ms, and optimizes the pose in 200ms to 1s depending on the number of points.&#x20;

**Training** We train two versions of PixLoc to demonstrate its ability to learn environment-specific priors. The benefits of such priors are analyzed in Supplemental B. One version is trained on the MegaDepth dataset, composed of crowd-sourced images depicting popular landmarks around the world, and the other on the training set of the Extended CMU Seasons dataset, a collection of sequences captured by car-mounted cameras in urban and rural environments. The latter dataset exhibits large seasonal changes with often only natural structures like trees being visible in the images, which are challenging for feature matching. We sample covisible image pairs and simulate the localization of one image with respect to the other, given its observed 3D points. The optimization runs for 15 iterations at each level and is initialized with the pose of the reference image.

#### Comparison to learned approaches

![](<../../../.gitbook/assets/image (1067).png>)

#### Large-scale localization

![](<../../../.gitbook/assets/image (301).png>)

#### Additional insights

![](<../../../.gitbook/assets/image (836).png>)
