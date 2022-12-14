---
description: Deep Fundamental Matrix Estimation
---

# \[ECCV 2018] DeepF

{% embed url="https://openaccess.thecvf.com/content_ECCV_2018/papers/Rene_Ranftl_Deep_Fundamental_Matrix_ECCV_2018_paper.pdf" %}

{% embed url="https://github.com/isl-org/DFE" %}

## Abstract

We present an approach to robust estimation of fundamental matrices from noisy data contaminated by outliers. The problem is cast as a series of weighted homogeneous least-squares problems, where robust weights are estimated using deep networks. The presented formulation acts directly on putative correspondences and thus fits into standard 3D vision pipelines that perform feature extraction, matching, and model fitting. The approach can be trained end-to-end and yields computationally efficient robust estimators. We present an approach to robust estimation of fundamental matrices from noisy data contaminated by outliers. The problem is cast as a series of weighted homogeneous least-squares problems, where robust weights are estimated using deep networks. The presented formulation acts directly on putative correspondences and thus fits into standard 3D vision pipelines that perform feature extraction, matching, and model fitting. The approach can be trained end-to-end and yields computationally efficient robust estimators.

## Introduction

In this work we present an approach that is able to learn a robust algorithm for fundamental matrix estimation from data. Our approach combines deep networks with a well-defined algorithmic structure and can be trained end-to-end. In contrast to naive deep learning approaches to this problem, our approach disentangles local motion estimation and geometric model fitting, leading to simplified training problems and interpretable estimation pipelines. As such it can act as a drop-in replacement for applications where the RANSAC family of algorithms is commonly employed. To achieve this, we formulate the robust estimation problem as a series of weighted homogeneous least-squares problems, where weights are estimated using deep networks.

## Preliminaries

定义输入数据的单个d维元素为一个点$$p_i \in R^d$$，记$$P\in \mathcal{P}=R^{N \times d}$$为N个d维点​的集合。用$${(P)}_i$$​来表示P矩阵的第$$i$$​行。注意这个点可以是一些度量空间里的点，也可以是F矩阵或H矩阵估计中所使用的点correspondence（例如，在这种情况下，$$p_i \in R^4$$表示一对相互匹配的像素坐标$$\tilde{p}_i \leftrightarrow \tilde{p}'_i$$）

在许多几何模型拟合问题中，都会出现齐次最小二乘优化问题:

![](<../../../.gitbook/assets/image (444).png>)

其中$$x\in R^{d'}$$​是模型参数，A是一个任务相关的数据点投影关系$$A: \mathcal{P}\rightarrow R^{kN \times d'} (kN \ge d', k>0)$$，公式1是一个闭式解。​

以简单的超平面拟合为例。令$${(n^T,c)}^T$$为一个法向量为n，截距为c为超平面。超平面拟合旨在从一组点P中推理出$${(n^T,c)}^T$$。以最小二乘的方式拟合超平面：

![](<../../../.gitbook/assets/image (382).png>)

据此求解公式1，我们可以用模型提取函数g(x)来提取平面，将x映射到模型参数上：\


![](<../../../.gitbook/assets/image (425).png>)

如果数据中没有外点的话，最小二乘法一定可以求出最优解。但是在实际问题中，数据经常包含了外点，使用最小二乘法求解问题会产生错误的解。

为了鲁棒地求解几何模型拟合问题，一个解决方法是在公式1的残差中加入鲁棒的损失函数$$\Phi$$，但这样的优化问题一般不是一个闭式解。​实际中，一般通过求解一系列重新加权的最小二乘问题来近似解决优化问题：

![](<../../../.gitbook/assets/image (404).png>)

其中权重w的具体形式由$$\Phi$$和几何模型来决定。

以超平面拟合问题为例，当$$p_i$$​是内点时，令$$w(p_i,x^j)=w_i=1$$，否则$$w(p_i,x^j)=w_i=0$$，这样的话，根据公式4可以一次就求解出正确的模型：\


![](<../../../.gitbook/assets/image (446).png>)

## Deep Model Fitting

本文的方法是基于公式4的，它可以视为一个迭代的加权最小二乘算法（IRLS），使用了一个复杂的、可学习的加权函数。Since we are learning weights from data, we expect that our algorithm is able to outperform general purpose approaches whenever one or more of the following assumptions are true. (1) The input data admits regularity in the inlier and outlier distributions that can be learned. An example would be an outlier distribution that is approximately uniform and sufficiently dissimilar to the inlier noise distribution. This is a mild assumption that in fact has been exploited in sampling-based approaches previously. (2) The problem has useful side information that can be integrated into the reweighting function. An example would be matching scores or keypoint geometry. (3) The output space is a subset of the full space of model parameters. An example would be fundamental matrix estimation for a camera mounted on a car or a wheeled robot.

In the following we adopt the general structure of algorithm (4), but do not assume a simple form of the weight function w. Instead we parametrize it using a deep network and learn the network weights from data such that the overall algorithm leads to accurate estimates. Our approach can be understood as a meta-algorithm that learns a complex and problem-dependent version of the IRLS algorithm with an unknown cost function. We show that this approach can be used to easily integrate side information into the problem, which can enhance and robustify the estimates.

**Model estimator** 定义权重函数为$$w： \mathcal{P} \times \mathcal{S} \times R^{d'} \rightarrow {(R_{>0})^N}$$，其中$$S\in \mathcal{S}=R^{N \times s}$$​是每个点的side information。这一函数是在全局范围内定义的，所以每个点都会彼此影响。因为w可以是非平凡函数，作者用神经网络$$\theta$$​来得到它。则公式4可以变为：

![](<../../../.gitbook/assets/image (472).png>)

问题的关键在于如何找到合适的$$\theta$$​来得到鲁棒和正确的解。用矩阵形式表示：

![](<../../../.gitbook/assets/image (456).png>)

其中$${(W^j(\theta))}_{i,i}=\sqrt{w^j_i}$$，W是一个对角矩阵。​

**Proposition 1** 令$$X=U\Sigma V^T$$表示X的SVD奇异值分解。则公式7的解​$$x^{j+1}$$由矩阵$$W(\theta)A$$的最小奇异值对应的特征向量$$v_{d'}$$​给出。

因此，模型拟合问题的解是$$g(f(W(\theta)A))$$，其中$$f(X)=v_{d'}$$​，g(x)将SVD结果映射到几何模型的参数。

为了用梯度优化来学习权重$$\theta$$​，需要让梯度从SVD层反向传播（没看懂..:cry:）:

![](<../../../.gitbook/assets/image (458).png>)

![](<../../../.gitbook/assets/image (461).png>)

模型估计器的语义示意图如图1所示。模块以点P和一组权重w为输入，在预处理阶段构建矩阵$$W(\theta)A$$​，进行SVD奇异值分解，然后进行模型提取步骤g(x)，得到几何模型的估计。根据新估计的模型、输入的点、side information和当前估计模型的残差，估计新的权重。

**Weight estimator** 为了准确估计权重，estimator需要满足两个要求：它必须和输入数据的顺序是一致的，并且能够处理任意数量N的输入点。由于喂给estimator的输入数据的顺序是不一定的，所以该函数要能够不受顺序影响地集成全局信息。

作者根据PointNet的方法来用深度神经网络处理无序数据。核心思路是：为了使网络不受输入数据顺序的影响，网络中的每个操作都应当是与顺序无关的，尤其是那些需要在多个数据点上进行计算的层。全局平均和最大池化可以满足这一需求。作者在PointNet的结构上做了微小的改动：作者没有采用池化层来集成全局信息，而是在每层后加入instance normalization：

![](<../../../.gitbook/assets/image (430).png>)

其中$$h_i$$​为点i的特征向量，均值$$\mu(h)$$和方差$$\sigma^2(h)$$是在维度N上计算的。这一步将所有店的全局信息分布集成起来，进行正则化。instance normalization是与输入数据顺序无关，所以整个网络也与数据顺序无关。

算法的流程如图1所示，它包含了一个重复计算的线性层（对每个点独立计算）+peaky ReLU+instance normalization。为了得到正的权重，误差估计器加入了一个softmax。作者定义了两个网络，$$w_{init}(P,S)$$​用于计算初始权重，$$w_{iter}(P,S,r,w^j)$$​来在模型估计后更新权重，其中r表示当前估计的几何残差：$$(r)_i=r(p_i,g(x^j))$$​

**Architecture** The complete architecture consists of an input weight estimator $$w_{init}$$, repeated application of the estimation module, and a geometric model estimator on the final weights. In practice we found that five consecutive estimation modules strike a good balance between accuracy and speed.

![](<../../../.gitbook/assets/image (412).png>)

**训练过程** We implement the network in PyTorch. In all applications that follow we use Adamax with an initial learning rate of $$10^{-3}$$ and a batch size of 16. We reduce the learning rate every 10 epochs by a factor of 0.8 and train for a total of 100 epochs.

## Fundamental Matrix Estimation

![八点法求解F](<../../../.gitbook/assets/image (394).png>)

本文基于8点法求解F矩阵，将所有坐标缩放到$${[-1,1]}^2$$​，定义预处理函数：

![](<../../../.gitbook/assets/image (401).png>)

其中$$\hat{p}_i={({(p_i)}_1,{(p_i)}_2,1)}^T$$和$$\hat{p'}_i={({(p_i)}_3,{(p_i)}_4,1)}^T$$是两张图像中对应像素的归一化坐标。T和T'为根据预测权重来中心化、缩放数据的正则化矩阵。定义模型提取器为：

![](<../../../.gitbook/assets/image (462).png>)

其中F为矩阵矩阵。The model extractor explicitly enforces rank deficiency of the solution by projecting to the set of rank-deficient matrices. It is well-known that this projection can be carried out in closed formed by setting the smallest singular value of the full-rank solution to zero. 用对称极线距离作为残差函数：

![](<../../../.gitbook/assets/image (465).png>)

由于F矩阵无法直接比较。所以作者通过比较它们对于匹配的作用来比较F矩阵。为此，作者在两图中采样了网格化的点，并得到其在另一张图像中的真值极线，作为符合真值极线几何的虚拟匹配对。这些虚拟的、不受噪声影响的内点匹配$$p^{gt}_i$$​可以用于定义损失函数：

![](<../../../.gitbook/assets/image (475).png>)

Clamping the residuals ensures that hard problem instances in the training set do not dominate the training loss. $$\gamma=0.5$$​. 由于网络可以得到可解释的中间结果（就是迭代计算中的中间值），所以作者对这D个中间结果都计算了loss。

为了高效训练，作者限制每张图中的关键点数为1000，不够的话用随机点来补充。At test time we evaluate the estimated solution and perform a final, non-robust model fitting step to the 20 points with smallest residual error in order to correct for small inaccuracies in the estimated weights.

## Experiments

用SIFT特征获得匹配。

![](<../../../.gitbook/assets/image (476).png>)

![](<../../../.gitbook/assets/image (473).png>)

![](<../../../.gitbook/assets/image (448).png>)

![](<../../../.gitbook/assets/image (426).png>)
