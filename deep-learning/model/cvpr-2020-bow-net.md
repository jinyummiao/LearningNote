---
description: Learning Representations by Predicting Bags of Visual Words
---

# \[CVPR 2020] BoW-Net

{% embed url="https://openaccess.thecvf.com/content_CVPR_2020/papers/Gidaris_Learning_Representations_by_Predicting_Bags_of_Visual_Words_CVPR_2020_paper.pdf" %}

{% embed url="https://github.com/valeoai/bownet" %}

### Abstract

作者提出一种基于空间上稠密描述子的自监督方法，该方法将离散的视觉信息（或称视觉单词）编码。为了构建这个离散的表征，作者将一个预训练的自监督convnet输出的feature map进行量化为一个基于K-means的词典上。然后，作为一个自监督任务，当给定图像的扰动版本，作者训练了另一个convnet，预测图像视觉单词的直方图向量，即BoW表征。所提出的任务可以让convnet学习具有扰动不变性和基于内容的图像特征，对下流的图像理解任务有很大作用。

### Introduction

作者先回顾了基于重建的自监督学习存在问题，发现NLP领域中基于重建的自监督方法更为成功，因为1）单词毫无疑问地比图像像素更能表示高层的语义信息；2）单词是在离散空间中定义的，而图像像素是连续的，即使不改变描述的内容，小的像素扰动就会很显著地影响重建任务。&#x20;

**Spatially dense image quantization into visual words** 受NLP方法的启发，作者在这篇论文中提出一种图像领域的自监督方法，旨在预测/重建离散的视觉信息，而非低层的像素信息。为了达到这样的离散目标，作者首先用现有的自监督方法来训练一个初始的convnet，可以学习到中到高层图像特征的表征。然后，对于每张图像，稠密地将convnet输出的feature map量化到基于K-means的词典上，产生基于离散编码的稠密图像描述子，称为visual words。这样离散的描述便于使用NLP领域的自监督方法。&#x20;

**Learning by "reconstructing" bags of visual words** 作者受BoW模型的启发，提出一种自监督任务，来预测在对图像施加扰动时视觉单词的直方图向量，即BoW向量。&#x20;

![](<../../.gitbook/assets/image (555).png>)

**Contribution** 这篇工作的贡献在于：1）在图像领域的自监督学习任务中提出使用离散的视觉单词表征；2）基于此，提出一种如图1所示的自监督学习方法，不同于传统的预测或重建图像像素信息，该方法首先使用一个自监督的预训练网络来将图像稠密地量化为一组视觉单词，然后训练第二个网络来预测受扰动的图像的BoW表征；3）实验验证，该方法可以学习到更高质量的图像表征，比第一个convnet的效果更好；4）提出的方法可以很容易嵌入其他自监督学习工作中，帮助提升效果。

### Approach

作者的目标是用自监督学习的方法来学习一个特征提取模型或者convnet $$\Phi(\cdot)$$，参数定义为$$\theta$$，当输入图像x，产生高质量的表征$$\Phi(x)$$。这里定义高质量指对其他视觉任务有帮助，如分类、检测等。为此，作者假设有一批未标注的图像X，来训练模型。定义初始化的自监督学习的预训练模型为$$\hat{\Phi}(\cdot)$$。这里作者使用了自监督学习方法RotNet（Unsupervised representation learning by predicting image rotations）来训练初始化模型。&#x20;

作者先使用初始化模型$$\hat{\Phi}(\cdot)$$来得到基于视觉单词的空间上稠密的描述子。然后讲这些描述子聚合到BoW向量上，训练$$\Phi(\cdot)$$来重建当图像受扰动时的BoW向量。注意，$$\hat{\Phi}(\cdot)$$在训练过程中保持不变。在训练好$$\Phi(\cdot)$$后，可以设置$$\hat{\Phi}(\cdot) \leftarrow \Phi(\cdot)$$然后重复训练。

#### Building spatially dense discrete description

给定一个训练图像x，第一步先利用预训练的convnet $$\hat{\Phi}(\cdot)$$构建空间上稠密的基于视觉单词的描述子q(x)。记$$\hat{\Phi}(x)$$为输入为x时$$\hat{\Phi}(\cdot)$$的输出，分辨率为$$\hat{h} \times \hat{w}$$，维度为$$\hat{c}$$。且$$\hat{\Phi}^u(x)$$为位置处的$$\hat{c}$$维特征向量，其中$$U=\hat{h} \times \hat{w}$$。为了产生描述子$$q(x)=[q^1(x),...,q^U(x)]$$，用一个预定义的词典$$V=[v_1,...,v_K]$$来量化$$\hat{\Phi}(x)$$，其中单词维度为$$\hat{c}$$，单词数量为K。&#x20;

![](<../../.gitbook/assets/image (547).png>)

词典V可以用K-means算法通过优化以下目标来构建：&#x20;

![](<../../.gitbook/assets/image (359).png>)

其中视觉单词$$v_k$$是第k个聚类的中心值。

#### Generating Bag-of-Words representations

有了图像x的离散描述q(x)，第二步是构建图像的BoW向量y(x)：&#x20;

![](<../../.gitbook/assets/image (363).png>)

或：&#x20;

![](<../../.gitbook/assets/image (1036).png>)

然后经过L1-normalization处理。

#### Learning to “reconstruct” BoW

基于上述的BoW表征，作者提出一种自监督方法：给定图像x，加入扰动$$g(\cdot)$$，得到扰动图像$$\tilde{x}=g(x)$$，然后训练模型来预测/重建原始图像的BoW向量y(x)。为此，作者定义了一个预测层$$\Omega(\cdot)$$，以$$\Phi(\tilde{x})$$为输入，输出BoW向量中K个视觉单词的K维Softmax分布。更准确的说，预测层是通过一个linear-plus-softmax层实现的：&#x20;

![](<../../.gitbook/assets/image (1024).png>)

其中$$\Omega^k(\Phi(\tilde{x}))$$是第k个视觉单词的softmax概率，$$W=[w_1,...w_K]$$是线性层中的K个c维权重向量（每个对应一个视觉单词）。作者使用了L2-normalized权重向量$$\overline{w}k$$，并加入了一个可学习的幅度参数$$\gamma$$. 对线性层进行重新参数化的原因是，视觉单词在数据集中的分布不平衡(例如视觉词出现的频率或在图像中出现的次数)。因此，如果没有这样的重新参数化，网络将试图使每个权重向量的大小与其对应的视觉单词的频率成比例(因此基本上倾向于最经常出现的词)。

**Self-supervised training objective** 用于训练$$\Phi(\cdot)$$的损失函数为预测的softmax分布$$\Omega(\Phi(\tilde{x}))$$和BoW分布y(x)之间的cross-entropy：&#x20;

![](<../../.gitbook/assets/image (1000).png>)

其中$$loss(\alpha,\beta)=-\sum^K_{k=1}\beta^k log \alpha^k$$为离散分布$$\alpha=(\alpha^k)$$和$$\beta=(\beta^k)$$之间的cross-entropy loss。$$\theta$$为$$\Phi(\cdot)$$的可学习参数，$$(W,\gamma)$$为$$\Omega(\cdot)$$中的可学习参数，$$\tilde{x}=g(x)$$.&#x20;

**Image preturbations** 图像扰动包括：color jittering、以p的概率将图像转换为灰度图、随机图像裁剪、尺度或长宽比畸变、水平翻转、CutMix（Cutmix: Regularization strategy to train strong classifiers with localizable features.）。根据CutMix，当给定两张图像$$\tilde{x}_A=g(x_A),\tilde{x}_B=g(x_B)$$，通过从B中随机采样一个patch，替换到A中，产生一张合成的图像$$\tilde{x}_S$$。patch的位置和尺寸都是随机的。则BoW向量应当是两张图像的凸结合：$$\lambda y(x_A)+(1-\lambda) y(x_B)$$，其中$$1-\lambda$$为patch占图像面积的比例。（**个人感觉，CutMix可以有效的将BoW中的visual words与图像中的patch一一对应起来，增强可解释性**）

### Experiments

固定模型的BoW表征不变，在模型后面接线性全连接层，作为分类head，head需要训练。

![](<../../.gitbook/assets/image (1003).png>)

![](<../../.gitbook/assets/image (537).png>)

![](<../../.gitbook/assets/image (1022).png>)

![](<../../.gitbook/assets/image (1040) (1).png>)

![](<../../.gitbook/assets/image (188).png>)

![](<../../.gitbook/assets/image (491).png>)

![](<../../.gitbook/assets/image (174) (1).png>)

![](<../../.gitbook/assets/image (873).png>)

![](<../../.gitbook/assets/image (191).png>)

![](<../../.gitbook/assets/image (488).png>)
