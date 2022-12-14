---
description: 'SEKD: Self-Evolving Keypoint Detection and Description'
---

# \[arxiv 2020] SEKD

{% embed url="http://arxiv.org/abs/2006.05077" %}

{% embed url="https://github.com/aliyun/Self-Evolving-Keypoint-Demo" %}

### Abstract

论文提出，现存的一些特征提取方法（hand-crafted or learnt）都没有考虑到detector和descriptor之间的相互促进作用，所以导致效果或多或少不尽人意。所以这篇文章，其实是设计了一种自监督训练框架，强调repeatability和reliability，用完全无标注的自然图像去学习特征。

### Introduction

作者更加细化的分别定义了detector和descriptor的repeatability和reliability：&#x20;

![](<../../.gitbook/assets/image (1007).png>)

总的来说就是两大特性，细分为四部分：（1）Repeatability：detector的repeatability体现在如果两幅图像描述了同一场景，那么在图1中“看到”的一个keypoint在图2中也应该可以看到；descriptor的repeatability体现在相同真实位置的关键点在不同图像中应该是invariant；（2）Reliability（其实可以理解为我们常说的disciminativeness）：detector的reliability体现在给定描述子的计算方法，一个detected keypoint应该可靠的区别于其他点，直白点说就是应该落在利于分辨的区域，避开重复性纹理区域；descriptor的reliability体现在给定了detection方法，计算出的描述子应该足够区分这些detected keypoints。&#x20;

思考一下上面的这些特性，其实repeatability特性是detector和descriptor自己的inherent property，而reliability则体现了detector与descriptor之间的interactive property。&#x20;

这篇论文别出心裁，不一起训练detector和descriptor，而是采用了一种iterative training strategy。利用上面说的inherent and interactive property，去指导训练。&#x20;

简单的说，找出所有具有reliable descriptor的keypoints，作为ground-truth去训练detector，用优化后的detector去检测keypoints，基于这些keypoints训练descriptor。重复这一过程，直到模型收敛。这就是self-evolving framework。整个训练过程不需要带有标注的数据。

### Architecture

![](<../../.gitbook/assets/image (807).png>)

SEKD采用了类似于SuperPoint的结构，backbone由1个卷积和9个ResNet\_v2模块，得到1/4大小的feature map。detector branch由2个deconv和1个softmax构成，输出2 x H x W的map P，代表keypoint probability，为了提升定位精度，有两个来自低层feature map的shortcut。descriptor branch由1个ResNet\_v2模块和1个bi\_linear上采样层构成，输出C-d描述子。

### Self-Evolving !

![](<../../.gitbook/assets/image (674).png>)

在训练过程中，网络利用两方面的监督：（1）关键点的选取，有可靠descriptor的point被认为是keypoint；（2）不同图像间的keypoints correspondence，这个通过用一张图像，经过affine transformation获得匹配图像，因此correspondence也可以直接获得。&#x20;

训练的流程基本分为四步：&#x20;

1.用detector branch得到keypoint probability map P，利用NMS筛选keypoint；&#x20;

2.在这些keypoint上，通过增强descriptor的repeatability和reliability来更新descriptor branch；&#x20;

3.计算keypoint，具有reliable（repeatable and distinct） descriptor的point被作为keypoints;&#x20;

4.在这些新的keypoint上，根据detector的repeatability和reliability来更新detector branch。

#### 1.Detect Keypoints using Detector

![](<../../.gitbook/assets/image (878).png>)

![](<../../.gitbook/assets/image (182).png>)

具有较高响应的点被视为可能的keypoint，经过NMS（半径设为4），每张图像可以获得1000个keypoint。但是由于不同图像条件下，可能没法取到一样的keypoint，所以作者采用了affine adaption方法，即对于原始图像进行random affine transformation和color jitter，获得新的图像后经过网络，获得不同图像条件下的P，最后映射回原图，取平均，得到最后的P.&#x20;

由于训练刚开始时，detector无法很好地检测点，所以作者随机选了关键点。

#### 2.Update Keypoint Descriptor

根据上一节，获得了一张图像I中的keypoints Q，对I和Q进行random affine transformation和color jitter，得到$$\hat{I}$$和$$\hat{Q}$$，并且可以获得特征的匹配关系$$<Q,\hat{Q}>$$，那么根据descriptor repeatability，匹配特征的des应该很靠近，根据descriptor reliability，不匹配的特征应该有很好的区分度。所以作者使用了triplet loss with hardest example mining去训练descriptor。&#x20;

![](<../../.gitbook/assets/image (311).png>)

除此之外，由于网络使用共享参数的backbone，所以为了保证detection的结果不会变化，作者还加入了一个损失函数：&#x20;

![](<../../.gitbook/assets/image (332).png>)

其中N’表示更新后的N。所以，用于更新descriptor的总loss为：&#x20;

![](<../../.gitbook/assets/image (1037).png>)

#### 3.Compute Keypoints via Descriptor

更新了descriptor后，下一步是要从descriptor map中提取keypoint。特征的reliability可从repeatability和distinctiveness两方面度量。对于repeatability：&#x20;

![](<../../.gitbook/assets/image (830).png>)

$$D_{i,i}$$越低，说明descriptor在相同位置上更靠近，repeatability更好。而对于distinctiveness：&#x20;

![](<../../.gitbook/assets/image (9).png>)

$$D_{i,\overline{i}}$$越大，说明descriptor在不同位置的差异越大，distinctiveness更好。所以综合两点，特征的reliability定义为：&#x20;

![](<../../.gitbook/assets/image (876).png>)

R越大，说明特征更reliable。 由于匹配的图像对是用affine transformation获得的，所以图像中可能一些点没有对应的R，所以这里依然采用了之前所用的affine adaption方法来获得一张average R。 另外，计算$$D_{i,\overline{i}}$$的计算量很大，所以作者在邻域内计算$$D_{i,\overline{i}}$$：&#x20;

![](<../../.gitbook/assets/image (318).png>)

在实验中，作者在1/4，1倍分辨率的descriptor map上分别计算R，然后将两个map融合，一起使用，以获得足够精细的R。

#### 4.Update Keypoint detector

根据descriptor计算出的keypoint可以看作ground-truth对detector进行训练，作者使用了focal loss：&#x20;

![](<../../.gitbook/assets/image (175).png>)

其中，Y是上一步计算出的keypoints。FL为focal loss，形式为：&#x20;

![](<../../.gitbook/assets/image (672).png>)

&#x20;同时，为了减小匹配图像对上detection结果的差异，作者加入了：&#x20;

![](<../../.gitbook/assets/image (1070).png>)

&#x20;来训练detector的repeatability，其中KLD是KL散度。并且还需要保证descriptor在这期间不受干扰，所以加入了：&#x20;

![](<../../.gitbook/assets/image (205).png>)

&#x20;最后综合三部分，训练detector的loss为：&#x20;

![](<../../.gitbook/assets/image (868).png>)

### Training

特征在MS COCO验证集上完成训练。 训练进行了5个iteration，每个iteration中分别训练detector和descriptor20个epoches

### Model Size

![](<../../.gitbook/assets/image (689).png>)

### Performance

![](<../../.gitbook/assets/image (352).png>)

![](<../../.gitbook/assets/image (158).png>)
