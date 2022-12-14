---
description: >-
  CALC2.0：Combining Appearance, Semantic and Geometric Information for Robust
  and Efficient Visual Loop Closure
---

# \[IROS 2019] CALC2.0

{% embed url="https://arxiv.org/abs/1910.14103" %}

{% embed url="https://github.com/rpng/calc2.0" %}

### Abstract

作者认为现有的基于CNN的回环检测算法虽然使用了语义、外观或者几何特征信息，但是没有很充分的利用一张图像可以提供过的全部信息（语义、外观、几何信息等），并且需要人工设置参数去完成实际的回环检测。这篇论文中，作者提出了一个专为场景识别设计的神经网络，由semantic segmentator、Variational Autoencoder(VAE)和triplet embedding network组成。该网络用于提取一个全局特征空间来描述图像的外观和语义分布。然后从低层卷积特征图中提取最大响应的区域作为局部关键点，关键点描述子也参考hand-crafted特征的思路从这些特征图中提取。关键点被用于全局匹配搜索候选的回环，并用于最后的几何验证来提出错误回环。

### Method

这篇论文提出的方法核心思想是，尽可能充分的利用单目图像可以提供的信息，如外观、语义和几何一致性，以此实现无需人工设计参数的回环检测。

#### Network Design

![](<../../.gitbook/assets/image (43).png>)

网络由三部分组成：1 VAE， 1 semantic segmentator和1 siamese triplet embedding。最后使用时，只使用encoder部分。网络的输入是RGB图像，大小为$$H\times W$$。encoder部分由一个3x3卷积，两个residual block和四个2x2卷积+pool组成。最后用两个1x1卷积来计算隐藏变量$$\mu, \sigma$$。隐藏变量是训练用于确定一个高斯分布的参数$$\mathcal{N}(\mu, diag(exp(\sigma)))$$.取$$\sigma$$的指数只是为了提升数值稳定性。在这种解释下，隐藏参数应该都是向量，它们通过对它们所在的$$\frac{H}{16} \times \frac{W}{16} \times M(N+1)$$的3D数组展平得到。 隐藏变量通过让其构建一个标准正太分布来优化，使用KL散度作为损失函数：&#x20;

![](<../../.gitbook/assets/image (58).png>)

用一个符合标准正太分布的$$\epsilon$$来采样，隐藏变量$$z=\mu + diag(exp(\sigma))^{\frac{1}{2}}\epsilon$$被切分成N+1组特征图，对应着视觉外观和N个语义类别。&#x20;

视觉外观部分的decoder用RGB reconstruction loss来训练：&#x20;

![](<../../.gitbook/assets/image (171).png>)

其中$$x_{h,w,c},r_{h,w,c}$$分布是输入图像和重建图像在(h,w,c)处的值。&#x20;

语义分割部分decoder的输出在channel维度拼接在一起，用一个标准的pixel-wise softmax cross entropy loss $$L_s$$来训练，用一个权重来平衡类别偏差，每个类的权重是数据集中所有当前类别的像素的百分比的倒数经标准化（最多的类别权重为1）后得到。&#x20;

作者在COCO stuff数据集上进行训练，没有使用COCO提供的92个类别，而是构建了13个超类来更普遍的描述场景的语义信息。这样可以帮助提升模型的语义分割精度，减少所需局部描述子的数量，来获得更紧密的嵌入表示。所有动态物体都被包含在“other”类中，让模型更关注静态的物体。&#x20;

在网络结构方面，除了计算隐藏变量和decoder最后的输出层外，所有卷积层都使用了Exponential Linear Unit（ELU）激活函数，语义分割decoder输出层和计算隐藏变量的卷积层没有激活函数，图像重建decoder加入sigmoid激活函数。在encoder层中，步长为2，卷积核尺寸为2x2的max-pooling被用于下采样特征，而subpixel convolution用于上采样特征。 全局图像描述子从隐藏变量$$\mu$$中获得，其可以视作一个3D的数组，一组Mx(N+1)个D维的局部描述子，对N+1个decoder每个输入M个feature map；或者可以视为一个长度为DxMx(N+1)的向量，其中$$D=\frac{H}{16} \times \frac{W}{16}$$&#x20;

![](<../../.gitbook/assets/image (826).png>)

根据$$\mu$$的第二种定义，作者计算了残差$$\mu-c$$，其中c是由M x (N+1)个在维度D上学习到的聚类中心在channel维度拼接而成的，它是用一个高斯分布随机初始化得到的，训练去最小化triplet embedding loss。该残差然后利用NetVLAD中的intra-normalization处理，用channel维度的L2-norm来防止描述子崩坏。然后，根据$$\mu$$的第一种定义，作者标准化整个描述子，以适应用内积来计算cosine相似度。采用triplet embedding loss来训练：&#x20;

![](<../../.gitbook/assets/image (7).png>)

#### Network Training

网络利用Adam训练，总的损失函数为&#x20;

![](<../../.gitbook/assets/image (504).png>)

作者用COCO数据集完成训练，由于没有true positive数据，所以作者用homography随机warp图像，随机将图像变黑来仿真夜视图像，随机左右翻转图像，来获得fake true positive。

#### Inference

**keypoint extraction**

全局描述子可以用最近邻搜索完成图像检索，但是需要阈值来确定匹配。为了解决这一问题，作者选择提取低层conv5层的卷积特征图中的最大激活区域来作为关键点。conv5层是全分辨率的，具有32维的特征图。为了获得的特征数量是有意义的，作者提取图像中每个大小为$$\frac{H}{N_w} \times \frac{W}{N_w}$$的划窗中的最大响应区域作为特征。重复的特征被剔除。&#x20;

得到关键点后，作者设计了一种类似于BRIEF的描述子，在conv5层的输出特征图(32d)上，作者在关键点周围3x3的邻域内计算特征向量的残差，将这些残差拼接在一起，得到256d关键点描述子。这些描述子在匹配时直接用欧拉距离度量相似度，在匹配时，作者使用K(=2)-NN来搜索，利用传统的ratio test来确定一个有效的匹配，

**loop closure detection**

作者用K(=7)-NN；来搜索可能的回环，然后用特征匹配来验证回环，只有可以通过RANSAC获得有效fundamental矩阵的匹配（也就是至少8对有效匹配）的图像才被认为是正确回环。

### Performance

![](<../../.gitbook/assets/image (32).png>)

作者展示了在wall、structure other、visual appearance分量上，database、positive、negative的相似度，可以看到，appearance产生了混淆，但是根据语义信息，则可以较好的分辨positive和negative。&#x20;

![](<../../.gitbook/assets/image (506).png>)

![](<../../.gitbook/assets/image (882).png>)

![](<../../.gitbook/assets/image (532).png>)

![](<../../.gitbook/assets/image (1006).png>)

比NetVLAD表现好，很强了
