---
description: E2BoWs：An End-to-End Bag-of-Words Model via Deep Convolutional Neural Network
---

# \[neurocomputing 2020] E2BoWs

{% embed url="https://www.sciencedirect.com/science/article/pii/S0925231219309105" %}

{% embed url="https://github.com/liu-xb/E2BoWs" %}

### Abstract

传统的BoW模型需要很多步骤来构建BoW模型，如提取局部特征、训练词典、特征量化等。这些步骤彼此独立，难以一起优化。并且，由于依赖于人工设计的局部特征，BoW模型难以包含高层的语义信息。这些问题阻碍了BoW模型在一些大规模图像应用场景中的使用。因此，本文作者提出了一个可以end-to-end训练的BoW模型，模型输入一张图像，识别并分离出其中的语义目标，最后输出具有高语义分辨能力的视觉单词。更明确的说，模型首先通过卷积神经网络生成了对应不同物体类别的语义特征图（SFMs），然后引入BoW层来对每个特征图产生视觉单词。作者也引入了新的学习算法来提升所生成的BoW模型的稀疏性。

### Introduction

![](<../../.gitbook/assets/image (546).png>)

### Proposed Method

E2BoWs模型是通过修改带有batch-normalization(BN)层的GoogLeNet得到的。在Inception5层之前，模型的结构与带有BN层的GoogLeNet一样。作者从卷积层中获得特征，带有更多的语义信息。因此，作者将最后n维的全连接层(FC)转换为一个卷积层，来生成n个SFMs，对应着n个训练类别。然后通过bag-of-words层将每个SFM转化为m个稀疏的视觉单词，得到$$m \times n$$个视觉单词。最后，作者用三部分损失函数来训练模型。

#### Semantic Feature Maps Generation

在GoogLeNet中，因为有标签监督，所以输出层包含着语义信息。但是它损失了一些视觉细节，比如目标的位置和大小。而Inception5包含了比语义信息更多的视觉信息。从输出层和Inception5层学习视觉单词可能会损失一些来自视觉细节或语义信息的分辨力。为了保存视觉信息和语义信息，作者提出由Inception5生成SFMs，然后从SFMs中产生视觉单词。&#x20;

![](<../../.gitbook/assets/image (157).png>)

SFMs是通过将全连接层的参数转化到卷积层中实现的。如图2所示，全连接层中的参数维度为$$1024 \times n$$，其中1024是pooling后的特征维度，而n是训练类别数。这些参数可以被转换到n个$$1024 \times 1 \times 1$$大小的卷积核。换言之，作者将$$1024 \times 1$$的全连接层的参数转换到了一个$$1024 \times 1 \times 1$$的卷积核中。因此，可以得到n通道的卷积核，从Inception5层可以生成n个SFMs。 在FC层中，每一维输出都是对应一个类别的分类分数。与FC层输出相比，SFMs也包含这样的分类线索。例如，平均池化每个SFM上的响应值将获得对应类别的分类分数。此外,SFMs保留了某些视觉线索，因为它们是从Inception5层产生的，没有经过pooling处理，没有丢失空间信息。&#x20;

![](<../../.gitbook/assets/image (321).png>)

可以看到，产生的SFMs具有较好的语义信息和空间信息。

#### Bag-of-Words Layer

为了保留SFMs中的空间和语义信息，作者引入了BoWL来对每个SFM生成稀疏的视觉单词。作者用一个局部的FC层+ReLU来对每个SFM生成视觉单词。最后每幅图像可以得到$$m \times n$$个视觉单词。与传统的FC层相比，局部FC层可以更好的保留每个SFM中的语义和视觉信息，并且参数量较小。举例，BoWL需要$$49 \times m \times n$$个参数，而传统的FC层需要$$(49 \times n) \times (m \times n)$$个参数。在生成词典时，作者丢掉了具有负平均响应值的SFMs，来减少非零视觉单词的数量，并且提高检索的效率。生成的视觉单词经过了L2-normlization，并且丢掉了响应值小于$$\beta$$的视觉单词。&#x20;

但是$$\beta$$很难确定，所以作者设立了基于KLD的sparsity loss来自动决定该阈值：&#x20;

![](<../../.gitbook/assets/image (1048).png>)

其中$$\hat{\rho}$$是非零单词数与总单词数间的理想比值(**Q:所以这个值需要人为设置?**)，而$$\rho$$是在N张训练集图像上计算的：&#x20;

![](<../../.gitbook/assets/image (1029).png>)

![](<../../.gitbook/assets/image (33).png>)

用此函数，模型可以学习阈值$$\beta$$来控制视觉单词的稀疏性。

#### Model Training

整个网络通过SGD进行训练，目标函数如下：&#x20;

![](<../../.gitbook/assets/image (346).png>)

因为triplet loss $$l_{tri}$$需要较长时间去收敛，所以引入了classification loss $$l_{cla}$$加快收敛。triplet loss中应用cosine distance去度量vector之间的相似度：&#x20;

![](<../../.gitbook/assets/image (323).png>)

由于sparsity loss中的sign()不可微，作者定义了梯度为：&#x20;

![](<../../.gitbook/assets/image (1071).png>)

#### Generalization Ability Improvement

由于ImageNet中有些类别看起来很相似，不利于训练该模型。所以作者根据训练时两个类别的相似程度调整了triplet loss function中的$$\alpha$$，如果两个类别很相似，则调小参数，否则调大。首先，作者根据ImageNet的类别树结构(http://image-net.org/explore)计算两个类别的相似度，令H代表树的高度，而$$h^{c_2}_{c_1}$$表示两个不同类别$$c_1$$和$$c_2$$的共同父节点的高度，则两类别的相似度可以定义为：$$S(c_1, c_2)=\frac{h}{H}$$，然后调整$$\alpha$$:&#x20;

![](<../../.gitbook/assets/image (693).png>)

### Experiments

![](<../../.gitbook/assets/image (46).png>)

![](<../../.gitbook/assets/image (1073) (1).png>)

![](<../../.gitbook/assets/image (197).png>)
