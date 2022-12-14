---
description: >-
  Visual Transformers: Token-based Image Representation and Processing for
  Computer Vision
---

# \[CVPR 2020] Token ViT

{% embed url="https://arxiv.org/abs/2006.03677" %}

### Abstract

卷积层认为图像中所有像素都是同等重要的，不管信息的内容，将图像中的所有信息都建模，并且很难将远距离的信息联合起来。因此，这篇论文中，作者提出：(a)用语义视觉tokens来表示图像；(b)用transformer来表示对tokens之间的关系。关键的是，visual transformer在一个 semantic token space中工作，根据上下文信息处理不同的图像部分，相比pixel-space中的transformer减少了大量的计算量。

### Introduction

作者认为，在视觉领域，一般利用的是pixel-level的视觉信息，这些信息被卷积网络所利用。虽然这一方法已经取得了巨大成功，但是也存在一些问题：

1. 像素的重要性是不同的：比如分类模型应该更重视前景物体，分割模型应该更重视行人而非不成比例的天空、道路和植被。但是，卷积层统一处理这些图像块，没有考虑重要性。导致了计算量和表征上的低效。
2. 图像中的信息是片面的：如角点等低层信息，所有自然图像中都会存在，所以采用低层的卷积核是正确的。但是高层的特征，如耳朵的形状，只会在某些特定图像中出现，所以对所有图像都使用全部高层卷积核是低效的。
3. 卷积无法获取相对较远距离的信息：卷积处理只能感受到较小范围的信息，无法获取长距离语义信息之间的关系。 为了解决以上问题，作者提出visual transformer（VT），一种新的模式来表征图像，处理图像的高层信息。作者认为少量的视觉tokens足以表示图像的高层信息，所以作者用spatial attention来将特征图转化为semantic tokens，然后将这些tokens输入transformer中，来获取tokens之间的相互作用。生成的tokens可以直接用于各类视觉任务，如分类和分割。&#x20;

![](<../../../.gitbook/assets/image (350).png>)

### Visual Transformer

VT利用卷积层去学习密集分布的低层图像特征，输入transformer中，学习和关联稀疏分布的高层语义信息。该网络包括三个部分：1.将像素聚集成语义信息块，来获得一组紧凑的visual tokens；2.用transformer获取tokens之间的关联；3.将visual tokens投影回pixel-space来获得一个增强了的特征图。

#### Tokenizer

作者认为少数的视觉单词（或者visual tokens）就足以表述图像。所以作者提出了一种tokenizer模块来将特征图转换为一组visual tokens。令输入的特征图为$$X\in \mathbb{R}^{HW\times C}$$，visual tokens为$$T\in \mathbb{R}^{L\times C}$$，其中$$L \ll HW$$

**Filter-based Tokenizer**

对于特征图X，用point-wise卷积将每个像素$$X_p\in \mathbb{R}^C$$映射到L个语义组中的一个。在每组中，得到tokens T：&#x20;

![](<../../../.gitbook/assets/image (543).png>)

其中$$W_A\in \mathbb{R}^{C\times L}$$从X中构成了语义组，$${SOFTMAX}_{HW}(\cdot)$$将响应值转换为空间注意力。最后A乘X得到了X的加权平均值，获得了L个visual tokens。 但是，很多高层的视觉信息是稀疏的，只存在于少数图像中。所以，固定的已训练好的权重$$W_a$$会浪费计算资源，应为他们将所有的视觉信息都建模了。作者称此为"filter-based" tokenizer，因此该模型用了卷积核来提取visual tokens。

![](<../../../.gitbook/assets/image (694).png>)

**Recurrent Tokenizer**

作者提出了recurrent tokenizer来根据前一层的visual tokens来设置权重。让前一层的tokens $$T_{in}$$来指导这一层新tokens的提取：&#x20;

![](<../../../.gitbook/assets/image (883).png>)

其中$$W_{T\rightarrow R}\in \mathbb{R}^{C\times C}$$。这种情况下，VT可以增量式的调整visual tokens的集合。

![](<../../../.gitbook/assets/image (885).png>)

#### Transformer

作者用了标准的transformer结构：&#x20;

![](<../../../.gitbook/assets/image (812).png>)

其中$$T_{in}，{T'}_{out}，T_{out}\in \mathbb{R}^{L\times C}$$为visual tokens。与图卷积不同，在transformer中，tokens之间的weight是根据输入而定的，并且通过一种key-query之间的点乘来计算：&#x20;

![](<../../../.gitbook/assets/image (852).png>)

这使得该模型可以使用很少的16个visual tokens，而不是图卷积方法中上百个类似的节点。经过self-attention，作者使用了一个非线性激活函数和两个点乘，其中$$F_1，F_2\in \mathbb{R}^{C\times C}$$为权重，$$\sigma(\cdot)$$是ReLU函数。

#### Projecter

许多视觉任务需要pixel-level的细节信息，但是这些信息在visual tokens中没有保存。因此，作者将transformer的输出融合到特征图中，来调整特征图的pixel-array表征能力：&#x20;

![](<../../../.gitbook/assets/image (344).png>)

其中$$X_{in}，X_{out}\in \mathbb{R}^{HW\times C}$$为输入和输出的特征图。$$(X_{in}W_Q)\in \mathbb{R}^{HW \times C}$$是根据输入特征图$$X_{in}$$计算的query，$${(X_{in}W_Q)}_{p}\in \mathbb{R}^C$$编码了p点从visual tokens中所需的信息。$$(TW_k)\in \mathbb{R}^{L\times C}$$是从token T中计算的key。$${(TW_K)}_l\in \mathbb{R}^C$$表示了第l个token编码的信息。这一key-query点乘决定了如果将visual tokens T中编码的信息映射到原本的特征图中。

### Using Visual Transformers in Vision Models

在图像分类模型中，作者将ResNet的最后一个卷积模块，替换成VT模块。&#x20;

![](<../../../.gitbook/assets/image (855).png>)

在图像分割模型中，作者利用FPN作为baseline，作者用VT模块来代替FPN中的卷积。

![](<../../../.gitbook/assets/image (284).png>)

### Experiments

![](<../../../.gitbook/assets/image (317).png>)

比baseline效果好，且FLOP低。&#x20;

![](<../../../.gitbook/assets/image (531).png>)

不同tokenizer方法的对比，用聚类和卷积的方法可以更好的将特征按照语义信息聚合在一起。&#x20;

![](<../../../.gitbook/assets/image (202).png>)

加入recurrent tokenizer效果更好。&#x20;

![](<../../../.gitbook/assets/image (146).png>)

用不同方法去获取token之间的相互关系，transformer效果最好。&#x20;

![](<../../../.gitbook/assets/image (367).png>)

较少的token足以表示图像。&#x20;

![](<../../../.gitbook/assets/image (522).png>)

projection过程是有必要的，原特征图中包含着一些细节信息。&#x20;

![](<../../../.gitbook/assets/image (154).png>)

与其他基于attention的增强版ResNet相比较，VT效果更好，FLOP更低。&#x20;

![](<../../../.gitbook/assets/image (354).png>)

在语义分割上，比baseline更好。
