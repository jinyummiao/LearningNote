# \[ICLR 2020] ViT

{% embed url="https://openreview.net/forum?id=YicbFdNTTy" %}

{% embed url="https://github.com/google-research/vision_transformer" %}

### Abstract

作者提出，可以不用卷积结构，完全用transformer去处理图像块序列，也可以在视觉任务上达到很好的效果。

### Introduction

作者将图像分割成图像块，并用线性映射将图像块映射为tokens，作为transformer的输入。当模型在中等大小的数据集上训练时，如ImageNet，模型相比有同等大小的ResNet表现稍差，说明transformer缺少CNNs固有的归纳偏差，比如平移不变性和局部性，因此当训练数据不够时，泛化能力不强。但是，当模型在大规模数据集上训练时，作者发现大规模训练要优于归纳偏差。迁移后的ViT模型可以达到很好的效果。

### Method

![](<../../../.gitbook/assets/image (181).png>)

模型基本参照原本NLP任务中的transformer结构。

#### VisionTransformer (ViT)

作者将图像$$x\in \mathbb{R}^{H \times W \times C}$$reshape为一组flattened 2D图像块$$x_p\in \mathbb{R}^{N \times (P^2 \cdot C)}$$，其中H和W是图像分辨率，C为通道数，(P,P)是图像块的分辨率，$$N=HW/P^2$$为图像块的数量。Transformer需要定长的输出向量，所以作者将图像块展平，并通过一个可训练的线性映射将图像块向量映射为D维的向量，称为patch embeddings.&#x20;

![](<../../../.gitbook/assets/image (810).png>)

同时，作者在embedding中加入了一个class token ($$z^0_0 = x_{class}$$)，从Transformer encoder输出时的状态$$z^0_L$$作为图像表征y。在预训练和fine-tuning阶段，在$$z^0_L$$上连接一个classification head。该分类head在预训练阶段通过一个有一个隐层的MLP实现，在fine-tuning阶段通过一个单全连接层实现。 作者在patch embeddings中加入了1D的position embeddings。&#x20;

tranformer encoder包含交替出现的multi-head self-attention层(MSA)和MLP模块，在每个模块前加入LayerNorm(LN)，在每个模块后加入residual connection。MLP模块中有两层和一个GELU非线性函数。&#x20;

![](<../../../.gitbook/assets/image (521).png>)

#### Fine-tuning and Higher Resolution

作者在大数据集上预训练ViT模型，然后在下流任务上fine-tune。在fine-tune时，去掉预训练的prediction head，用一个初始化为0的$D \times K$全连接层代替，其中K是下流任务的分类数量。在比pre-training更高的分辨率上进行fine-tuning效果会比较好。当输入更高分辨率的图像时，作者固定了patch size，选择将patch序列的长度变大。ViT可以处理任何长度的序列，但是预训练好的position embeddings可能就没有意义了，所以作者根据position embeddings在原图中的位置来进行2D插值。要注意，这种分辨率调节和patch提取是唯一手动加入ViT模型的关于图像2D结构的感知偏差。

### Experiments

![](<../../../.gitbook/assets/image (139).png>)

模型的几种变型。&#x20;

![](<../../../.gitbook/assets/image (881).png>)

![](<../../../.gitbook/assets/image (207).png>)

![](<../../../.gitbook/assets/image (833).png>)
