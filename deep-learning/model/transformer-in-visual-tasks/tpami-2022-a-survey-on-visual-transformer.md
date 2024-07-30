# \[TPAMI 2022] A Survey on Visual Transformer

{% embed url="https://arxiv.org/abs/2012.12556" %}

![](<../../../.gitbook/assets/image (1008).png>)

![](<../../../.gitbook/assets/image (994).png>)

### 1.Formulation of Transformer

![](<../../../.gitbook/assets/image (840).png>)

Transformer最早被用于NLP领域的机器翻译任务。如图所示，它包含一个encoder模块和一个decoder模块，其中包括多个结构相同的encoders和decoders。每个encoder和decoder都包含一个self-attention层和一个前向神经网络，其中decoder还包括一个encoder-decoder attention layer层。在transformer进行句子翻译前，需要先将句子中的每个单词embed到一个$$d_{model}=512$$维的向量.

#### 1.1.Self-Attention Layer

在self-attention层中，输入向量先被转化为三个不同的向量：query向量q，key向量k和value向量v，它们的维度都是$$d_q=d_k=d_v=d_{model}=512$$。不同输入产生的向量然后分别构成三个矩阵，即Q,K,V。然后，不同输入向量间的注意力可由下式计算：

1.计算不同输入向量间的分数$$S=Q\cdot K^T$$

2.为了梯度的稳定性，归一化分数$$S_n=S/\sqrt{d_k}$$

3.用SoftMax函数将分数转化为概率$$P=softmax(S_n)$$

4.获得加权的value矩阵$$Z=V\cdot P$$&#x20;

这一过程可以用一个公式表示：&#x20;

![](<../../../.gitbook/assets/image (499).png>)

上式的逻辑很简单。第一步计算不同输入之间的分数，该分数提供了在当前位置编码单词时需要给予其他单词的注意度。第二步归一化分数，来加强梯度的稳定性，促进训练。第三部将分数转换为概率。最后，每个value向量用概率加权求和。具有较大概率的向量可以从之后的层中获得额外的关注。&#x20;

Decoder模块中的encoder-decoder注意层与encoder模块中的self-attention层相似，但有以下几点不同：key矩阵K和value矩阵V是从编码器模块中派生出来的，query矩阵Q是从上一层派生出来的。&#x20;

注意到之前的步骤和单词的位置无关，说明self-attention层缺乏捕获单词位置的能力。为了解决这一问题，并获得单词的最后输入向量，一个$$d_{model}$$维的位置编码被加入原本的输入中。特别的，位置通过下式编码：&#x20;

![](<../../../.gitbook/assets/image (49).png>)

其中，pos表示单词在句子中的位置，i表示位置编码的当前维度。

#### 1.2.Multi-Head Attention

![](<../../../.gitbook/assets/image (1075).png>)

Multi-head attention是一种可以用于促进原始self-attention层性能的机制。要注意对于一个给定的参考单词，我们在阅读句子时通常会关注其他几个单词。一个single-head self-attention层限制了我们在不影响其他同样重要位置的注意力时同时考虑一个或多个特定位置的能力。这一问题可以通过赋予attention层不同的表征子控件来解决。更确切的说，在不同head中使用不同的query、key和value矩阵，在从随机初始化开始训练后，这些矩阵可以将输入映射到不同的表征子空间中。&#x20;

给定一个输入向量，和head数量h，输入向量先被转化为三个不同组的向量，query组，key组和value组。每个组中，有h个维度为$$d_{q'}=d_{k'}=d_{v'}=d_{model}/h=64$$的向量。这些向量从不同的输入中衍生出来，然后聚合在一起，构成三组不同的矩阵：$${\{Q_i\}}^h_{i=1},{\{K_i\}}^h_{i=1},{\{V_i\}}^h_{i=1}$$.multi-head注意力过程为：&#x20;

![](<../../../.gitbook/assets/image (241).png>)

其中Q'是由$${\{Q_i\}}^h_{i=1}$$拼接而成的。$$W^{o}\in \mathbb{R}^{d_{model} \times d_{model}}$$是一个线性映射矩阵。

#### 1.3.Other Key Concepts in Transformer

**1.3.1.Residual Connection in the Encoder and Decoder**

![](<../../../.gitbook/assets/image (848).png>)

如上图所示，在encoder和decoder的每一个子层中加入residual connection。这增强了信息的流动，来提升表现。在redisual connection后加入了layer-normalization。这些操作的输出可以写为：&#x20;

![](<../../../.gitbook/assets/image (827).png>)

其中X为self-attention的输入，query、key和value矩阵Q、K和V都是从相同的输入矩阵X中衍生出的。

**1.3.2.Feed-Forward Network**

每个encoder和decoder中在self-attention层后加入一个feed-forward network（FFN）。他包括两个线性transformer层和它们之间的一个非线性激活函数：&#x20;

![](<../../../.gitbook/assets/image (143).png>)

其中$$W_1,W_2$$是两个线性transformer层的参数矩阵，$$\sigma$$是非线性激活函数，比如GELU。隐层的维度是$$d_h=2048$$维。

**1.3.3.Final Layer in the Decoder**

decoder中的最后一层被用于将一堆向量转换回单词。这是通过一个线性层加一个softmax层来实现的。线性层将logits向量映射到一个$$d_{word}$$维的向量，其中$$d_{word}$$是词典中单词的数量。softmax层用于将logits向量转换为概率。&#x20;

当被用于视觉任务时，大多数transformer采用了原本transformer的encoder模块。这种transformer可以看做是一个新的特征提取器。与只关注局部特性的CNNs相比，transformer可以获取长距离的特性。与隐层状态必须循序地计算的RNNs不同，transformer层中self-attention和全连接层的输出可以并行计算并且易于加速，所以更加高效。

### 2.Revisiting Transformers for NLP

![](<../../../.gitbook/assets/image (517).png>)

### 3.Visual Transformer

#### 3.1.Backbone for Image Classification

![](<../../../.gitbook/assets/image (808).png>)

"_Visual transformers: Token-based image representation and processing for computer vision_"中作者利用ResNet作为baseline，用visual transformer代替了卷积的最后一步，更确切的说，他们用卷积层来提取低层的特征，然后将特征输入一个visual transformer。在visual transformer中，作者利用一个tokenizer来将像素聚合成分量的visual tokens，每个token代表一个图像中的小语义信息。transformer用于构建tokens之间的关系，这些语义信息被直接用来图像分类。 而直接用transformer来做图像分类的工作有iGPT，ViT和DeiT。

**3.1.1.iGPT**

![](<../../../.gitbook/assets/image (862).png>)

"Generative pretraining from pixels"&#x20;

这部分没了解过，所以没看懂。大概是训练过程pre-training和fine-tuning两个阶段，在pre-training阶段，使用两种loss（auto-regressive和BERT）来约束训练。在fine-tuning阶段，加入classification head。

**3.1.2.ViT**

![](<../../../.gitbook/assets/image (334).png>)

"_An image is worth 16x16 words: Transformers for image recognition at scale_"&#x20;

这篇论文，作者尽可能的采用transformer原有的结构，将图像分割成图像块序列，输入模型中，实现图像分类。在该模型中，$$H\times W \times C$$的图像被划分为一串展平的图像块$$x_P\in \mathbb{R}^{N\times (P^2 \cdot C)}$$序列。使用一个可训练的线性映射函数来将展平的图像块映射到固定维度D的向量中，被称为patch embeddings。与BERT中的class token类似，该模型使用了一个可学习的embedding，嵌入到embedding patches的序列中。用embedding来表征图像。在pre-training和fine-tuning阶段，classification head被设置为固定尺寸。除此之外，1维的position embedding也加入了patch embeddings中来保存位置信息。所有的embeddings联合在一起，作为encoder的输入。ViT只使用了标准transformer的encoder层，输出接一个MLP head。

在预训练时，ViT在大数据集上训练，然后在小任务上fine-tune。fine-tuning时，去掉原本的head，加入一个被全部初始化为0的head。在fine-tuning阶段使用比pre-training阶段更高的分辨率通常是有益的。例如，当提供更高分辨率的图像时，可以获得更大的有效序列长度，即使patch大小保持不变。尽管ViT可以处理任意长度的序列，但预先训练的位置嵌入可能不再有意义。因此，作者根据pre-trained position embeddings在原图中的位置进行2D插值。值得注意的是，只有在分辨率调整和图像块提取时，关于图像2D结构的归纳偏差是人工引入ViT的。&#x20;

transformer在中等规模数据集上表现与CNNs相近。由于transformers缺少一些CNNs固有的感知偏差——如平移不变性和局部性，所以当训练数据不够多时，它们的泛化性不好。在大规模数据集上，transformer表现得更好。&#x20;

![](<../../../.gitbook/assets/image (12) (1).png>)

**3.1.3.DeiT**

"_Training data-efficient image transformers & distillation through attention_"&#x20;

这篇论文中，作者提出一种competitive convolution-free transformer，称为Data-efficient image transformer（DeiT），只需在ImageNet上训练。DeiT-B模型与ViT模型结构一样，有86M参数。除此之外，作者发现用一个CNN teacher获得比用transformer更好的表现（知识蒸馏）。

**3.1.4.Conclusion**

iGPT回顾了生成性的预训练方法，并将其与自我监督方法相结合，但效果并不理想。ViT获得了更好的结果，特别是在使用更大规模的数据集(JFT-300M)的情况下。同样，DeiT通过更细致的训练策略和基于token的蒸馏实现了更好的性能。考虑到ViT的结构与NLP中transfromer的结果很相似，如何显式识别intra-patch和inter-patch的相关性成为了一个问题。另外，尽管ViT对相同大小的patch进行的处理相同，但每个patch的复杂度是不同的。到目前为止，这一特点还没有得到充分利用。

#### 3.2.High/Mid-level Vision

这一部分的任务包括目标检测、车道线检测、分割和姿态估计等高/中层视觉任务。

**3.2.1.Generic Object Detection**

基于transformer的目标检测方法可以大概的分为两类：transformer-based set prediction methods和transformer-based backbone methods。

![](<../../../.gitbook/assets/image (560).png>)

**3.2.1.1.transformer-based set prediction methods**

![](<../../../.gitbook/assets/image (149).png>)

作为基于transformer的检测方法的开创者，detection transformer（DETR）重新设计了目标检测的框架。DETR，是一个简单的端到端的目标检测器，它将目标检测任务看做一个直观的set prediction问题，避免了传统人工设计的部件，如anchor generation和non-maximum suppression（NMS）。如图7所示，DETR用一个CNN backbone从输入图像中提取特征。为了给图像特征补充位置信息，固定的位置编码被加入到展平后的特征上，然后特征输入encoder-decoder transformer。decoder使用来自encoder的embeddings以及N个学习到的位置编码（object queries），然后产生N个输出的embeddings。其中N为一个提前设定的参数，一般需大于图像中的目标个数。用一个简单的FFN来计算最后的预测，包bounding box的坐标和指示目标类别的标签。不同于传统transformer循序地计算预测值，DETR并行的解码N个目标。DETR使用一种bipartite匹配算法来分配预测的和真值目标。对于所有目标的匹配对，计算Hungarian损失：&#x20;

![](<../../../.gitbook/assets/image (542).png>)

其中y和$$\hat{y}$$分别是真值和预测的目标。$$\hat{\sigma}$$为最优分配，$$c_i$$和$${\hat{p}}_{\hat{\sigma}(i)}(c_i)$$是目标类别标签和预测的标签，$$b_i$$和$${\hat{b}}_{\hat{\sigma}}(i)$$为真值和预测的bounding box。&#x20;

虽然DETR在COCO效果很好，但是需要很长的训练时间，对于小目标检测表现较差。Deformable DETR (_Deformable DETR: Deformable transformers for end-to-end object detection_)利用deformable attention模块关注reference point周围的一小部分关键位置，而不是像transformer中原来的multi-head attention机制那样关注图像特征图上的所有空间位置。这一方法减少了计算复杂度，并且加快了收敛。更重要的是，deformable attention模块可以很容易的用于融合多尺度特征。Deformable DETR比DETR快1.6倍，训练消耗少10倍，且效果更好。采用迭代的bounding box微调方法和two-stage方案，可进一步提高Deformable DETR的检测性能。&#x20;

ACT（_End-to-end object detection with adaptive clustering transformer_）利用Adaptive Clustering Transformer （ACT）来减少预训练DETR的计算消耗，无需任何训练步骤。还有TSP-FCOS、TSP-RCNN（_Rethinking transformer-based set prediction for object detection_）被提出，用特征金字塔来改进encoder-only的DETR。UP-DETR（_UP-DETR: unsupervised pre-training for object detection with transformers_）使用无监督的预训练策略，获得了更好的表现。

**3.2.1.2.Transformer-based Backbone for Detection**

（_Toward transformer-based object detection_）中用transformer作为一般目标检测框架的backbone。输入图像被划分为图像块，输入vision transformer，输出的embeddings根据空间信息被重新组合，再输入detection head来获得最终结果。&#x20;

![](<../../../.gitbook/assets/image (528).png>)

**3.2.2.Other Detection Tasks**

**3.2.2.1.Pedestrian Detection**

由于在遮挡和拥挤场景中物体的分布非常密集，在将普通的检测网络应用于行人检测任务时，往往需要额外的分析和适应。（_DETR for pedestrian detection_）中分析了当直接在行人检测任务中应用DETR或Deformable DETR时，sparse uniform queries和decoder中的弱注意域会导致性能下降。为了缓解这些问题，作者提出了Pedestrian End-to-end Detector (PED)，它采用了一种新的解码器，叫做Dense Queries and Rectified Attention field (DQRF)来支持dense queries，缓解了queries的噪声或狭窄注意域。他们还提出了V-Match，它通过充分利用可见的注释来实现额外的性能改进。

**3.2.2.2.Lane Detection**

（_End-to-end lane shape prediction with transformers_）中提出LSTR，来用transformer提供的全局上下文信息来提升曲线车道线检测的表现。LSTR将车道线检测认为是一个用多项式拟合车道线的任务，用神经网络去估计多项式的参数。除此之外，作者还用了Hungarian损失来优化网络参数。

**3.2.2.3. Segmentation**

DETR可以很自然的拓展到全景分割任务上。(_Max-deeplab: End-to-end panoptic segmentation with mask transformers_)中提出的Max-DeepLab通过一个mask transformer直接估计全景分割结果，不涉及如box detection等代理子任务。与DETR相似，该模型用end-to-end的方法完成全景分割任务，直接预测一组不重叠的masks和相应label。模型用panoptic quality style loss训练。该模型采用了一种dual-path框架，促进了CNN和transformer的结合。VisTR(_End-to-end video instance segmentation with transformers_)从输入图像序列中获得实例预测结果，被用于视频实例分割。Cell-DETR（_Attention-based transformers for instance segmentation of cells in microstructures_）用于在显微镜图像中进行细胞实例分割。SETR（_Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers_）被用于语义分割。还有一些工作利用transformer结构来处理点云信息（_Point transformer，PCT: Point cloud transformer，Point transformer_）。

**3.2.2.4. Pose Estimation**

(_Handtransformer: Non-autoregressive structured modeling for 3d hand pose estimation, Hot-net: Non-autoregressive transformer for 3d hand-object pose estimation_ )用于3D手势估计，（_End-to-end human pose and mesh reconstruction with transformers_）用于从单张RGB图像中恢复人体位姿和mesh。

#### 3.3. Low-level Vision

低层图像任务比如图像超分辨重建和图像生成。&#x20;

（_Image transformer_）用于图像迁移和图像生成任务。&#x20;

（_Learning texture transformer network for image super-resolution_）用于图像超分辨率重建&#x20;

IPT（_Pre-trained image processing transformer_）用大数据集预训练，包含多个heads，一个encoder，一个decoder和多个tails。该模型可以用于多种任务。&#x20;

![](<../../../.gitbook/assets/image (829).png>)

（_Sceneformer: Indoor scene generation with transformers_）用于3D室内场景生成。通过将一个场景视为目标的序列，利用transformer的decoder来预测objects的序列和位置、类别和尺寸。&#x20;

![](<../../../.gitbook/assets/image (30) (1).png>)

综上所述，与分类和检测任务不同，图像生成和处理的输出是图像。上图演示了在low-level视觉任务中使用transformer的一般框架。通过将图像作为像素或图像块的序列，transformer encoder使用该序列作为输入，让transformer decoder生成所需的图像。

#### 3.4.Video Processing

见原文

#### 3.5.Self-attention for Computer Vision

self-attention是transformer的核心部分。

**3.5.1.General Formulation of Self-attention**

机器翻译中的self-attention模块通过关注所有位置并根据嵌入空间中的对应权重将他们求和，来计算序列中一个位置的响应。这可以看做是一种可用于计算机视觉的non-local filtering操作。给定一个输入信号$$X\in\mathbb{R}^{n\times c}$$，其中$$n=H\times W$$代表特征中的像素点数，c是通道数，输出信号为：&#x20;

![](<../../../.gitbook/assets/image (178).png>)

其中$$x_i\in \mathbb{R}^{1\times c}, y_i \in \mathbb{R}^{1\times c}$$表示输入信号X和输出信号Y中的第i个位置。下标j遍历了全部位置。一个pairwise的函数$$f(\cdot)$$计算出i和所有j之间的表征相关性。$$g(\cdot)$$计算了输入信号在j处的表征。响应被$$C(x_i)$$归一化。 要注意pairwise函数$$f(\cdot)$$有很多选择，比如：&#x20;

![](<../../../.gitbook/assets/image (27) (1).png>)

其中$$\theta(\cdot), \phi(\cdot)$$可以是任何embedding层。如果我们假设$$\theta(\cdot), \phi(\cdot), g(\cdot)$$都是线性embedding：$$\theta(X)=XW_{\theta}, \phi(X)=XW_{\phi}, g(X)=XW_{g}$$，其中

![](<../../../.gitbook/assets/image (370).png>)

&#x20;并且令归一化因子为：&#x20;

![](<../../../.gitbook/assets/image (1063).png>)

那么公式16可以重写为：（**这公式没推对吧...**）&#x20;

![](<../../../.gitbook/assets/image (1047).png>)

其中$$w_{\theta, i}\in\mathbb{R}^{c\times 1}$$是权重矩阵$$W_{\theta}$$的第i行，对于i，$$\frac{f(x_i,x_j)}{C(x_i)}$$是一个沿着维度j的softmax。因此，公式可以重写为：&#x20;

![](<../../../.gitbook/assets/image (345).png>)

其中$$Y\in\mathbb{R}^{n \times c}$$是一个和输入信号X有相同尺寸的输出信号。与机器翻译模块中的query、key和value表征相比，如果令$$W_q=W_{\theta},W_k=W_{\phi},W_v=W_g$$，那么公式19可以写为：&#x20;

![](<../../../.gitbook/assets/image (59).png>)

用于机器翻译的self-attention模块在一定程度上与前面提出的用于计算机视觉的non-local filtering操作相同。 一般地，用于计算机视觉的self-attention模块的最终输出信号会被转换为：&#x20;

![](<../../../.gitbook/assets/image (567).png>)

其中Y通过公式19得到。如果$$W_z$$初始化为0，这个self-attention模块可以插入任何现存的模型中，不会破坏它的最初行为。

**3.5.2. Applications on Visual Tasks**

见原文

#### 3.6. Efficient Transformer

![](<../../../.gitbook/assets/image (817).png>)
