---
description: 'SuperGlue: Learning Feature Matching with Graph Neural Networks'
---

# \[CVPR 2020] SuperGlue

{% embed url="https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf" %}

{% embed url="https://github.com/magicleap/SuperGluePretrainedNetwork" %}

{% embed url="https://blog.csdn.net/shizhuoduao/article/details/107120805" %}

### Abstract

本文提出的神经网络模型SuperGlue可以对两组特征进行匹配，得到correspondences并且丢弃未匹配到的点。这一分配问题是通过求解一个可微分的最优运输问题来估计的，该问题的损失是用图神经网络来预测的。我们引入了一种基于注意力的上下文聚合机制，使SuperGlue能够推理出潜在的3D场景和特征分配。与传统人工设计的启发式方法相比，我们的方法通过对图像对进行端到端的训练，学习三维世界的几何变换和规律。

### Introduction

![](<../../.gitbook/assets/image (877).png>)

这篇论文中，作者关注的是特征匹配问题，作者将特征匹配任务视为对两组局部特征寻找partial assignment的问题。通过求解线性分配问题，作者讨论了经典的基于图的匹配策略，该策略在放宽到最优运输问题时，可以可微求解。这一优化过程的损失函数用GNN来估计。受Transformer的启发，模型用self-attention（intra-image）和cross-attention（inter-image）来平衡关键点的空间关系和它们的视觉外观信息。这一范式加强了预测的分配结构，同时使损失函数能够学习到复杂的先验，可以很好地处理遮挡和不可重复的关键点。该模型可以用图像对进行end-to-end训练。

### The SuperGlue Architecture

整个网络由两个主要模块构成：Attentional GNN和Optimal Matching Layer。其中AGNN将特征点位置和描述子编码成一个向量（特征匹配用的向量），随后利用self-attention和cross-attention来增强（L次）；进入OML后，通过计算特征匹配向量的内积来得到匹配度得分矩阵，然后通过sinkhorn算法（迭代T次）解算出最优特征分配矩阵。

**Motivation**

简单来说，就是2D关键点是静态3D点的投影，类似角点等，这些图像中的点都会符合图像的同一种变化关系，符合一些物理限制：

1. 一个关键点在其他图像中最多只能有一个匹配点；
2. 一些关键点可能由于遮挡或没有被检测到而没有匹配点。

作者提出SuperGlue就是为了找到相同3D投影点之间的correspondence，并且识别出没有匹配的关键点。而这其中的规律是直接从数据中学习得到的，无需相关经验和启发式算法。&#x20;

![](<../../.gitbook/assets/image (1027).png>)

**Formulation**

设图像A和B，每个图像都包含着一组关键点位置p和与其相关联的视觉描述子d，作者将(p,d)称为局部特征。关键点位置包括x，y坐标和检测可信度c，$$p_i:={(x,y,c)}_i$$。视觉描述子$$d_i\in\mathbb{R}^D$$可以是从CNN中提取的特征，如SuperPoint，也可以是传统的描述子，如SIFT。A和B各有M和N个局部特征。

**Partial Assignment**

前文提到的两个约束说明correspondence来自对两组keypoint的partial assignment。为了集成到下流任务并且为了更好的可解释性，每个correspondence都应具有一个可信度。因此，作者定义了一个partial soft assignment matrix $$P\in{[0,1]}^{M\times N}$$（**即P中每行和每列都最多只有一个1，即一个匹配**）:&#x20;

![](<../../.gitbook/assets/image (26) (1).png>)

本文的目标就是设计一个神经网络，可以根据两组局部特征，来预测P。

#### Attentional Graph Neural Network

作者利用Attentional Graph Neural Network来作为SuperGlue的第一个主模块。当给定初始的局部特征，它通过让特征与其他特征相互作用来计算匹配描述子

**Keypoint Encoder**

每个keypoint最初的表征$${}^{(0)}x_i$$结合了它的visual appearance和位置，作者利用Multilayer Perception (MLP)来讲位置信息嵌入到一个高维向量中：&#x20;

![](<../../.gitbook/assets/image (201).png>)

这种encoder使图网络能够在以后同时对外观和位置进行推理，特别是当与注意力结合在一起时。

#### Multiplex Graph Neural Network

作者构建了一个完整的图，图的节点是两幅图像中的所有关键点，图中有两种无向边，称为多元图。Intra-image edges，或者self edges，$$\varepsilon_{self}$$，连接着同一幅图像中的keypoints。Inter-image edges，或者cross edges，$$\varepsilon_{cross}$$，将keypoint i与另一幅图像中的keypoints连接起来。通过两种边来传递信息，通过聚合边传来的信息，GNN可以根据每个节点的高维状态，在每一层计算节点的更新表征。&#x20;

令$${}^{(l)}x_i^{A}$$为图像A的元素i在l层的中间表征。信息$$m_{\varepsilon\rightarrow i}$$是从所有关键点$${j: (i,j)\in \varepsilon}$$聚合来的结果，其中$$\varepsilon\in {\varepsilon_{self}, \varepsilon_{cross}}$$。A中所有元素i传递更新的残差信息为：&#x20;

![](<../../.gitbook/assets/image (1001).png>)

其中，$$[x || x]$$表示concatenation。同时，图像B上的所有关键点也有相似的更新过程。具有不同参数的L层串联在一起，交替聚合self edge和cross edge。照此规则，从l=1开始，如果l为奇数，则$$\varepsilon=\varepsilon_{self}$$，否则当l为偶数时，$$\varepsilon=\varepsilon_{cross}$$。

#### Attentional Aggregation

利用注意力机制来完成聚合和计算信息$$m_{\varepsilon\rightarrow i}$$。self edges基于self-attention，而cross edges基于cross-attention。与数据检索类似，i的一个表征，query $$q_i$$，根据某些元素的性质（key $$k_j$$）检索它们的value $$v_j$$。信息通过计算这些value的加权平均获得：&#x20;

![](<../../.gitbook/assets/image (23) (1).png>)

其中，注意力权重$$\alpha_{i,j}$$是在key-query之间相似度上取SoftMax获得的：$$\alpha_{i,j}={Softmax}_j({q_i}^Tk_j)$$. key、query和value通过对图神经网络的深层特征进行线性投影计算获得。令query keypoint i在图像Q上，所有source keypoints在图像S上，$$(Q,S)\in {\{A,B\}}^2$$，则有：&#x20;

![](<../../.gitbook/assets/image (545).png>)

每一层都用各自的投影参数，这些参数被图像中所有keypoints共享。在实践中，作者通过multi-head attention来提升表达性。（**这里的意思应该就是q就是图像Q中要检索匹配点的一个点的一种高维表征，k和v是图像S中某个点的一种表征，**$$\alpha_{i,j}$$**是特征i和j的相似度，通过**$$q_i$$**和**$$k_j$$**来计算，越大表示两个特征越相似，利用这种相似性做权重，来对**$$v_j$$**加权就平均得到边上的信息**$$m_{\varepsilon\rightarrow i}$$**，实现特征的聚合**）&#x20;

![](<../../.gitbook/assets/image (514).png>)

该模型提供了最大的灵活性，因为网络可以学习关注基于特定属性的关键点子集（如上图）。SuperGlue会基于视觉信息和关键点位置来检索和关注，因此模型可以关注附近点，也会关注相似或显著的点。最终匹配描述子为线性映射：&#x20;

![](<../../.gitbook/assets/image (28) (1).png>)

B中关键点也一样。

### Optimal matching layer

SuperGlue的第二个主要模块为optimal matching layer，该模块输出partial assignment matrix。与标准的图匹配模型一样，assignment P可以通过对所有可能的匹配计算分数矩阵$$S\in \mathbb{R}^{M\times N}$$然后在公式1的约束下最大化总体分数$$\sum_{i,j}S_{i,j}P_{i,j}$$来获得。这相当于解决一个线性分配问题。

#### Score Prediction

不应当为所有$$M\times N$$个可能的匹配构建独立的表征。作者用匹配描述子的相似度（内积）来表示pairwise score：&#x20;

![](<../../.gitbook/assets/image (670).png>)

与视觉描述符不同，匹配描述子没有归一化，它们的大小可以在每个特征和训练过程中变化，以反映预测的置信度。

**Occlusion and Visibility**

为了让模型抑制一些关键点，我们在每个集合中增加一个dustbin，这样就可以显式地将不匹配的关键点分配给它。作者通过添加一个新的行和列，即point-to-bin和bin-to-bin分数，将分数S扩张为$$\overline{S}$$，并用一个可学习的参数填充:&#x20;

![](<../../.gitbook/assets/image (1044).png>)

图像A中的一个特征点被分配到B中的一个点或者被分配到dustbin，这也意味着，dustbin有与另一个集合中keypoints一样多的匹配：A中dustbin有N个，B中有M个。而扩张后的$$\overline{P}$$有如下约束：&#x20;

![](<../../.gitbook/assets/image (996).png>)

其中，$$a={[1_M^T~N]}^T, b={[1_N^T~M]}^T$$

**Sinkhorn Algorithm**

上述问题的优化可以视为两个离散分布a和b之间基于分数$$\overline{S}$$的最优运输问题，可以用Sinkhorn算法求解，该算法是Hungarian算法的可微形式，常用于bipartite matching，它包括沿着行和列迭代normalize $$exp(\overline{S})$$，类似于行和列上的Softmax。经过T次迭代，作者丢弃了dustbin，取$$P={\overline{P}}_{1:M,1:N}$$ 具体原理略，形象理解Sinkhorn算法可以参考[Sinkhorn算法](https://blog.csdn.net/zsfcg/article/details/112510577?utm\_medium=distribute.pc\_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control\&dist\_request\_id=1331645.12678.16184069373174013\&depth\_1-utm\_source=distribute.pc\_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)

#### Loss

GNN和OML都是可微的，所以可以用反向传播来进行训练。SuperGlue采用监督学习的方法，用真值匹配$$\mathcal{M}={(i,j)} \subset \mathcal{A}\times\mathcal{B}$$来训练。真值匹配是根据真值的相对变换估计的，比如使用pose和深度图或homographies。并且还加入了一些没有匹配的点$$\mathcal{I}\subseteq\mathcal{A}, \mathcal{J}\subseteq\mathcal{B}$$给定这些标签，模型需要最小化$\overline{P}$的负log似然值：&#x20;

![](<../../.gitbook/assets/image (1058).png>)

这一监督旨在同时最大化matching的precision和recall。

### Implementation details

所有中间表征（key，query，value，descriptors）的维度与SuperPoint描述子一致，为256维。使用共L=9层交替的multi-head self-attention和cross-attention，attention采用4个head，执行T=100次Sinkhorn迭代。模型参数量为12M，在NVIDIA GTX 1080 GPU上实时性为一次69ms（15FPS）。

### Experiments

#### Homography estimation

对真实图像（Oxford和Paris）用随机homograhies和随机photometric distortion来获得图像对，构成train、validation和test数据集。对所有真值correspondence计算匹配的precision和recall。计算图像四个角的平均重投影误差，并记录最大值为10像素的累积误差曲线(AUC)下的面积。&#x20;

![](<../../.gitbook/assets/image (530).png>)

#### Indoor pose estimation

在ScanNet上训练(230M)、测试(1500)。&#x20;

![](<../../.gitbook/assets/image (700).png>)

#### Outdoor pose estimation

在PhotoTourism数据集上测试，在MegaDepth上训练。&#x20;

![](<../../.gitbook/assets/image (839).png>)

![](<../../.gitbook/assets/image (296).png>)

#### Ablation study

![](<../../.gitbook/assets/image (858).png>)
