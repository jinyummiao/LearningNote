---
description: End-to-End Object Detection with Transformers
---

# \[ECCV 2020] DETR

{% embed url="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf" %}

{% embed url="https://github.com/facebookresearch/detr" %}



### Abstract

作者将目标检测问题视为直接的set prediction问题，并设计了新的检测流程，有效地避免了人工设计的如NMS和anchor generation等组件，这些组件常常暗含了很多人类关于检测任务的先验知识。新框架的主要成分，称为DETR，包括一个基于set的全局损失函数，通过bipartite matching来实现唯一的预测，以及一个基于transformer的encoder-decoder结构。给定固定数量的学习到的object queries，DETR推理出objects和全局图像内容的关系，并直接并行地输出最终预测的set。新模型无需专门的库，并与经过优化的经典算法有同等的准确率和实时性。模型用一个统一的模式可以很容易的拓展到全景分割任务，并且获得不错的表现。

### Introduction

目标检测的目标是预测一组bounding boxes并且对每个目标进行分类。现有算法用一种间接的方法，在很多proposals、anchors或window centers上定义代理的回归任务和分类任务来解决这类set prediction问题。这类方法很容易因为后处理步骤被影响。为了简化这一流程，作者提出一种直接的set prediction方法来绕过代理任务。&#x20;

作者提出一种训练流程，将目标检测视为直接的set prediction问题。作者使用了一个基于transformer的encoder-decoder结构，这种transformer的自注意力机制可以明确地建模序列中所有元素两两之间的相互作用，使得这种结构特别适合于set prediction的特殊限制，比如提出重复的预测。&#x20;

![](<../../../.gitbook/assets/image (52).png>)

本文所提的DETR模型一次性预测所有目标，用一个在预测与真值目标间进行bipartite matching的set损失函数来完成端到端训练。DETR无需很多结合了先验知识的人工组件，如spatial anchor或NMS。因此，可以很容易的结合到任何包括标准ResNet和Transformer的框架中。&#x20;

与现有的直接预测set的工作相比，DETR的主要区别在于结合了bipartite matching和有parallel decoding的transformer（不是自回归的）。之前的工作都利用RNNs实现自回归的decoding。所提出的损失函数可以唯一的将一个预测分配给一个真值目标，并且与预测目标的顺序无关，因此可以并行实现。 从结果上看，DETR在大目标上表现更好，这可能得益于transformer的非局部运算。但是在小目标上效果不佳。

### The DETR model

在检测任务中直接预测set有两个关键的要素：1. 一个可以实现预测与真值目标间唯一匹配的set prediction损失函数；2. 一个可以预测一组目标并建模它们之间关联的结构。&#x20;

![](<../../../.gitbook/assets/image (38).png>)

#### Object detection set prediction loss

DETR通过decoder一次性输出固定数量的N个预测，其中N远大于图像中目标的数量。所提出的损失函数应当可以获得预测与真值目标间的最优bipartite matching，然后优化object-specific（bounding box）的损失。&#x20;

令y是目标的真值集合，$$\hat{y}={\hat{y_i}}^N_{i=1}$$是N个预测。假设N大于图像中的目标数量，把y用$$\varnothing$$填充为大小为N的集合。为了获得两个集合之间的bipartite matching，需要搜索N个元素$$\sigma$$的一种排列，使得损失最小：&#x20;

![](<../../../.gitbook/assets/image (172).png>)

其中$$\mathcal{L}_{match}$$是一个在真值$$y_i$$和第$$\sigma(i)$$个预测之间的pair-wise matching loss。可以用Hungrian算法有效地计算出最优的分配。&#x20;

匹配损失需要同时考虑类别的预测、预测及真值boxes之间的相似度。真值集合中的每个元素i可以看作$$y_i=(c_i,b_i)$$，其中$$c_i$$是目标类别标签（包括$$\varnothing$$），$$b_i\in {[0,1]}^4$$是一个决定真值box中心坐标和高、宽的向量。对于第$$\sigma(i)$$个预测，定义类别$$c_i$$的可能性为$$\hat{p_{\sigma(i)}}(c_i)$$，预测的box为$$\hat{b}_{\sigma(i)}$$.由这些符号，定义$$\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})$$为$$-1_{{c_i\ne \varnothing}}\hat{p_{\sigma(i)}}(c_i)+1_{{c_i\ne \varnothing}} \mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})$$
