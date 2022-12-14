---
description: 'NetVLAD: CNN architecture for weakly supervised place recognition'
---

# \[CVPR 2016] NetVLAD

这篇经典的利用DL的PR论文中，作者受VLAD全局描述子的启发，将其转化为一个可微的CNN层，设计了一个可以end-to-end训练的全局描述子，并用谷歌街景获取triplet训练数据，用triplet ranking loss进行训练。构思非常巧妙，效果很好，算是里程碑式的一个工作吧。
