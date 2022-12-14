---
description: Where is your place, Visual Place Recognition?
---

# \[IJCAI 2021] Visual Place Recognition

一篇综述论文，从agent、environment和tasks三个方向讨论了VPR。

2016年Lowry的TRO综述中定义VPR问题为"given an image of a place, can a human, animal, or robot decide whether or not this image is of place it has already seen?"

这篇论文中，作者根据visual overlapping对VPR做出了新的定义："the ability to recognitize one's localtion based on two observations preceived from overlapping field-f-views."&#x20;

作者指出VPR与image retrieval的区别在于，image retrieval旨在搜索类别相同的相似图像，而VPR旨在搜索相同地点的图像，而非相同类别的图像，相同地点的图像可能视觉相似度并不高。&#x20;

作者提出要根据场景和任务需求来平衡viewpoint-和appearance-invariance。比如室内的UAV，需要更强的viewpoint-invariance，而非appearance-invariance。 在SLAM中使用VPR（比如回环检测），错误的匹配会产生灾难性的建图失败，所以需要很高精度的VPR。&#x20;

对于全局描述子，如果不增加训练数据，提升viewpoint-invariance必定会损失appearance-invariance。
