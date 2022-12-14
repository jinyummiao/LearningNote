---
description: >-
  Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place
  Recognition
---

# \[CVPR 2021] Patch-NetVLAD

这篇论文中，作者用original NetVLAD检索出top-K个候选相似图像后，利用patch-level NetVLAD descriptor进行了spatial score的计算，对候选图像进行了挑选和重排。作者利用patch作为局部区域，提取NetVLAD描述子，进行patch之间的匹配，用匹配分数作为spatial score。作者还提出融合多尺度patch的匹配分数，提升算法表现，利用integral VLAD特征图的技术避免了重复计算不同尺度的VLAD描述子。在实时性和检索精度上都获得了很好的表现，获得了ECCV2020 Facebook Mapillary Visual Place Recognition Challenge的冠军。
