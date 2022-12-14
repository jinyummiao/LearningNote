---
description: >-
  CoHOG: A Light-Weight, Compute-Efficient, and Training-Free Visual Place
  Recognition Technique for Changing Environments
---

# \[RAL 2020] CoHOG

计算图像像素的熵，然后进一步计算图像区域的熵，选择大于阈值的区域作为信息丰富的区域，计算区域内的HOG描述子，匹配query与reference图像中region HOG descriptor，得到匹配分数，求平均作为图像匹配分数。
