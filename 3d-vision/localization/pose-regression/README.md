---
description: image-based camera pose regression
---

# pose regression

### 论文总结

| 算法                     | 来源          | 主要贡献                                                                                                                                                                               |
| ---------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PoseNet                | ICCV 2015   | 用一个神经网络直接预测输入图像的6DOF pose(3维位置+4维四元数表示的方向)。欧氏距离直接拟合，用SfM获得真值pose                                                                                                                   |
| CaTiLoc                | ICASSP 2021 | 用MobileNet提取特征图，输入ViT，输出接全连接层，估计3D位置和4D四元数信息                                                                                                                                       |
| HourglassNet           | ICCVW 2017  | 采用一个沙漏型的网络结构来尽可能保留图像的细节信息，从RGB图像直接预测图像的位姿                                                                                                                                          |
| PoseNet2               | CVPR 2017   | 在PoseNet的基础上，提出了两种新的损失函数，一种使用自动优化平衡参数的L1距离，一种使用重投影误差                                                                                                                               |
| AtLoc                  | AAAI 2020   | 利用ResNet得到特征向量后，加入一个non-local self-attention模块和残差连接，让特征更关注图像稳定、几何鲁棒的区域                                                                                                             |
| Local Support Global   | ICCV 2019   | 基于序列图像，估计图像的全局位姿，和相邻帧之间的局部相对位姿，联合优化。加入了位姿图优化，在测试时进一步减小误差                                                                                                                           |
| MS-Transformer         | ICCV 2021   | 使用两个彼此独立的transformer来分别编码与位置和旋转有关的信息，分别预测位置和旋转。提出一种单一模型多场景位姿回归框架，输出当前图像在不同场景中的可能状态，挑选当前场景下的状态用于位姿回归，用位姿回归和分类两种监督同时训练                                                               |
| MultiScene PoseNet     | CVPRW 2020  | 对每个场景训练scene-specific heads来回归位姿，将位姿回归与场景分类同时训练，实现多场景绝对位姿估计                                                                                                                        |
| MapNet                 | CVPR 2018   | 提出用单位四元数的对数来表示旋转；用VO提供的相对位姿来进一步优化模型，无需额外的数据标注，推理时加入位姿图优化，优化预测的绝对位姿                                                                                                                 |
| Auxiliary Colorization | arxiv 2021  | 利用着色任务作为定位任务的辅助任务，一起训练。结合定位子网络和着色子网络获得的特征，用注意力机制筛选高响应区域，进行位姿回归                                                                                                                     |
| How to improve         | ICCVW 2019  | 作者从数据角度提出了三个提升posenet表现得方法，1.使用全视野图像而非裁剪后的图像，2.使用数据增强缓解过拟合，3.使用LSTM拟合位姿                                                                                                            |
| Anchor Point           | BMVC 2018   | 作者在场景中设定了多个anchor point，估计当前图像相对anchor point的X、Y坐标偏移和Z轴全局坐标、旋转角度，训练网络自动寻找最相关的anchor point                                                                                          |
| Bayesian PoseNet       | ICRA 2016   | 作者对PoseNet的其中几层加入dropout来实现伯努利贝叶斯卷积神经网络的变分推理，通过对模型参数的后验分布进行蒙塔卡罗采样，获得若干样本，以样本均值为估计的位姿，以样本的协方差矩阵的迹作为估计的不确定性，该不确定性可以帮助提升模型效果                                                          |
| X-View                 | RAL 2018    | 提取图像语义块作为拓扑图节点，根据3D空间中的相邻关系定义边。利用random walk描述子描述拓扑图，匹配图像。                                                                                                                         |
| GN-Net                 | RAL 2020    | 用Gauss-Newton loss训练的具有天气不变性的特征，适用于直接法图像对齐。网络用来自不同序列的图像间的像素对应关系来训练。提出了一个用于在不同季节下测试relocalization tracking性能（就是计算query和keyframe之间相对位姿）的benchmark。                                   |
| deepFEPE               | IROS 2020   | SuperPoint+softargmax detector+deepF，对估计的F矩阵和F分解后得到的位姿进行约束，实现端到端训练。可以直接估计两幅图像间的相对位姿。                                                                                               |
| E2E RPE                | arxiv 2021  | SuperPoint+correlation layer+matching layer+softargmax detector。网络输出了匹配点对和匹配的权重，可以直接嵌入PNP中计算相对位姿。关键点检测部分拟合数据增强后的SuperPoint关键点；位姿估计部分包含DSAC损失函数和内点数损失函数。                            |
| DeepF                  | ECCV 2018   | 使用端到端的神经网络实现根据输入匹配点的坐标得到F矩阵的功能。算法基于带权重的最小1迭代优化算法，分别model estimator和weight estimator。model estimator采用PointNet结构，根据输入点和权重求解模型，weight estimator采用PointNet结构，重新对匹配赋予权重，迭代优化，得到最后的F矩阵。 |
| Direct PoseNet         | 3DV 2021    |                                                                                                                                                                                    |
