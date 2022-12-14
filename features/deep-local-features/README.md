# deep local features

| 算法         | 来源           | 主要创新点                                                                                                                                                                 |
| ---------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MagicPoint | arxiv 2017   | 该论文用一个FCN去预测图像中中可能的关键点，将heatmap输入第二个FCN预测两幅图像间的H矩阵                                                                                                                    |
| SuperPoint | CVPRW 2018   | 分步训练，在训练好MagicPoint的基础上，用homographic adaption方法增广真实数据，用真实数据训练detector和descriptor                                                                                      |
| D2-Net     | CVPR 2019    | detect-and-describe，得到dense的feature map后，检测在通道维度最大并且局部最大的点作为关键点                                                                                                       |
| R2D2       | NeurIPS 2019 | 将特征的repeatability和reliability分开训练                                                                                                                                     |
| SEKD       | arxiv 2020   | self-evolving training，定义了detector和descriptor的repeatability(inherent)和reliability(interactive)性质，并以此迭代训练                                                              |
| ASLFeat    | CVPR 2020    | 对D2-Net进行了改进，用DCN估计local shape信息并在DCN中引入几何约束，用一种新的multi-scale detection策略来保证关键点检测的准确性，用peakness代替D2-Net的ratio-to-max度量channel-wise extremeness                        |
| MDA        | ICCV 2021    | 提出一种局部特征，同时关注单张图像中多个有区分度的局部模式。在该模型中，作者首先自适应地调整了attention map的通道，得到多个group。对每个group，设计了新的动态注意力模块来得到attention map，并通过diveristy regularization来让不同的attention map关注于不同的模式 |
| DISK       | NeurIPS 2020 | 使用UNet结构，使用深度强化学习从特征匹配的角度训练网络。将特征提取及匹配过程用概率模型定义，根据特征匹配情况定义匹配的回报值，用策略梯度进行端到端训练。                                                                                        |
| DELG       | ECCV 2020    | 用一个backbone提取深层特征D和浅层特征S，S输入注意力模块，根据注意力图提取局部特征；D输入GeM层，得到全局特征。全局特征训练时，用ArcFace margin+交叉熵损失；局部特征训练时，用特征恢复的MSE损失+注意力pooling后的交叉熵损失。只用全局特征的损失来优化backbone。               |
