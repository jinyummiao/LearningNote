# feature matcher



<table><thead><tr><th width="150">Method</th><th width="150">from</th><th width="447.2">innovation</th></tr></thead><tbody><tr><td>SuperGlue</td><td>CVPR 2020</td><td>利用self-attention和cross-attention与GNN相结合获得每个keypoint的更稳健的matching descriptor，用optimal transport problem的方法(Skinhorn算法)来获得partial assignment，效果超赞！</td></tr><tr><td>LoFTR</td><td>CVPR 2021</td><td>提出基于Transformer的LoFTR模块来将特征转换为更易于匹配的特征，利用1/8分辨率的特征进行粗略匹配，利用1/2分辨率的特征对粗略匹配进行微调。可以在无纹理区域也获得稠密的匹配。</td></tr><tr><td>AdaLAM</td><td>ECCV 2020</td><td>一种基于传统方法的特征筛选方法，基于局部单应性假设，先选择seed point，再在每个seed周围选择邻域内的correspondence，在并行RANSAC过程中不是根据投影误差来选择内点， 而是基于统计学显著性上自适应的设置阈值，筛选内点匹配。</td></tr><tr><td>Patch2Pix</td><td>CVPR 2021</td><td>首先预测patch-level match proposal，然后对每个patch-level match proposal定义的区域中拟合pixel-level matches，同时根据可信度分数来剔除外点。Patch2Pix采用弱监督方法训练，无需真值的匹配，去学习图像间符合极线约束的correspondence，用Sampson距离度量匹配符合极限约束的程度，作为损失函数</td></tr></tbody></table>
