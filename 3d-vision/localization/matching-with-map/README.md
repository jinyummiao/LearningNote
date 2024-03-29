---
description: map-based visual re-localization
---

# matching with map

### 论文总结

<table><thead><tr><th width="162.78353745371686">算法</th><th width="150">来源</th><th width="402.2">主要贡献</th></tr></thead><tbody><tr><td>NRE</td><td>CVPR 2021</td><td>提出重投影误差的一种替代形式-NRE，相比2D-3D匹配，可以保留更多的信息，减少传递给位姿估计的错误信息，无需选择鲁棒损失和参数调整。NRE将特征学习与位姿估计结合起来。作者还提出一种由粗到精的优化方法，来根据NRE训练位姿估计。</td></tr><tr><td>PixLoc</td><td>CVPR 2021</td><td>该算法输入query图像和3D模型，输出图像对应的位姿。PixLoc训练了一个CNN模型特征，通过直接对齐多尺度CNN特征，来利用LM法迭代优化初始位姿。初始位姿可由图像检索获得。</td></tr><tr><td>Pole Map</td><td>ICCR 2020</td><td>用场景中的杆状物体进行定位，利用地面载体的几何约束，将平移和旋转去耦，利用例子滤波进行定位，优势是所需的语义地图较小（因为只需要保存杆的信息），计算速度快</td></tr><tr><td>X-View</td><td>RAL 2018</td><td>提取图像语义块作为拓扑图节点，根据3D空间中的相邻关系定义边。利用random walk描述子描述拓扑图，匹配图像。</td></tr><tr><td>CMRNet</td><td>ITSC 2021</td><td>利用光流网络PWCNet，输入为RGB图像（query）和Lidar地图投影到某一初始位姿下的深度图，网络直接预测query图像与深度图之间的相对位姿，根据相对位姿和初始位姿得到query图像的绝对位姿。可迭代优化。</td></tr><tr><td>CMRNet++</td><td>ICRAW 2020</td><td>利用光流网络PWCNet，输入为RGB图像（query）和Lidar地图投影到某一初始位姿下的深度图，网络直接预测点云和RGB图像的匹配，再利用PnP+RANSAC求解位姿。可迭代优化。</td></tr><tr><td>HyperMap</td><td> ICRA 2021</td><td>在CMRNet的基础上，先对Lidar地图进行栅格化，对栅格化的地图进行3D特征提取，用Kmeans将栅格特征转换为id值。在地图投影得到深度图的过程中，只对id map进行投影，减少计算量，然后将id值转换成特征(late projection)。输入CMRNet的是query图像和地图投影的特征图，其他流程与CMRNet类似，采用了新的遮挡处理方法。</td></tr><tr><td>BVMatch</td><td>RAL 2021</td><td>在BEV视角下，对稀疏3D点云进行栅格化，基于栅格内点云的数量（density）构建BV image。基于Log-Gabor filter构建MIM，以此设计具有旋转不变性的局部特征描述子BVFT，用FAST检测关键点。用K-means构建词典，实现场景识别。同时可以对两个BV image进行相对位姿计算（三个自由度），实现定位。</td></tr><tr><td>TM3Loc</td><td>TITS 2022</td><td></td></tr></tbody></table>
