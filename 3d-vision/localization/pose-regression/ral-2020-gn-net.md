---
description: 'GN-Net: The Gauss-Newton Loss for Multi-Weather Relocalization'
---

# \[RAL 2020] GN-Net

{% embed url="https://ieeexplore.ieee.org/abstract/document/8954808" %}

{% embed url="https://vision.in.tum.de/research/vslam/gn-net" %}

## Abstract

直接法SLAM在里程计任务上表现良好，但是它们容易受到动态光和季节变化的影响，并且在基线较大时初始化结果不好。为了解决这一问题，作者提出GN-Net，用Gauss-Newton loss训练的具有天气不变性的特征，适用于直接法图像对齐。网络可以用来自不同序列的图像间的像素对应关系来训练。实验证明本文方法对于较差的初始化、时间和季节变化更加鲁棒，表现出色。并且，本文提出了一个用于在不同季节下测试重定位性能的benchmark。

## Introduction

直接法无需提取人工设计的稀疏特征，直接根据传感器数据的光度信息来定位和建图，在精度和鲁棒性上比间接法更好。但是直接法存在两个主要的缺陷：1.直接法需要一个较好的初始化，这使得直接法对于大极线跟踪和低频相机不够鲁棒；2.直接法易受光照、天气变化影响。

本文旨在解决剧烈光照和天气变化下的SLAM问题。在本文中，我们将图像转换到具备光照、天气不变性的高维特征上。特征利用一种先进的Gauss-Newton loss来自监督训练。作者根据虚拟数据或SLAM系统标定的数据训练Siamese网络，减少人工标注的成本。作者使用Gauss-Newton算法的概率学形式来直接对齐图像。为此，作者设计了Gauss-Newton loss来最大化正确匹配像素的概率。The proposed loss function thereby enforces a feature representation that is designed to admit a large basin of convergence for the subsequent Gauss- Newton optimization.

在常用的benchmark中，重定位需要两部分，首先找到相近的关键帧，然后跟踪相对位姿。本文关注于第二个步骤，称为relocalization tracking。为了评估不同天气下relocalization tracking的表现，作者提出一个benchmark，具有三个特性：1.包含多个不同天气的图像序列；2.提供了pixel-wise correspondence；3.将relocalization tracking和image retrieval解耦。

本文的贡献在于：

1. We derive the Gauss-Newton loss formulation based on the properties of direct image alignment and demonstrate that it improves the robustness to large baselines and illumination/weather changes.&#x20;
2. Our experimental evaluation shows, that GN-Net outperforms both state-of-the-art direct and indirect SLAM methods on the task of relocalization tracking.&#x20;
3. We release a new evaluation benchmark for the task of relocalization tracking with ground-truth poses. It is collected under dynamic conditions such as illumination changes, and different weathers. Sequences are taken from the the CARLA simulator as well as from the Oxford RobotCar dataset.

## Deep Direct SLAM

为了训练有助于直接法SLAM的特征，神经网络应具备一些特性：同一3D点对应的像素的特征应该是一致的；不同3D点对应的像素特征应该不相似；Gauss-Newton法应该可以收敛到正确的解。

**Architecture** 用U-Net神经网络将图像转换为稠密的特征$$R^{H \times W \times D}$$​。输入一对图像$$I_a$$​和$$I_b$$​，得到多尺度特征金字塔$$F^l_a$$​和$$F^l_b$$​，其中$$l$$表示decoder的第l层。对于每对图像，使用固定数量的匹配$$N_{pos}$$​和固定数量的非匹配$$N_{neg}$$​。

**Pixel-wise contrastive loss** 这个loss旨在最小化匹配点之间的距离，最大化非匹配点间的距离，$$L_{contrastive}$$包含两个部分：

![](<../../../.gitbook/assets/image (386).png>)

其中D表示特征描述子间的欧式距离，M设为1.

**Gauss-Newton algorithm for direct image alignment** 我们的特征最终是要用于位姿估计的。位姿估计采用直接图像对齐的方法，输入一个reference特征图F和图中一些像素的已知深度，以及一个target特征图F'。输出是预测的相对位姿$$\xi$$​。从一个初始估计开始，迭代进行以下步骤：

1.将F中已知深度的点$$p$$投影到F‘中，得到$$p'$$，对每个点计算一个残差向量​$$r \in R^D$$，令reference像素和target像素尽可能相似：

![](<../../../.gitbook/assets/image (428).png>)

2.对每个残差，其关于相对位姿的偏导数为：

![](<../../../.gitbook/assets/image (392).png>)

3.根据堆叠的残差向量r、堆叠的Jacobian J和一个对角权重矩阵W，Gaussian系统和步长$$\delta$$计算如下：

![](<../../../.gitbook/assets/image (439) (2).png>)

这一偏导数和标准的图像对齐（DSO）是相近的。在计算Jacobian时，需要用到特征的数值偏导$$\frac{F'(p'_i)}{p'_i}$$。原本图像是非凸的，所以这一偏导只在很小的区域内​有效，这导致直接图像对齐需要一个较好的初始化结果。为了解决这一问题，作者使用了金字塔策略。作者通过训练网络让正确匹配附近区域有较好的平滑性。

**Gauss-Newton on individual pixels** 除了在6DOF位姿上使用GN算法，我们还可以对每个像素点单独使用GN算法（类似于LK光流法）。这一优化问题有相同的残差，但是优化的是像素的位置而非相对位姿。在这一情况中，Hessian是一个2x2的矩阵，更新量$$\delta$$​可以直接加到当前的像素位置上（这里为了简洁，省略了W）：

![](<../../../.gitbook/assets/image (469).png>)

这些独立的GN系统可以与6DOF位姿估计的GN系统相结合：

![](<../../../.gitbook/assets/image (423).png>)

The difference between our simplified systems and the one for pose estimation is only the derivative with respect to the pose, which is much smoother than the image derivative. This means that if the Gauss-Newton algorithm performs well on individual pixels it will also work well on estimating the full pose. Therefore, we propose to train a neural network on correspondences which are easy to obtain, e.g. using a SLAM method, and then later apply it for pose estimation. We argue that training on these individual points is superior to training on the 6DOF pose. The estimated pose can be correct even if some points contribute very wrong estimates. This increases robustness at runtime but when training we want to improve the information each point provides. Also, when training on the 6DOF pose we only have one supervision signal for each image pair, whereas when training on correspondences we have over a thousand signals. Hence, our method exhibits exceptional generalization capabilities as shown in the results section.

**The probabilistic Gauss-Newton loss** 公式6所描述的线性系统定义了一个二维的高斯概率分布，这是因为GN算法本质上是在最小二乘形式中寻找具有最大概率的解。这可以用高斯分布的负对数似然来推导：

![](<../../../.gitbook/assets/image (396).png>)

其中x是像素坐标，$$\mu$$​是均值。

在GN算法中，均值（对应着具有最大概率的点）可以通过$$\mu = x_s+\delta$$​来求得，其中$$\delta$$​来自公式5，$$x_s$$​是起始点。由于后一部分UI与所有的解x是一致的，所以只用到了第一项。但是，在我们的例子中，第二项是有关联的，因为网络对$$\mu$$​和H都有影响。

这说明由GN算法求得的H，b还定义了一个高斯概率分布，其均值为$$x_s+H^{-1}b$$​，协方差为$$H^{-1}$$​.

当从一个初始解x开始，网络应当赋予正确匹配点最大的概率。如果x是正确匹配，用E(x)=公式9作为损失函数，称为Gauss-Newton loss：

![](<../../../.gitbook/assets/image (449).png>)

在算法中，给Hessian矩阵的对角线加了一个小值$$\epsilon$$​，来保证它是可逆的。

**Analysis of the Gauss-Newton loss** 通过最小化公式9，网络最大化正确解得到概率密度。由于概率密度的总和始终为1，所以网络要不会将所有概率集中在一小部分解上，要不会将概率分散到更多解，使得每个解的概率密度较低。通过最大化正确解的概率，网络改进了估计的解和它的确定性。

损失函数的两部分也反映了这一点。第一项e1旨在缩小估计解和正确解之间的差异，用H进行缩放。当网络没有足够的确定性时，第二项e2会较大。这说明，网络会通过减小H来使e1减小，但是由于H的行列式值会减小，e2会随之增大。Notice also that this can be done in both dimensions independently. The network has the ability to output a large uncertainty in one direction, but a small uncertainty in the other direction. This is one of the traditional advantages of direct methods which are naturally able to utilize also lines instead of just feature points.

根据公式9，可以看到预测的不确定度只取决于初始位置target image的偏导。梯度越高，预测的确定性越高。在本文中，这让网络可以表达确定性，并让网络能够输出有区分能力的特征。

## Relocalization Tracking Benchmark

relocalization tracking应当具有两个输入：1.一个图像序列；2.一组独立的图像（可能采集自不同二等天气和时间），并且每张图像能够和1中的某张图像相匹配。

作者利用CARLA模拟器和Oxford RobotCar数据集，提出了一个relocalization tracking benchmark。

**CARLA**: For synthetic evaluations, we use CARLA version 0.8.2. We collect data for 3 different weather conditions representing WetNoon, SoftRainNoon, and WetCloudySunset. We recorded the images at a fixed framerate of 10 frames per second (FPS). At each time step, we record images and its corresponding dense depth map from 6 different cameras with different poses rendered from the simulation engine, which means that the poses in the benchmark are not limited to just 2DOF. The images and the dense depth maps are of size 512 x 512. For each weather condition, we collected 3 different sequences comprising 500-time steps with an average distance of 1.6m. This is done for training, validation, and testing, meaning there are 27 sequences, containing 6 cameras each. Training, validation, and test sequences were all recorded in different parts of the CARLA town.

**Oxford RobotCar**: Creating a multi-weather benchmark for this dataset imposes various challenges because the GPSbased ground-truth is very inaccurate. To find the relative poses between images from different sequences we have used the following approach. For pairs of images from two different sequences, we accumulate the point cloud captured by the 2D lidar for 60 meters using the visual odometry result provided by the Oxford dataset. The resulting two point clouds are aligned with the global registration followed by ICP alignment using the implementation of Open3D. We provide the first pair of images manually and the following pairs are found using the previous solution. We have performed this alignment for the following sequences: 2014-12- 02-15-30-08 (overcast) and 2015-03-24-13-47-33 (sunny) for training. For testing, we use the reference sequence 2015-02- 24-12-32-19 (sunny) and align it with the sequences 2015-03- 17-11-08-44 (overcast), 2014-12-05-11-09-10 (rainy), and 2015-02-03-08-45-10 (snow). The average relocalization distance across all sequences is 0.84m.

## Experimental Evaluation

We train our method using sparse depths created by running Stereo DSO on the training sequences. We use intra-sequence correspondences calculated using the DSO depths and the DSO pose. Meanwhile, inter-sequence correspondences are obtained using DSO depths and the ground-truth poses provided by our benchmark. The ground truth poses are obtained via Lidar alignment for Oxford and directly from the simulation engine for CARLA as explained in Section IV. Training is done from scratch with randomly initialized weights and an ADAM optimizer with a learning rate of $${10}^{-6}$$. The image pair fed to the Siamese network is randomly selected from any of the training sequences while ensuring that the images in the pair do not differ by more than 5 keyframes. Each branch of the Siamese network is a modified U-Net architecture with shared weights. Note that at inference time, only one image is needed to extract the deep visual descriptors, used as input to the SLAM algorithm. While in principle, our approach can be deployed in conjunction with any direct method, we have coupled our deep features with Direct Sparse Odometry (DSO).

![](<../../../.gitbook/assets/image (445).png>)

![](<../../../.gitbook/assets/image (474).png>)

![](<../../../.gitbook/assets/image (406).png>)
