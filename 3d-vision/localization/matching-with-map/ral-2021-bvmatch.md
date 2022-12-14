---
description: 'BVMatch: Lidar-Based Place Recognition Using Bird’s-Eye View Images'
---

# \[RAL 2021] BVMatch

{% embed url="https://github.com/zjuluolun/BVMatch" %}

## Abstract

In this letter we present BVMatch, a Lidar-based frame-to-frame place recognition framework, that is capable of estimating 2D relative poses. Based on the assumption that the ground area can be approximated as a plane, we uniformly discretize the ground area into grids and project 3D Lidar scans to bird’s-eye view (BV) images. We further use a bank of Log-Gabor filters to build a maximum index map (MIM) that encodes the orientation information of the structures in the images. We analyze the orientation characteristics of MIM theoretically and introduce a novel descriptor called bird’s-eye view feature transform (BVFT). The proposed BVFT is insensitive to rotation and intensity variations of BV images. Leveraging the BVFT descriptors, we unify the Lidar place recognition and pose estimation tasks into the BVMatch framework.

## Introduction

BVMatch将地面划分为等间隔的grids，将扫描到的点累加到每个grid中，来构成BV图像。但是LiDAR得到的BV图像是比较稀疏，并且强度的畸变较大。因此，作者提出bird's-eye view feature transform (BVFT)，这是一种对强度和旋转变化不敏感的BV图像特征，作者基于maximum index map (MIM) of Log-Gabor filter responses来构建BVFT，然后从理论上分析了它对于旋转偏移的的性能。基于BVFT，BVMatch利用BoW方法进行场景识别，用RANSAC去估计相对位姿。论文贡献如下：

* We propose a novel local descriptor called BVFT that is insensitive to intensity and rotation variation of BV images, with which the BVMatch framework can unify the Lidar place recognition and pose estimation tasks.&#x20;
* We theoretically prove that BVFT can achieve rotation invariance by shifting the orientations in the local MIM.&#x20;
* We experimentally validate on three large-scale datasets that BVMatch outperforms the state-of-the-arts on both place recognition and pose estimation.

## BV Image based Place Recognition

The BVMatch method consists of the place recognition step and the pose estimation step. In the place recognition step, BVMatch generates the BV image and performs frame retrieval with the BoW approach. In the pose estimation step, BVMatch matches the BV image pair using RANSAC and reconstructs the coarse 2D pose of the Lidar pair with a similar transform. At last, BVMatch uses iterative closest point (ICP) refinement with the 2D pose as an initial guess to accurately align the Lidar pair.

<figure><img src="../../../.gitbook/assets/image (992).png" alt=""><figcaption></figcaption></figure>

### BV Image Generation

作者用density map作为BV image表现的形式。令$$P=\{P_i|i=1,...,N_p\}$$​为点云。假设点云是在道路场景中采集的，x轴指向右，y轴指向前，z轴指向上。在这个坐标系中，x-y平面是地面平面。给定一个点云P，用一个leaf size为g米的voxel grid filter来让点均匀分布。然后将地面平面离散化为grid，每个grid的分辨率为g米。点云density为在每个grid中点的数量。设在坐标系原点有一个\[-C,C]米的正方形滑窗，然后BV图像B(u,v)是一个大小为$$\lceil \frac{2C}{g}, \frac{2C}{g} \rceil$$​的矩阵。BV图像的density B(u,v)定义为：

<figure><img src="../../../.gitbook/assets/image (984).png" alt=""><figcaption></figcaption></figure>

其中$$N_g$$​为grid中点的数量，$$N_m$$​为normalization factor，设为the 99th percentile of the point cloud density.

用FAST算法进行特征检测。

### Pose Reconstruction

由于BV图像将地面平面均匀离散化，所以一对LiDAR scan之间的变换$$(P_i,P_j)$$​和一对BV图像之间的变换$$(B_i(u,v),B_j(u,v))$$是相似的。当获得$$(B_i(u,v),B_j(u,v))$$​之间的变换，有$$B_i(u,v)=B_j(u',v')$$：

<figure><img src="../../../.gitbook/assets/image (954).png" alt=""><figcaption></figcaption></figure>

其中$$(t_u,t_v,\theta)$$​为变换参数。$$(P_i,P_j)$$​之间变换矩阵$$T_{ij}$$​为：

<figure><img src="../../../.gitbook/assets/image (983).png" alt=""><figcaption></figcaption></figure>

其中g是voxel grid filter的leaf size。

### Dictionary and Keyframe Database

BVMatch leverages the bag-of-words approach to extract global descriptors and uses a keyframe database to detect the best match frame of a query Lidar scan.

Keyfame database stores BV images with their global poses and descriptors. We extract keyframe Lidar scans every S meters the robot moves and generate a global descriptor for every keyframe. The keyframe database is built using all these global descriptors, poses, and BV images.

## Proposed BVFT Descriptor

Although BV image preserves the vertical structures that are stable in a scene, it suffers severe intensity distortion due to the sparsity nature of Lidar scans. To extract distinct local descriptors, we first leverage Log-Gabor filters to compute the local responses of BV images. We then construct a maximum index map (MIM) originally used for multi-modal image matching. Finally, we build bird’s-eye view feature transform (BVFT) that is insensitive to intensity and rotation variations of BV images.

### Maximum Index Map (MIM)

用极坐标来表示图像。一个欧式坐标$$(u,v)$$的极坐标$$(\rho,\theta)$$为：

<figure><img src="../../../.gitbook/assets/image (910).png" alt=""><figcaption></figcaption></figure>

其中$$(\overline{u},\overline{v})$$是图像中心。

基于Log-Gabor filter构建MIM。频域的2D Log-Gabor filter为：

<figure><img src="../../../.gitbook/assets/image (924).png" alt=""><figcaption></figcaption></figure>

其中$$f_s$$​是在尺度s上的中心频率，$$\omega_o$$​是方向o处的中心频率。对于一个$$N_s$$​个尺度、$$N_o$$​个方向的Log-Gabor filter集合，滤波器在频谱上均匀分布。作者从集合$$O=\{0,\pi/N_o,2\pi/N_o,...,(N_o-1)\pi/N_o\}$$​中选择方向，因此$$w_o=o\pi/N_o$$.​2D Log-Gabor filter是各向同性的。

用一组$$N_s$$​个尺度、$$N_o$$​个方向的Log-Gabor filter来构建MIM。对于一个在尺度s、方向o上的Log-Gabor filter，令$$L(\rho,\theta,s,o)$$​为空间域中公式（5）的对应滤波器，$$A(\rho,\theta,s,o)$$​为方向o和尺度s上$$B(\rho,\theta)$$​的Log-Gabor幅值响应，则：

<figure><img src="../../../.gitbook/assets/image (957).png" alt=""><figcaption></figcaption></figure>

其中\*为卷积操作。则方向o上Log-Gabor幅值响应为：

<figure><img src="../../../.gitbook/assets/image (922).png" alt=""><figcaption></figcaption></figure>

maximum index map (MIM)是Log-Gabor响应最大的方向：

<figure><img src="../../../.gitbook/assets/image (930).png" alt=""><figcaption></figcaption></figure>

### BVFT Descriptor

在BVFT中，作者用Log-Gabor filters去捕获严格垂直结构中的方向信息（如公式8）。这使得MIM对于BV图像的intensity畸变不敏感。但是，MIM对于旋转比较敏感，通过将MIM中的方向相对于指定的主方向进行变换，可以实现旋转不变性。

<figure><img src="../../../.gitbook/assets/image (925).png" alt=""><figcaption></figcaption></figure>

图2中（g）和（b）之间的关系为：

<figure><img src="../../../.gitbook/assets/image (912).png" alt=""><figcaption></figcaption></figure>

可以看到，旋转后，MIM发生了很大的变化。为了获得旋转不变性，需要弥补这一变化。

计算$$B_\alpha$$​关于方向o的Log-Gabor幅值响应：

<figure><img src="../../../.gitbook/assets/image (948).png" alt=""><figcaption></figcaption></figure>

因为2D log-gabor filter的频率部分在频域内是各向同性的，尺度s上每个滤波器的都可以通过旋转同一尺度上其他滤波器来获得，当$$\alpha$$​是旋转集合O中的一个角度：

<figure><img src="../../../.gitbook/assets/image (896).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (905).png" alt=""><figcaption></figcaption></figure>

这里用到了mod处理，因为$$N_o$$​个角度构成了一个环。因为2D 傅里叶变换是旋转不变的，2D 空间 Log-Gabor filter具有和频率一样的性质：

<figure><img src="../../../.gitbook/assets/image (911).png" alt=""><figcaption></figcaption></figure>

将（13）代入（10）得到：

<figure><img src="../../../.gitbook/assets/image (991).png" alt=""><figcaption></figcaption></figure>

这说明Log-Gabor 响应图$$A_\alpha(\rho,\theta,o)$$可以通过将$$A(\rho,\theta+\alpha,o_\alpha)$$旋转$$\alpha$$​来获得，将（14）带入（8）得到：

<figure><img src="../../../.gitbook/assets/image (981).png" alt=""><figcaption></figcaption></figure>

这揭示了当点云以方向集合O中任何角度旋转时，通过将未旋转的MIM进行简单的循环移位操作，来获得它的MIM。

作者进一步地在图像块上进行以上分析，并设计了具备旋转不变性的局部描述子。对于每个检测到的关键点，找到它的主方向，在以关键点为中心的JxJ方形MIM块$$patch(\rho,\theta)$$上建立局部直方图$$h(o)$$。在计算直方图时，对增量进行高斯窗口加权，高斯均值为(0, 0)，标准差为(J/2, J/2)​。假设直方图的峰值在方向$$o_m$$​，即$$o_m=argmax_o h(o)$$​，主方向为：

<figure><img src="../../../.gitbook/assets/image (944).png" alt=""><figcaption></figcaption></figure>

然后将块旋转$$\beta$$：

​

<figure><img src="../../../.gitbook/assets/image (945).png" alt=""><figcaption></figcaption></figure>

经过如此处理，局部块具有旋转不变性。将块分为lxl大小的网格，对每个网格构建分布直方图，这些直方图被拼接起来作为BVFT特征面描述子，长度为$$l\times l \times N_o$$​.

The above analysis is conducted under the assumption that the rotation angles are within the orientation set O. However, (11) and (15) still hold when the rotation angles are within the set $$\hat{O}=\{\pi, (N_o+1)\pi/N_o,...,(2N_o-1)\pi/N_o\}$$. Thus, we cannot determine whether the dominant orientation is $$\pi \frac{o_m}{N_o}$$ or $$2\pi-\pi \frac{o_m}{N_o}$$ given $$o_m$$. To avoid this ambiguity, we rotate every shifted MIM patch by $$\pi$$ and assign every keypoint with an additional descriptor generated using this patch. When the rotation angle is not within O or $$\hat{O}$$, it is obvious that (15) does not hold. In fact, BVFT cannot cover continuous orientation because the Log-Gabors are band-pass filters. In practice, we have found that BVFT has good matching ability in continuous orientation, and we will show that our BVFT has excellent performance for the place recognition problem in the experiment section.

## Experiments

<figure><img src="../../../.gitbook/assets/image (974).png" alt=""><figcaption><p>参数设置</p></figcaption></figure>

对比算法有：[M2DP](https://github.com/LiHeUA/M2DP), [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), [PCAN](https://github.com/XLechter/PCAN), [LPD-net](https://github.com/Suoivy/LPD-net), [DH3D](https://github.com/JuanDuGit/DH3D), [OverlapNet](https://github.com/Suoivy/OverlapNet).

All the training sequences are sampled every 2 meters, while the test sequences are sampled every 10 meters. Point cloud data in a \[−50 m, 50 m] cubic window of a frame are used for all methods.

### Place Recognition

<figure><img src="../../../.gitbook/assets/image (936).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (915).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (939).png" alt=""><figcaption><p>泛化性</p></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (917).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (975).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (908).png" alt=""><figcaption></figcaption></figure>

### Pose Estimation

<figure><img src="../../../.gitbook/assets/image (966).png" alt=""><figcaption></figcaption></figure>

### Runtime Evaluation

We implemented the BVFT descriptor generation and RANSAC using C++ and implemented BVMatch framework using Python. Our platform running the experiments is a desktop computer equipped with an Intel Quad-Core 3.40 GHz i5-7500 CPU and 16 GB RAM. The average time cost for each frame is 0.29 seconds to extract BVFT descriptors, 0.02 seconds to generate a global descriptor, 0.05 seconds to perform retrieval, and 0.23 seconds to register BV image pair using RANSAC. The total time cost for each frame is 0.59 seconds.
