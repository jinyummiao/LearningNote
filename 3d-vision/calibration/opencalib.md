---
description: A Multi-sensor Calibration Toolbox for Autonomous Driving
---

# OpenCalib

{% embed url="https://github.com/PJLab-ADG/SensorsCalibration" %}

## Abstract

Accurate sensor calibration is a prerequisite for multi-sensor perception and localization systems for autonomous vehicles. The intrinsic parameter calibration of the sensor is to obtain the mapping relationship inside the sensor, and the extrinsic parameter calibration is to transform two or more sensors into a unified spatial coordinate system. Most sensors need to be calibrated after installation to ensure the accuracy of sensor measurements. To this end, we present OpenCalib, a calibration toolbox that contains a rich set of various sensor calibration methods. OpenCalib covers manual calibration tools, automatic calibration tools, factory calibration tools, and online calibration tools for different application scenarios. At the same time, to evaluate the calibration accuracy and subsequently improve the accuracy of the calibration algorithm, we released a corresponding benchmark dataset. This paper introduces various features and calibration methods of this toolbox.

## Introduction

IMU可以提供短期内的相对位置偏移和角度变化；GNSS可以提供米级的绝对位置，但是GNSS信号的质量易受影响；相机能够提取环境的细节信息，除颜色外，还可提供纹理和对比度数据，能够可靠地检测路标、交通标识，分辨动静物体。但是需要较好的光照条件，且只能提供二维数据，没有直接的深度信息；LiDAR拥有高准确性、长距离、良好实时性的测量能力，不受光照影响，环境适应性强。但是缺乏颜色信息，对于有反射性和透明性的目标无法准确检测，数据扫描需要大量计算资源，扫描速度慢；毫米波雷达成本低，检测准确率高，对特定材质敏感，响应快，易于处理，对恶劣天气适应性好。但是分辨率低，无法判断被识别目标和行人的大小。

因此需要多种传感器的融合。

传感器标定包含两类：内参标定和外参标定。内参标定旨在标定传感器内部的映射关系。如相机的焦距和相机畸变；IMU的陀螺仪和加速度计的零偏、尺度因子和安装误差；LiDAR内部激光发射器之间的转换关系、LiDAR坐标设备。外参则是指不同传感器之间的6DoF相对位姿，包含旋转和平移。

本文的贡献总结如下：

* We propose OpenCalib, a multi-sensor calibration toolbox for autonomous driving. The toolbox can be used for different calibration scenarios, including manual calibration tools, automatic calibration tools, factory calibration tools and online calibration tools.
* We propose many novel calibration algorithms in the toolbox, such as various automatic calibration methods based on road scenes. For the factory calibration in the toolbox, we propose more robust recognition algorithms for multiple calibration board types and the calibration board recognition program removes the OpenCV library dependency.&#x20;
* To benchmark the calibration performance, we introduce a synthesized dataset based on Carla, where we can get the ground truth for the calibration results. In the future, we will further open-source the calibration benchmark dataset.
* We open-source the toolbox code on GitHub to benefit the community. The open-source code is the v0.1 version, and we will continue introducing more stateof-the-art calibration algorithms in future versions.

## Methodology

### Manual Target-less Calibration Tools

本文提出的标定toolbox提供了四种target-less的人工标定工具，适用于道路场景。

<figure><img src="../../.gitbook/assets/image (580).png" alt=""><figcaption><p>用户界面</p></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (664).png" alt=""><figcaption><p>键盘控制输入</p></figcaption></figure>

#### LiDAR to camera calibration

该过程需要通过调整内外参来对齐道路场景中的3D LiDAR点云和图像。3D点云表示为LiDAR坐标系中的$$p^L_i={(X_i,Y_i,Z_i)}^T$$，该点在相机坐标系中为$$p^C_i={(X_C,Y_C,Z_C)}^T$$，转换关系为：

<figure><img src="../../.gitbook/assets/image (627).png" alt=""><figcaption></figcaption></figure>

$$p^C_i$$投影到相机平面中：

<figure><img src="../../.gitbook/assets/image (604).png" alt=""><figcaption></figcaption></figure>

相机内参包括焦距和相机畸变，使用张氏标定法来标定的。

IntensityColor 按钮可以改变intensity map表示的模式。

OverlapFilter 按钮用于剔除深度在0.4m内的重叠LiDAR点。

#### LiDAR to LiDAR calibration

该过程需要通过调整外参来在source点云和target点云之间实现3D点云和注册。source点云表示为$$p^S_i={(X_i,Y_i,Z_i)}^T$$，target点云表示为$$p^T_i={(X_i,Y_i,Z_i)}^T$$，两点云间的刚体变换表示为：

<figure><img src="../../.gitbook/assets/image (630).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (632).png" alt=""><figcaption></figcaption></figure>

#### Radar to LiDAR calibration

该过程实现2D radar点云和3D LiDAR点云间的注册，所以需要调整的参数更少。

#### Radar to camera calibration

radar是一个二维传感器，因此希望让它在安装时与地面平行。对于相机，需要找到它与地面间的单应性矩阵，然后在图像和BEV中可视化，当两图像中的对齐完成，标定就完成了。图4显示了在左右车道线中各挑选两个点，相机到地面的单应性矩阵是如何计算的。

<figure><img src="../../.gitbook/assets/image (636).png" alt=""><figcaption></figcaption></figure>

图5展示了radar在图像中的投影结果和BEV结果。BEV中的平行线指示了车道线的方向。

<figure><img src="../../.gitbook/assets/image (600).png" alt=""><figcaption></figcaption></figure>

### Automatic Target-based Calibration Tools

#### Camera calibration

由于每个相机镜头畸变的自由度是不同的，且相机镜头矫正对于下游的感知任务或联合标定任务很重要，因此需要发展一种相机畸变的定量评估方法。

<figure><img src="../../.gitbook/assets/image (625).png" alt=""><figcaption></figcaption></figure>

畸变评估的目标由钓鱼线和半透明纸组成，流程图如图6所示。首先，用相机去采集目标的推向，目标填满了图像。该图像然后用图像内参去畸变，得到矫正图像。然后，用Canny描述子去提取直线，用线性插值去获得连续的直线。经过高斯采样，得到需要线性采样的点。最后，用最小二乘法来获得拟合的直线。根据公式4和公式5，计算采样点到拟合直线之间的均方根距离和最大误差距离。回归的直线表示为

$$\alpha * x+\beta*y-\gamma=0$$

给定L条直线：

<figure><img src="../../.gitbook/assets/image (597).png" alt=""><figcaption></figcaption></figure>

这两个指标反映了畸变参数的质量。

#### LiDAR to camera calibration

LiDAR和camera的联合标定一般是先标定相机的内参，再标定LiDAR和相机间的外参。但是相机内参标定的误差会影响外参的标定。为此，作者提出了一个联合标定方法\[Joint camera intrinsic and lidar-camera extrinsic calibration]。

<figure><img src="../../.gitbook/assets/image (663).png" alt=""><figcaption></figcaption></figure>

作者设计了一款标定板，包含一个棋盘格板（用于标定相机内参）和多个圆形孔（用于定位LiDAR点云）。首先用张氏标定法来标定相机内参和相机-棋盘格板之间的初始外参。然后，用这些参数和板子尺寸来计算图像中2D圆的中心点。提取LiDAR的圆心点，根据LiDAR-相机间的外参来将3D LiDAR圆心点投影到图像中。计算的2D点和投影点构成了多个2D点对，计算这些点对的欧式距离，来优化标定参数。同时，将标定板角点的3D-2D点投影限制也加入优化过程。

标定板角点的数量要远多于圆孔的数量，因此，LiDAR-相机间对齐误差的权重要大于板子-相机间对齐误差，以平衡两个损失函数。总的来说，需要最小化的目标函数为：

<figure><img src="../../.gitbook/assets/image (578).png" alt=""><figcaption></figcaption></figure>

其中$$J_{board}$$​表示标定板的角点重投影误差，$$J_{lidar}$$表示标定板的圆心重投影误差。实验中，设定$$\alpha=1,\beta=60$$.

<figure><img src="../../.gitbook/assets/image (655).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (657).png" alt=""><figcaption></figcaption></figure>

### Automatic Target-less Calibration Tools

#### IMU heading calibration

该过程旨在矫正IMU和车辆的正前方向的安装误差。因此，只标定了IMU的yaw角偏离量$$\gamma_{offset}$$，来对齐朝向，驾驶轨迹中每一刻的车辆方向记为$$\gamma_{gd}$$。估计的驾驶方向和测量的IMU yaw角$$\gamma_{IMU}$$之间的偏离量为标定结果。

作者用b-spline方法来平滑基于传感器定位结果的驾驶轨迹，只挑选正前向驾驶的轨迹来进行标定。标定公式为：

​

<figure><img src="../../.gitbook/assets/image (651).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (590).png" alt=""><figcaption></figcaption></figure>

#### LiDAR to camera calibration

使用的方法为\[Crlf: Automatic calibration and refinement based on line feature for lidar and camera in road scenes]，该方法可以在常见的道路场景中标定LiDAR和相机间的外参。

<figure><img src="../../.gitbook/assets/image (576).png" alt=""><figcaption></figcaption></figure>

首先，采集一些道路场景数据，从图像和点云数据中提取线特征（如车道线和路标）。用BiSeNet-V2来提取图像中的车道线和路标，根据强度和几何方法来从点云中提取车道线和路标，然后设计了一个损失函数来优化初始的标定外参结果，保证误差在可接受的范围内。根据语义类别直接获得杆子的像素$$Q_{pole}$$和车道的像素$$Q_{lane}$$，结合分割结果，得到两个二进制mask $$M_{line}: R^2 \rightarrow \{0,1\}, line \in \{pole,lane\}$$，在像素坐标系中：

<figure><img src="../../.gitbook/assets/image (574).png" alt=""><figcaption></figcaption></figure>

在接下来的优化过程中，作者对mask进行inverse distance transformation (IDT)处理来避免重复的局部最大值，得到的高度map，$$H_{line}$$为：

<figure><img src="../../.gitbook/assets/image (629).png" alt=""><figcaption></figcaption></figure>

然后提出投影损失函数$$J_{proj}: (r,t)\rightarrow R$$，表示投影像素与对应mask之间的一致性

<figure><img src="../../.gitbook/assets/image (589).png" alt=""><figcaption><p>其实本质上还是调整R,t来让projected pixel落在mask中，H是为了给mask中每个像素一个“独一无二”的值吧 </p></figcaption></figure>

符号$$\circ$$表示投影点处的高度值。该损失函数越大，说明两种数据的语义特征匹配越好。

<figure><img src="../../.gitbook/assets/image (656).png" alt=""><figcaption></figcaption></figure>

#### Lidar to IMU calibration

一般而言，当标定LiDAR到IMU的外参时，标定准确率通过判断LiDAR的局部建图质量来决定。标定过程就是通过滑窗局部建图来求解LiDAR到IMU的外参。标定工具基于\[Balm: Bundle adjustment for lidar mapping]，The feature points are distributed on the same edge line or plane on the local map by minimizing the eigenvalues of the covariance matrix. The method minimizes the sum of distances from feature points to feature planes or edge lines by minimizing the eigenvalues of the covariance matrix and optimizes to achieve the purpose of extrinsic parameter calibration from LiDAR to IMU. 用BA算法来最小化每个平面特征点到平面的距离：

<figure><img src="../../.gitbook/assets/image (628).png" alt=""><figcaption></figcaption></figure>

其中$$p_i$$​是滑窗内被优化的点，被投影到局部地图，q是一些特征上的点。n是平面法向量。在该方法中，任意时刻LiDAR在世界坐标系下的位姿是可以获得的，可以通过下式来获得粗略的初始外参：

<figure><img src="../../.gitbook/assets/image (587).png" alt=""><figcaption></figcaption></figure>
