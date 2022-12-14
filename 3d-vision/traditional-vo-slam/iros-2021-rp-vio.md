---
description: 'RP-VIO: Robust Plane-based Visual-Inertial Odometry for Dynamic Environments'
---

# \[IROS 2021] RP-VIO

{% embed url="https://arxiv.org/abs/2103.10400" %}

{% embed url="https://github.com/karnikram/rp-vio" %}

### Abstract

现有的VINS系统仅仅依靠目标的语义来剔除动态物体，难以处理所有动态场景。另一方面，真实场景中有很多平面，提供了结构性的规律，这些区域是静态的。因此，作者提出RP-VIO算法，用这些平面中的简单几何规则来获得面对动态场景的鲁棒性。作者还提出一个高动态的仿真数据集，用于测试VINS系统的性能。

### Introduction

VINS系统中，相机和IMU是互补的：IMU解决了单目相机的尺度因子不确定性；而相机令不可观测的IMU偏差和内参可以被观测到。但是，VINS还存在一些限制：硬件之间需要准确地同步和标定；系统需要有足够的旋转和加速运动来保证重力和尺度可被观测；当轨迹退化时，可能外参和内参没法被观测到，VINS需要在线标定。 此外，VINS在有多个独立动态物体的场景中表现不佳。基本的多视几何约束只在静态点上满足，在动态物体上会产生误差。这一问题在单目VINS的初始化阶段尤为显著，在这一阶段，由视觉SfM估计的位姿通常直接与预积分的IMU测量值对齐，来初始化尺度和IMU参数。不正确的位姿估计会导致整个跟踪过程失败。传统的方法是加入RANSAC来剔除动态特征，这种方法只在场景中动态特征较少时有效，并且前提是这些动态点不是沿着极线运动的。Motion segmentation方法直接预测每个像素点的运动状态，往往设计了多阶段的处理来从目标运动中分离出ego-motion，这种方法计算量较大，不适用于实时SLAM系统。 如果利用语义信息去剔除动态物体，枚举所有的动态物体会很难处理，并且动、静属性是人为定义的。更易处理的方法是直接识别场景中静态的结构，跳过语义信息。作者发现平面是人造环境中最丰富的静态区域，更重要的是平面可以提供简单的几何约束。 RP-VIO利用一个平面分割模型获得平面，只用了场景中平面上的特征，并利用平面自带的单应性进行运动估计。论文的主要贡献为：

1.在VINS-Mono的基础上提出RP-VIO，一个单目VIO系统，在初始化和滑窗估计时只采用平面特征和平面自带的单应性；&#x20;

2.一个仿真的视觉-IMU数据集，一直具有动态物体，具备足够的IMU激励；&#x20;

3.在所提出的室内数据集、VIODE数据集、OpenLORIS-Scene的两个序列和ADVIO数据集上测试了算法。

### Method

RP-VIO是基于VINS-Mono开发的，VINS-Mono基于IMU预积分和视觉特征进行紧耦合滑窗优化，作者没用使用重定位和回环模块，只用前端来检测和跟踪平面特征，在初始化和优化模块中引入单应性约束。

#### Definitions

![](../../.gitbook/assets/rpvio\_1.png)

W表示world frame，z轴沿重力方向向下。B表示body frame，定义在IMU上。C表示camera frame。$$R_{ji},t_{ji},T_{ji}$$分别为从frame $$t_i$$到$$t_j$$的旋转、平移和单应性矩阵。这个frame可以是camera frame或body frame。$$R_{i},t_{i}$$表示在$$t_i$$时frame相对于world frame的旋转和平移。$$u^l$$表示第l个特征的归一化图像坐标，响应的3D点$$p_l$$用相对于观察的第一帧的拟深度$$\lambda_l$$来表示。一个平面$$\pi_p$$用相对于$$C_0$$的法向量和距离(n,d)来表示。平面单应性矩阵$$H_j$$（图2）将$$C_0$$中平面点的2D图像坐标投影到$$C_j$$上。&#x20;

$$t_i$$时刻系统的状态$$x_i$$由IMU位置、朝向、速度、偏差、3D点的拟深度和平面参数定义，即$$x_i=[R_i,t_i,v_i,b_i,{\lambda_l},{\pi_p}]$$&#x20;

$$\mathcal{X}$$表示滑窗$$\mathcal{K}$$内所有帧的状态，即我们想要估计的状态，$$\mathcal{X}={\{x_i\}}_{i\in\mathcal{K}}$$

#### Front-end

系统以灰度图、IMU测量和平面分割mask为输入，只检测和跟踪静态平面上的特征点，并且保留每个跟踪到的特征所属平面的信息。为了避免检测到mask边缘上可能属于动态对象的特征，算法对原始mask进行了侵蚀处理。此外，我们使用RANSAC将一个独立的平面单应性模型拟合到每个平面的特征上，以剔除外点。图像帧之间的IMU测量被转换到预积分测量中，具有足够视差和特征跟踪的图像帧被选为关键帧。

#### Initialization

主要的视觉惯性滑动窗口优化方法是非凸的，并且是迭代最小化的，需要精确的初始估计。为了获得一个较好的初始估计，并且不做任何关于初始配置的假设，算法使用了一个独立的松耦合初始化过程，这个过程中视觉观测和惯导观测是独立处理的，分别求解出对应的位姿估计，然后对齐，来多步求解未知数。&#x20;

算法首先解决相机姿态、3D点和平面参数。从初始的图像帧的滑窗中，选择两个足够视差的base frame。在所有的匹配特征中，我们只选择那些来自场景中最大的平面特征，如具有最多特征的平面。利用这些特征，算法用RANSAC拟合了两个base frame中最大平面之间的平面单应性矩阵H。单应性矩阵经过正则化，然后用OpenCV分解为旋转、平移和平面法向量。这个方法可以得到四个解，首先通过约束正深度来剔除两个解，即所有的平面特征必须位于相机前面。这作为一个约束，$$n_i^Tu_\mu>0$$，其中$$u_\mu$$是归一化图像坐标系中的平均2D特征点。对于剩下的两个解，选择那个与对应IMU预积分旋转$$\triangle\tilde{R_{ij}}$$更接近的解(转换到B frame上)：

![](../../.gitbook/assets/rpvio\_0.png)

虽然在IMU预积分旋转中陀螺仪的偏差没有被估计出来，它的幅度很小，不会对这个解产生影响。分解得到的位姿被用于三角化两个base frame间的特征，获得一个初始点云。滑窗内其他frame的位姿是由PnP根据这些初始点云估计得到的。由于两个base frame之间估计的位姿是以平面距离d为尺度的，因此三角化的点云和推理出的位姿也是这个尺度的。所有的位姿估计被输入一个视觉bundle adjustment solver，除了标准的3D-2D重投影误差，作者还引入了由平面单应性引出的2D-2D重投影误差：&#x20;

![](../../.gitbook/assets/rpvio\_2.png)

这一残差度量了将frame $$C_j$$中点$$p_l$$的对应像素点$$u^l$$用平面单应性矩阵从第一帧投影得到的预期观测，与真值观测$$u^l_j$$之间的差异。BA的输出是带有尺度(d)的相机位姿、3D特征点和平面法向量。这个未知尺度(d)，以及初始化主要优化所需的其余未知量，如重力矢量、速度和IMU偏差，都是使用相同的分治法进行估计的。&#x20;

一旦这些被估计后，相机位姿和3D特征点被重新归一化到单位尺度，world frame被重新对齐来让z轴在重力方向上。对于最大平面之外的其他平面，包括在计算中新观测到的平面，作者也相似地计算了他们之间的平面单应性矩阵，然后分解。为了节省计算，没有再进行一轮BA与位姿和IMU测量之间的对齐来估计分别的尺度因子$$d_p$$。作者直接令$$d_p$$为每个分解得到的平移$$t_p$$的反比（即$$d_p$$的尺度为$$\frac{t}{t_p}$$，t是利用最大平面和惯导测量估计出的平移。在此基础上，求解了状态中的所有视觉量和惯性量，并将这些估计量作为优化的初始种子输入滑动窗估计器。

#### Sliding-window Optimization

![](../../.gitbook/assets/rpvio\_3.png)

记滑窗$$\mathcal{K}$$内相邻帧i和j的所有IMU测量为$$\mathcal{I}_{ij}$$，帧i中所有平面特征记为$$\mathcal{C}_i$$，所有观测到的平面记为$$\mathcal{P}$$。滑窗内的这些状态和测量的因子图如图3所示。滑窗内所有状态的MAP估计$$\mathcal{X}^*$$可以用最小化平方残差和来得到：

&#x20;

![](../../.gitbook/assets/rpvio\_4.png)

其中$$r_p$$是从之前状态边缘化得到的先验残差，$$r_{\mathcal{I}{ij}}$$是IMU预积分残差，$$r_{\mathcal{C}_{ij}}$$是标准的3D-2D重投影误差，$$\rho$$为Cauchy loss来降低外点的权重，$$r_H$$是平面单应性残差：

&#x20;

![](../../.gitbook/assets/rpvio\_5.png)

这一项与初始化过程中用的残差很像，除了位姿和平面参数是在body frame系。第p个平面的法向量$$n^p$$和深度$$d^p$$由下式从第一个camera frame $$C_0$$转换到当前body frame $$B_i$$:&#x20;

![](../../.gitbook/assets/rpvio\_6.png)

This entire non-linear objective function is minimized iteratively using the Dogleg algorithm with Dense-Schur linear solver implemented in Ceres Solver. 在优化的结尾，滑窗向前移动一帧，来包含最新的帧。丢弃的帧用VINS-Mono的方法被边缘化。优化后的平面参数并没有被丢弃或忽略，而是在再次观察到平面时重复使用。

#### Plane Segmentation

作者用Plane-Recover模型来分割平面实例。这一模型使用一个同时预测平面mask和平面参数的结构相关的loss来训练的，训练时只需要语义标签，无需额外的3D标注。该模型在单张NVIDIA GTX TiTAN X (Maxwell) GPU上达到30FPS的推理速度。 为了让分割的平面连续并且避免同一平面被分割为多块，作者还引入了一个inter-plane损失将具有较小相对方向的平面约束为单个平面：

&#x20;

![](../../.gitbook/assets/rpvio\_7.png)

其中n是平面法向量，m是平面的数量（文中设为3），n为一个batch中的图像数量，$$l_i$$是在线生成的inter-plane标签，当$$\angle (n^j_i,n^j_i)  <\frac{\pi}{4}$$时，标签设为1（:confused:这个夹角一直为0吧...），否则设为0.

根据这个损失函数，我们在SYNTHIA数据集上重新训练了模型，并在室内ScanNet数据的两个序列（00,01）上进行了训练。为了改进分割效果，并且获得精细的边缘细节，作者使用了fully dense conditional random field (CRF)模型来优化分割，采用默认参数。

![](<../../.gitbook/assets/image (337).png>)

### Experiments

> All the evaluations are run on a 6-core Intel Core i5-8400 CPU with 8 GB RAM and a 1 TB HDD. To account for randomness from RANSAC and the multi-tasking OS, we report the median results from five runs for each evaluation.

#### Simulation Experiments

> We build a custom indoor warehouse environment with dynamic characters in Unreal Engine. We borrow several high-quality and feature-rich assets from the FlightGoggles project for photorealism. This environment is integrated with AirSim to spawn a quadrotor and collect visual-inertial data. We collect monocular RGB images and their plane instance masks at 20 Hz, IMU measurements and ground truth poses at 1000 Hz. The IMU measurements are sub-sampled to 200 Hz for our experiments. The camera and IMU intrinsics, and the camera-IMU spatial transform are obtained directly from AirSim. A time-offset of 0:03 s between the camera and IMU measurements, introduced by the recording process, is calibrated using Kalibr.
>
> The quadrotor is controlled to move along a circle of radius 15 m, while moving along a sine wave in the vertical direction, resulting in a sinusoidal pattern. The sine excitation along the height is to ensure a non-constant acceleration and keep the scale observable. We further command it to accelerate vertically at the beginning of its motion, before following the trajectory, to help the initialization. The total trajectory is of 200 m length and 80 s duration, with a maximum speed of 3 m/s. Within the circle formed by the quadrotor, we introduce dynamic characters that are performing a repetitive dance motion. We progressively add more dynamic characters to each sequence, keeping everything else fixed, starting from no characters (static) and going up to 8 characters (C8), recording six sequences in total. The yaw-direction of the quadrotor is also fixed to keep the camera pointing towards the center of the circle, such that the characters are in the FoV of the camera for the entire sequence. The quadrotor and the characters are controlled programmatically to ensure their motions are repeatable and are in sync across all the sequences.

![](<../../.gitbook/assets/image (151).png>)

![](<../../.gitbook/assets/image (500).png>)

#### Experiments on Standard Datasets

![](<../../.gitbook/assets/image (671).png>)

![](<../../.gitbook/assets/image (867).png>)
