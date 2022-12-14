---
description: >-
  A Survey of the State-of-the-Art Localization Techniques and Their Potentials
  for Autonomous Vehicle Applications
---

# \[JIoT 2018] SOTA Localization Techniques

{% embed url="https://ieeexplore.ieee.org/document/8306879" %}

## Abstract

The analysis starts with discussing the techniques which merely use the information obtained from on-board vehicle sensors. It is shown that although some techniques can achieve the accuracy required for autonomous driving but suffer from the high cost of the sensors and also sensor performance limitations in different driving scenarios (e.g., cornering and intersections) and different environmental conditions (e.g., darkness and snow). This paper continues the analysis with considering the techniques which benefit from off-board information obtained from V2X communication channels, in addition to vehicle sensory information. The analysis shows that augmenting off-board information to sensory information has potential to design low-cost localization systems with high accuracy and robustness, however, their performance depends on penetration rate of nearby connected vehicles or infrastructure and the quality of network service.

## Introduction

自动驾驶框架可以从功能上划分为五个部分：localization、perception、planning、control和system management。这些环节相互配合，来回答三个重要问题：“车在哪儿？”“车的周围有什么？”“车下一步的动作是什么？”其中感知、规划和控制系统需要准确的汽车位置来做出正确的驾驶决策。定位系统的准确度和鲁棒性非常重要。GPS信号很常用，但是由于信号阻挡、多径效应，可信度较低，并且精度不够。因此，需要其他传感器（如RADAR、LiDAR、相机等）的辅助，提升定位效果。这些传感器提供了比GPS更准确而可信的位置测量，但是成本也有所提高，此外它们的视距有限、鲁棒性易受干扰。可以通过物联网技术，实现协同定位，获得超视距信息。

## Mapping Techniques

一般用于定位的地图类型有两种：1. 平面的，即基于geographic information system的层或平面，例如高精度地图；2. 点云的，即GIS中的数据点集合。

需要根据所使用的传感器来决定使用哪种地图。视觉传感器（比如相机、LiDAR、RADAR和超声波）主要使用点云地图，而基于GPS的系统主要使用平面地图。在点云地图中使用map matching算法需要消耗大量计算力。但是，在定位精度上，这两种地图没有什么明显优势，因为高精度地图和3D地图在采集时都需要用到视觉传感器。

平面地图通过采集和分析高分辨率图像或航拍图像、GPS轨迹和成像来生产。这些地图的分辨率取决于捕获的数据和来自数据的附加信息层。例如，高精地图提供了一个道路网的基本地图层，精度达到sublane级，包括路标、道路设施和曲率。利用双目相机提供的信息，3D地图可以获得目标的高度。除了环境拓扑结构的静态信息外，动态物体也可以加入地图中。Local dynamic map (LDM)是一种标准的表达这种信息的方法，包含很多信息层，这些动态信息可以用于自适应的定位，让地图维护更高效，如图1.

<figure><img src="../../../.gitbook/assets/image (652).png" alt=""><figcaption></figcaption></figure>

另一方面，点云地图是用3D传感器生产的，表征了3D空间中的物体表面。在基于马尔科夫的定位系统中，车辆从车载传感器和V2X信息中收集有关环境和其他道路用户的信息，并利用这些信息在预先存在的静态特征地图中定位车辆；在SLAM系统中，当车辆经过环境，构建LDMs，与预先存在的静态地图进行比较，实现定位。

## Sensor-based Localization Techniques

### 坐标系定义

**大地坐标系 World Geodetic Coordinate System 1984 (WGS1984)**：为GPS全球定位系统建立的坐标系统，原点在地球质心，Z轴指向BIH（国际时间服务机构）1984.0定义的协定地球极（CTP）方向，X轴指向BIH1984.0的零度子午面和CTP赤道的交点，Y轴和Z、X轴构成右手坐标系。其参数为经度、纬度、海拔高度。

**地心惯性坐标系 The Earth-centered inertial (ECI)**：地心惯性坐标系是牛顿运动定律适用的非加速参考系，原点在地心，ox轴过0经线与赤道焦点，oy轴过90经线与赤道交点，oz轴指向北极星。

**地心地固参考系 Earth-Centered, Earth-Fixed (ECEF)**：ECEF坐标系与地球固联，且随着地球转动。坐标原点为地球质心。X轴通过格林尼治线和赤道线的交点，正方向为原点指向交点方向。Z轴通过原点指向北极。Y轴与X、Z轴构成右手坐标系。

### GPS/IMU based Techniques

GNSS依赖于至少四个卫星来低成本的估计全局位置。标准的GPS平均精度不满足需求，但是可以用differential GPS (DGPS)、assisted GPS (AGPS)或real-time kinematic (RTK)来提升。

<figure><img src="../../../.gitbook/assets/image (637).png" alt=""><figcaption></figcaption></figure>

A DGPS utilizes measurements from an on-board vehicle GPS unit and from a GPS unit on a fixed infrastructure unit with a known location. As GPS error is correlated between two nearby GPS units, calculating this correlated error can be used to eliminate the error of on-board vehicle GPS to improve its accuracy. DPGS uses the known position of the fixed infrastructure unit to calculate the local error in the GPS position measurement periodically. This correction is then broadcasted to the on-board vehicle GPS units to adjust their own GPS estimate to achieve an average accuracy in the range of 1–2 m.

AGPS uses information from a cellular network to reduce delays in obtaining a position from the satellite as well as increase the signal coverage by reducing acquisition thresholds. 精度比DGPS低。

An RTK-GPS utilizes dual-frequency GPS receivers to estimate the relative position of a vehicle with respect to the position of a base station with known position. In this case, the relative position is estimated based on carrier phase measurement of GPS signals to achieve centimeter level accuracy.

The time-to-first-fix (TTFF), which refers to the time required to obtain the signal from the satellites and acquire an initial position estimate after a GPS unit is turned. Typically, TTFF for a GPS unit can be up to 12.5 min if the unit was completely turned off. However, this can be partially mitigated by: 1) keeping the clock operational whilst the unit is off; 2) utilizing a stand-by mode; or 3) acquiring the satellite almanac information from a cellular network with AGPS methods.

IMUs use a combination of accelerometers and gyroscopes to measure linear accelerations and vehicle angular velocities, respectively. This information can be used to calculate the trajectory of the vehicle as it travels. For this reason, IMUs have been used to estimate vehicle position relative to its initial position, in a method known as dead reckoning. However, the main problem with IMUs is accumulated errors where the measured position drifts further away from its true position as the vehicle travels. This problem can be overcome by correcting the estimated position using other sensors, to avoid accumulated drift and to provide global positioning.

\[A sensor fusion approach for localization with cumulative error elimination]将IMU和GPS融合，行驶408米，类似均方根误差达7.2m，小于原本IMU的22.3m和GPS的13.2m。但是，这种精度依旧不满足自动驾驶的需求。

### Camera-based Techniques

> Vehicle localization in urban environments using feature maps and aerial images.
>
> Vision-based precision vehicle localization in urban environments.
>
> Robust visual odometry for vehicle localization in urban environments.
>
> Autonomous vehicle technologies: Localization and mapping.
>
> Sensor fusion-based low-cost vehicle localization system for complex urban environments.

### RADAR-based Techniques

A Radar sensor is a ranging sensor which utilizes radio waves. Radar functions by emitting periodic radio waves which can bounce off obstacles back to the receiver and distance to target is measured from the time of arrival of radio waves. Each radio wave provides a single range measurement which gives the distance to the obstacle that reflected it back to the receiver. Radars also have relatively low power consumption. Even lower power requirements can be achieved by frequency modulation continuous wave (FMCW)-based Radars, which use continuous Radar signals rather than the periodic ones used in traditional pulse-based Radar systems, however, the accuracy is typically lower than that of pulse-based radar systems

Radar maps are dependent on the quality of discernible features available, which can cause errors when such features are not available.

> [http://wiki.openstreetmap.org/wiki/Micromapping](http://wiki.openstreetmap.org/wiki/Micromapping)
>
> Vehicle localization with low cost radar sensors.
>
> Mobile ground-based radar sensor for localization and mapping: An evaluation of two approaches.

### LiDAR-based Techniques

<figure><img src="../../../.gitbook/assets/image (610).png" alt=""><figcaption></figcaption></figure>

A LiDAR sensor measures distance to a target using multiple laser beams which each measures the distance to the target, based on the time of arrival of the signal back at the receiver, as well as the infrared intensity of the obstacle.

> Map-based precision vehicle localization in urban environments
>
> Feature detection for vehicle localization in urban environments using a multilayer LIDAR
>
> Robust vehicle localization in urban environments using probabilistic maps
>
> Ground-edge-based LIDAR localization without a reflectivity calibration for autonomous driving
>
> Robust vehicle localization using entropy-weighted particle filter-based data fusion of vertical and road intensity information for a large scale urban area
>
> Robust LIDAR localization using multiresolution Gaussian mixture maps for autonomous driving
>
> Visual localization within LIDAR maps for automated urban driving
>
> Horizontal/vertical LRFs and GIS maps aided vehicle localization in urban environment

### Ultrasonic-based Techniques

Ultrasonic sensors can scan the environment by utilizing a mechanical wave of oscillating pressure which can propagate through air or other materials. Distance to target can be measured based on the time of arrival of the signals back to the receivers. Ultrasonic sensors were chosen due to their high performance with low electric power consumption and low cost. However, due to inaccurate extracting of feature points, the localization process could take very long times. Average processing times of 10.65 s were observed, thereby making the technique unsuitable for high-speed vehicle applications. Also, the associated long average processing time causes accumulated errors due to measurements from the sensors such as IMU. Moreover, the detection range of the ultrasonic sensors is limited to 3 m which is not sufficient for obstacle detection system of an autonomous system.

\[Simultaneous localization and mapping of a wheel-based autonomous vehicle with ultrasonic sensors]

### Discussion

Suitability of the techniques for autonomous vehicles is based on the robustness and reliability as well as capability for in-lane localization accuracy. The required accuracy for in-lane localization is taken as 30 cm.

Integrated GPS/IMU/camera localization systems provide accuracy up to 73 cm, however, further improvements are needed to offer the accuracy and robustness required for fully autonomous vehicles. Radar sensor-based techniques offer cheaper localization systems compared to LiDAR-based systems and can meet the accuracy requirements for autonomous vehicles. However, the robustness of these methods remains an obstacle for implementation. In contrast, LiDAR can offer high accuracy but at a significantly higher cost compared to Radar. Therefore, for LiDAR to be a commercially feasible option further technological advances would be required to reduce the cost or alternatively the approach used in \[Visual localization within LIDAR maps for automated urban driving] could be used to take advantage of the high accuracy and robustness of LiDARbased maps, but keep the cost of autonomous vehicles low by equipping them with cameras instead of LiDAR sensors. This type of approach could be the key to achieving high accuracy and formulating low-cost solutions but robust performance in different environment conditions is still a challenge due to limitations of camera systems.

<figure><img src="../../../.gitbook/assets/image (645).png" alt=""><figcaption></figcaption></figure>

> \[26] A sensor fusion approach for localization with cumulative error elimination
>
> \[27] Vision-based precision vehicle localization in urban environments
>
> \[28] Robust visual odometry for vehicle localization in urban environments
>
> \[29] Autonomous vehicle technologies: Localization and mapping
>
> \[30] Sensor fusion-based low-cost vehicle localization system for complex urban environments
>
> \[16] Vehicle localization in urban environments using feature maps and aerial images
>
> \[32] Mobile ground-based radar sensor for localization and mapping: An evaluation of two approaches
>
> \[31] Vehicle localization with low cost radar sensors
>
> \[33] Localizing ground penetrating RADAR: A step toward robust autonomous ground vehicle localization
>
> \[35] Feature detection for vehicle localization in urban environments using a multilayer LIDAR
>
> \[21] Map-based precision vehicle localization in urban environments
>
> \[36] Robust vehicle localization in urban environments using probabilistic maps
>
> \[37] Ground-edge-based LIDAR localization without a reflectivity calibration for autonomous driving
>
> \[38] Robust vehicle localization using entropy-weighted particle filter-based data fusion of vertical and road intensity information for a large scale urban area
>
> \[39] Robust LIDAR localization using multiresolution Gaussian mixture maps for autonomous driving
>
> \[40] Visual localization within LIDAR maps for automated urban driving
>
> \[41] Horizontal/vertical LRFs and GIS maps aided vehicle localization in urban environment
>
> \[42] Simultaneous localization and mapping of a wheel-based autonomous vehicle with ultrasonic sensors

## Cooperative Localization Techniques

可以通过V2V和V2I通讯系统来获得off-borad信息，来增强传感器信息，来提升定位准确性、鲁棒性和可信度。在这种系统中，车辆从其他车辆处获得如速度、朝向和位置等自我状态信息，即V2V，从设施出获得如天气变化或遮挡物等环境信息，即V2I。

To calculate the distance to the broadcaster only, one radio signal is required, while calculating a relative position requires three radio signals in case of a 2-D localization or four in the case of 3-D localization. In general, the positioning and ranging systems can work based on four principles, namely time-of-arrival (TOA), time-differenceof-arrival (TDOA), angle-of-arrival (AOA), and radio-signalstrength (RSS).

<figure><img src="../../../.gitbook/assets/image (660).png" alt=""><figcaption></figcaption></figure>

### Vehicle-to-Vehicle Localization Techniques

<figure><img src="../../../.gitbook/assets/image (643).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (595).png" alt=""><figcaption></figcaption></figure>

> \[53] A new decentralized Bayesian approach for cooperative vehicle localization based on fusion of GPS and VANET based inter-vehicle distance measurement
>
> \[52] Cooperative vehicle positioning via V2V communications and onboard sensors
>
> \[54] Co-operative vehicle localization algorithm—Evaluation of the COVEL approach
>
> \[55] Cooperative positioning in vehicular ad-hoc networks supported by stationary vehicles
>
> \[56] Vehicle localization in VANETs using data fusion and V2V communication
>
> \[57] F. Ahammed, J. Taheri, A. Y. Zomaya, and M. Ott, International Conference on Mobile and Ubiquitous Systems: Computing, Networking, and Services. Berlin, Germany: Springer, 2010.
>
> \[58] Fuzzy logic based localization for vehicular ad hoc networks

### Vehicle-to-Infrastructure Localization Techniques

<figure><img src="../../../.gitbook/assets/image (586).png" alt=""><figcaption></figcaption></figure>

> \[61] Vehicle localization system based on IR-UWB for V2I applications
>
> \[62] A localization algorithm based on V2I communications and AOA estimation
>
> \[63] High accuracy GPS-free vehicle localization framework via an INS-assisted single RSU
>
> \[64] RF infrastructure cooperative systems for in lane vehicle localization
>
> \[65] Feasibility study of 5G-based localization for assisted driving
>
> \[69] Vehicle localization and velocity estimation based on mobile phone sensing
>
> \[70] UPS: Combatting urban vehicle localization with cellular-aware trajectories
>
> \[71] Inter-vehicle sensor fusion for accurate vehicle localization supported by V2V and V2I communications,
>
> \[72] Collaborative vehicle selflocalization using multi-GNSS receivers and V2V/V2I communications

### Discussion

基于V2I和V2X的协同定位方法需要解决通讯延迟、丢包、路由协议等问题，还有在较差通讯条件下鲁棒性的问题，还需要保证通讯过程中的数据安全。

Among the sensor-based techniques, LiDAR-based techniques can achieve very high accuracy and robustness at the cost of high power requirement and costly sensor with very limited performance in harsh environment conditions. It was also shown that the methods such as camera-based localization or Localizing ground penetrating radar (LGPR) techniques can offer adequate accuracy for autonomous vehicles at a lower cost at the expense of the lack of robustness against variation of environmental conditions. In general, the main limitations of all sensor-based techniques are their limited line of sight, out of range environmental information, and limited operation in harsh environment.

V2V techniques have presented great potential to address such limitations of sensor-based techniques. Currently, the only available V2V localization technique is multilateration which can significantly improve the accuracy of GPS/IMU sensor-based technique when an adequate number of connected vehicles are available.

On the other hand, V2I localization techniques can mitigate the difficulty of guaranteeing an adequate number of signals and accuracy of broadcasted position estimates in V2V localization techniques as the Road side units (RSUs) are installed in known fixed positions and the RSU density can be optimized to provide adequate accuracy and robustness. However, V2I techniques require costly infrastructure implementations to ensure high accuracy and robustness. Alternatively, utilizing existing cellular networks provides a method for V2I localization without the need for implementation of new costly infrastructure. Similar to V2V techniques, the quality of service in V2I networks is also a limitation for implementation as noise can affect the received signals causing erroneous inputs and packet loss and latency can cause degradation in performance or failure of localization systems.

<figure><img src="../../../.gitbook/assets/image (569).png" alt=""><figcaption></figcaption></figure>

## Conclusion

从算法表现来看，LiDAR技术能够达到最好的定位效果，但是成本和计算消耗较大。因此，可以发展用LGPR和相机在LiDAR地图中定位的技术。

将V2V技术与onboard传感器结合，能够很有效地提升定位系统的准确性、鲁棒性和可靠性。协同定位能够提供超视距和更大范围的信息，对环境条件变化更鲁棒。但是V2V技术受到共享位置估计和相连车辆数量的限制，已知位置的设施可以用来共享位置，但是当前的协同定位系统还需要在真实场景中进行测试，还需要考虑通讯网络的传输强度、包传输频率等指标，需要考虑数据安全。网络失效的情况也需要考虑。
