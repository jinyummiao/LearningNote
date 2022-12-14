---
description: A Survey on 3D LiDAR Localization for Autonomous Vehicles
---

# \[IV 2020] LiDAR Localization

{% embed url="https://ieeexplore.ieee.org/document/9304812" %}

## Introduction

In order for a car to drive autonomously, the first challenge that a traditional pipeline would try to solve is to localize the car. Localization in this context means: finding the position and orientation of the vehicle inside of a map.

作者将基于3D Lidar数据的自动驾驶定位算法分为三类：

* 3D Registration Based Methods: Usually combined with a map that was built offline, these methods take advantage of the advances that were achieved in 3D pointclouds registration. These methods can be seen as ”dense” methods, since they take advantage of all the points present in the LiDAR data.
* 3D Features Based Methods: these approaches design relevant features in the 3D space that are then used to calculate the displacements between successive scans. These methods can be seen as ”sparse” method, since they only use a select number of points in the LiDAR data.
* 3D Deep Learning Based Methods.

## 3D LiDAR Localization for Autonomous Driving Cars

### 3D Registration based methods

Registration transforms a pair of point clouds in order to align them in the same coordinates frame, making it possible to deduct the transformation between both scans. In the context of ADC localization, registration can be used in two ways: (1) By combining the incoming scans with portions of a prebuilt point cloud map in order to localize the vehicle, or (2) By combining successive LiDAR scans in order to calculate the odometry of the vehicle.

基于3D registration的算法中最流行的是Iterative Closest Point (ICP)算法。In the ICP algorithm, a transformation between a source and target point cloud is iteratively optimized by minimizing an error metric between the points of both point clouds.

但是，ICP算法最终被3D Normal Distribution Transform (NDT)算法超越了。Similarly to the ICP algorithm, a transformation between a source and target point cloud is iteratively optimized. But in this case, the error that is being minimized is by first transforming the point clouds into a probability density functions which can be used with Newton’s algorithm to find the spatial transformation between them.

然而，ICP算法和NDT算法都难以满足实时性要求，并且为了算法得到准确的结果，需要提供一个初值来避免达到局部最小值，这一般需要其他传感器来提供一个初始位姿估计。

IMLS-SLAM算法包含三个步骤：第一部，对scan进行聚类，剔除小的cluster，来剔除动态物体；第二步，根据每个点的可观性，对scan进行下采样；最后，利用Implicit Moving Least Square (IMLS)表征方法，基于scan-to-model策略来优化变换关系。

还有一种惯用的预处理方法是在点云注册前，计算点云的surfel (SURFace ELement) representation。

Collar Line Segments (CLS) construction is a useful pre-processing method that makes it possible to achieve a good level of accuracy when aligning point clouds. In \[Collar line segments for fast odometry estimation from velodyne point clouds], the LiDAR scans are transformed into line clouds, by sampling line segments between neighbouring points from neighbouring rings. These line clouds are then aligned using an iterative approach: First, the center points of the generated lines are calculated. These points are then used to find the transformation between successive scans by finding the lines in the target pointcloud whose center is closest to the lines in the source pointcloud. Additional post processing is applied to boost the accuracy, using a global optimization based previous transformations.

降低LiDAR数据的维度也很有效。有方法\[Dlo: Direct lidar odometry for 2.5 d outdoor environment]将输入的scans投影到一个2.5D的grid map中，grip map有占据情况和高度。

* Jens Behley and Cyrill Stachniss. Efficient surfel-based slam using 3d laser range data in urban environments.
* Andrea Censi. An icp variant using a point-to-line metric. 2008.
* Xieyuanli Chen, Andres Milioto Emanuele Palazzolo, Philippe Gigu\` ere, Jens Behley, and Cyrill Stachniss. Suma + + : Efficient lidar-based semantic slam. 2019.
* Yang Chen and G ́ erard Medioni. Object modelling by registration of multiple range images. Image and vision computing, 10(3):145–155, 1992.
* Jean-Emmanuel Deschaud. IMLS-SLAM: scan-to-model matching based on 3d data. CoRR, abs/1802.08633, 2018.
* Kok-Lim Low. Linear least-squares optimization for point-to-plane icp surface registration. Chapel Hill, University of North Carolina, 4(10):1–3, 2004.
* Dmitri Kovalenko, Mikhail Korobkin, and Andrey Minin. Sensor aware lidar odometry. In 2019 European Conference on Mobile Robots (ECMR), pages 1–6. IEEE, 2019.
* Martin Magnusson. The three-dimensional normal-distributions transform: an efficient representation for registration, surface analysis, and loop detection. PhD thesis, ̈ Orebro universitet, 2009.
* Martin Magnusson, Achim Lilienthal, and Tom Duckett. Scan registration for autonomous mining vehicles using 3d-ndt. Journal of Field Robotics, 24(10):803–827, 2007.
* Ellon Mendes, Pierrick Koch, and Simon Lacroix. Icp-based posegraph slam. 2016 IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR), pages 195–200, 2016.
* Chanoh Park, Soohwan Kim, Peyman Moghadam, Clinton Fookes, and Sridha Sridharan. Probabilistic surfel fusion for dense lidar mapping. 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), pages 2418–2426, 2017.
* Szymon Rusinkiewicz and Marc Levoy. Efficient variants of the icp algorithm. In 3dim, volume 1, pages 145–152, 2001.
* Aleksandr Segal, Dirk Haehnel, and Sebastian Thrun. Generalizedicp. In Robotics: science and systems, volume 2, page 435. Seattle, WA, 2009.
* Lu Sun, Junqiao Zhao, Xudong He, and Chen Ye. Dlo: Direct lidar odometry for 2.5 d outdoor environment. In 2018 IEEE Intelligent Vehicles Symposium (IV), pages 1–5. IEEE, 2018.
* Martin Velas, Michal Spanel, and Adam Herout. Collar line segments for fast odometry estimation from velodyne point clouds. 2016 IEEE International Conference on Robotics and Automation (ICRA), pages 4486–4495, 2016.
* Zhengyou Zhang. Iterative point matching for registration of free-form curves and surfaces, 1994.

### 3D Features based methods

In this section, we tackle the 3D localization methods based on 3D features extraction and matching. 3D features are interest points that represent recognizable areas that are consistent in time and space, such as corners and planes. These features are usually represented using a unique vector called feature descriptor, which can be used to match features in two different point clouds. By finding sufficient and consistent matches, we can calculate the transform between scans using an optimization method and thus construct an odometry measurement.

* Philipp Egger, Paulo VK Borges, Gavin Catt, Andreas Pfrunder, Roland Siegwart, and Renaud Dub ́ e. Posemap: Lifelong, multienvironment 3d lidar localization. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 34303437. IEEE, 2018.
* Yulan Guo, Ferdous Sohel, Mohammed Bennamoun, Min Lu, and Jianwei Wan. Rotational projection statistics for 3d local surface description and object recognition. International journal of computer vision, 105(1):63–86, 2013.
* X. Ji, L. Zuo, C. Zhang, and Y. Liu. Lloam: Lidar odometry and mapping with loop-closure detection based correction. In 2019 IEEE International Conference on Mechatronics and Automation (ICMA), pages 2475–2480, Aug 2019.
* K. Ji, H. Chen, H. Di, J. Gong, G. Xiong, J. Qi, and T. Yi. Cpfgslam:a robust simultaneous localization and mapping based on lidar in off-road environment. In 2018 IEEE Intelligent Vehicles Symposium (IV), pages 650–655, June 2018.
* W Shane Grant, Randolph C Voorhies, and Laurent Itti. Finding planes in lidar point clouds for real-time registration. In 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 4347–4354. IEEE, 2013.
* Kaustubh Pathak, Andreas Birk, Narunas Vaskevicius, Max Pfingsthorn, S ̈ oren Schwertfeger, and Jann Poppinga. Online threedimensional slam by registration of large planar surface segments and closed-form pose-graph relaxation. Journal of Field Robotics, 27(1):52–84, 2010.
* Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast point feature histograms (fpfh) for 3d registration. In 2009 IEEE international conference on robotics and automation, pages 3212–3217. IEEE, 2009.
* Radu Bogdan Rusu, Gary Bradski, Romain Thibaux, and John Hsu. Fast 3d recognition and pose using the viewpoint feature histogram. In Proceedings of the 23rd IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Taipei, Taiwan, October 2010.
* Tixiao Shan and Brendan Englot. Lego-loam: Lightweight and groundoptimized lidar odometry and mapping on variable terrain. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4758–4765. IEEE, 2018.
* Bastian Steder, Radu Bogdan Rusu, Kurt Konolige, and Wolfram Burgard. Narf: 3d range image features for object recognition.
* Keisuke Yoneda, Hossein Tehrani Niknejad, Takashi Ogawa, Naohisa Hukuyama, and Seiichi Mita. Lidar scan feature for localization with highly precise 3-d map. 2014 IEEE Intelligent Vehicles Symposium Proceedings, pages 1345–1350, 2014.
* Ji Zhang and Sanjiv Singh. Loam: Lidar odometry and mapping in real-time.
* Jiarong Lin and Fu Zhang. A fast, complete, point cloud based loop closure for lidar odometry and mapping, 09 2019

### 3D Deep Learning based methods

Usually formulated as a regression problem, methods involving deep learning can either try to solve this task in an end-to-end fashion by using the raw point clouds as inputs and directly predicting the displacements of the vehicle using a single network, or by trying to substitute certain parts of the pre-established classical pipelines that could benefit for the generalizations possible with deep learning networks.

* Renaud Dub ́ e, Andrei Cramariuc, Daniel Dugas, Juan Nieto, Roland Siegwart, and Cesar Cadena. Segmap: 3d segment mapping using data-driven descriptors. arXiv preprint arXiv:1804.09557, 2018.
* Renaud Dube, Daniel Dugas, Elena Stumm, Juan Nieto, Roland Siegwart, and Cesar Cadena. Segmatch: Segment based place recognition in 3d point clouds. pages 5266–5272, 05 2017.
* Renaud Dub ́ e, Mattia G Gollub, Hannes Sommer, Igor Gilitschenski, Roland Siegwart, Cesar Cadena, and Juan Nieto. Incremental-segmentbased localization in 3-d point clouds. IEEE Robotics and Automation Letters, 3(3):1832–1839, 2018.
* G. Elbaz, T. Avraham, and A. Fischer. 3d point cloud registration for localization using a deep neural network auto-encoder. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2472–2481, July 2017.
* Andrei Cramariuc, Renaud Dub ́ e, Hannes Sommer, Roland Siegwart, and Igor Gilitschenski. Learning 3d segment descriptors for place recognition. arXiv preprint arXiv:1804.09270, 2018.
* Younggun Cho, Giseop Kim, and Ayoung Kim. Deeplo: Geometryaware deep lidar odometry. arXiv preprint arXiv:1902.10562, 2019.
* Qing Li, Shaoyang Chen, Cheng Wang, Xin Li, Chenglu Wen, Ming Cheng, and Jonathan Li. Lo-net: Deep real-time lidar odometry. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8473–8482, 2019.
* Weixin Lu, Guowei Wan, Yao Zhou, Xiangyu Fu, Pengfei Yuan, and Shiyu Song. Deepvcp: An end-to-end deep neural network for point cloud registration. In Proceedings of the IEEE International Conference on Computer Vision, pages 12–21, 2019.
* Weixin Lu, Yao Zhou, Guowei Wan, Shenhua Hou, and Shiyu Song. L3-net: Towards learning based lidar localization for autonomous driving. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6389–6398, 2019.
* Weixin Lu, Guowei Wan, Yao Zhou, Xiangyu Fu, Pengfei Yuan, and Shiyu Song. Deepicp: An end-to-end deep neural network for 3d point cloud registration. arXiv preprint arXiv:1905.04153, 2019.
* Austin Nicolai, Ryan Skeele, Christopher Eriksen, and Geoffrey A Hollinger. Deep learning for laser based odometry estimation.
* Tim Tang, David Yoon, Franc ̧ois Pomerleau, and Timothy D Barfoot. Learning a bias correction for lidar-only motion estimation. In 2018 15th Conference on Computer and Robot Vision (CRV), pages 166173. IEEE, 2018.
* Wei Wang, Muhamad Risqi U Saputra, Peijun Zhao, Pedro Gusmao, Bo Yang, Changhao Chen, Andrew Markham, and Niki Trigoni. Deeppco: End-to-end point cloud odometry through deep parallel neural network. arXiv preprint arXiv:1910.11088, 2019.
* Huan Yin, Li Tang, Xiaqing Ding, Yue Wang, and Rong Xiong. Locnet: Global localization in 3d point clouds for mobile vehicles. In 2018 IEEE Intelligent Vehicles Symposium (IV), pages 728–733. IEEE, 2018.

## Evaluation and Discussion

<figure><img src="../../../.gitbook/assets/image (894).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (960).png" alt=""><figcaption></figcaption></figure>
