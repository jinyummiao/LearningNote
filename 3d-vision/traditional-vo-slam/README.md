# traditional VO/SLAM

## 检索目录

RP-VIO: Robust Plane-based Visual-Inertial Odometry for Dynamic Environments (IROS 2021)

## 算法总结

| 算法        | 来源         | 特点                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RP-VIO    | IROS 2021  | 在VINS-Mono的基础上，去掉重定位和回环模块；只检测场景中平面上的特征，将平面的单应性约束加入优化过程；IMU数据作为图像帧间的先验，选择有足够的视差图像作为关键帧，进行滑窗优化；提出一个仿真数据集，有动态干扰，IMU激励充足。                                                                                                                                                                                                                                                                                       |
| ORB-SLAM3 | TRO 2021   | ORB-SLAM3, the most complete open-source library for visual, visual–inertial, and multisession SLAM, with monocular, stereo, RGB-D, pin-hole, and fisheye cameras. The main contributions, apart from the integrated library itself, are the fast and accurate IMU initialization technique and the multisession map-merging functions that rely on a new place recognition technique with improved recall. |
| DSO       | TPAMI 2018 |                                                                                                                                                                                                                                                                                                                                                                                                             |

###
