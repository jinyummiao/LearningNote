# deep VO/SLAM

## 检索目录

iMAP: Implicit Mapping and Positioning in Real-Time (ICCV 2021)

DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras (NIPS 2021)

## 算法总结

| 算法         | 来源        | 特点                                                                                                                                          |
| ---------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| iMAP       | ICCV 2021 | iMAP采用MLP结构来隐式表示场景，算法包括建图和跟踪两个过程，建图过程中联合优化网络参数及关键帧的相机位姿，跟踪过程中固定网络优化当前相机位姿。损失函数包括颜色信息相关的光度损失和深度相关的集合损失。为了减少训练消耗，在像素和图像级别采用了activate sampling |
| DROID-SLAM | NIPS 2021 | 迭代优化相机位姿和深度。参考correlation pyramid，根据correlation feature和context feature修正光流，在BDA层中修正位姿和深度，让其重投影结果与光流一致。整个系统是一个端到端的基于深度学习的SLAM系统。            |

###
