# survey and analysis

### 论文总结

| 文献                                        | 来源           | 主要贡献                                                                                                                                                                              |
| ----------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IR in VisLoc                              | 3DV 2020     | 本篇论文调研了定位表现与图像检索表现之间的关联。证明：（1）位姿估计需要描述子对视觉敏感，场景识别表现和位姿估计表现无关；（2）检索表现与基于局部地图的定位表现无关；（3）检索表现（R@K）与定位表现一般正相关。因此，作者展望可以从位姿估计的角度设计专门的图像表征。                                             |
| pGT in VisLoc                             | ICCV 2021    | 本论文证明了用于生成伪真值的reference算法会对评测算法的排名产生巨大影响。所以算法的评估需要考虑特定算法是否与reference算法相似，更相似的算法更容易复现reference算法的伪真值。并且证明现有很多结论是在没有考虑以上问题时得到的，所以并不客观。                                              |
| Localization in highway scenarios         | sensors 2022 | 本篇论文将高速场景下的视觉定位方法分为三类：1. Road Level Localization，即判断车辆行驶在那条路径上（一般为map-matching方法）；2. Ego-Lane Level Localization，即车道线检测，判断车辆在那条车道线上；3. Lane-Level Localization，即基于地图的匹配定位，解算具体位置。 |
| Map Localization using LiDARs and Cameras | arxiv 2022   | 本篇论文将定位任务分为场景识别和metric map localization两个步骤，对每个分别从基于视觉、基于LiDAR和基于cross-modal三个角度进行调研。得到结论：视觉信息进行可靠的场景识别；点云地图对于metric localization很重要；双目相机和LiDAR点云地图的搭配可以达到不错的效果，且传感器成本较低。         |
| SOTA Localization Techniques              | JIoT 2018    | 本篇综述将定位任务分为基于onboard传感器的单车定位和基于V2V、V2I的协同定位。得到结论：LiDAR表现最好，但是成本高、计算消耗大；在LiDAR地图中用LGPR和相机进行定位可以实现表现和成本的良好平衡；用V2V和V2I能获得超视距和大范围信息，有利于定位任务的鲁棒性，但仍需探索。                                |
| LiDAR Localization                        | IV 2020      | 本篇综述将基于Lidar的定位算法分为三类：3D registration based, 3D features based, 和3D deep learning based方法。                                                                                        |
