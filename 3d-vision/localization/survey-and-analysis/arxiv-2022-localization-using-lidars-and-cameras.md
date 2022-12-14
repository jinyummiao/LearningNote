---
description: A Survey on Visual Map Localization Using LiDARs and Cameras
---

# \[arxiv 2022] Localization using LiDARs and Cameras

{% embed url="https://arxiv.org/abs/2208.03376" %}

## Abstract

We define visual map localization as a two-stage process. At the stage of place recognition, the initial position of the vehicle in the map is determined by comparing the visual sensor output with a set of geo-tagged map regions of interest. Subsequently, at the stage of map metric localization, the vehicle is tracked while it moves across the map by continuously aligning the visual sensors’ output with the current area of the map that is being traversed.

## Introduction

<figure><img src="../../../.gitbook/assets/image (617).png" alt=""><figcaption></figcaption></figure>

## Background

### Visual Place Recognition

Visual place recognition represents the task of finding the best match possible for a visual input sensor, in a pre-built database, as efficiently as possible.

<figure><img src="../../../.gitbook/assets/image (583).png" alt=""><figcaption><p>视觉地图类型</p></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (659).png" alt=""><figcaption></figcaption></figure>

### Metric Map Localization

Once the initial position in the map is found, the vehicle must now keep track of its current position on the map.

## Visual Place Recognition

分为三类讨论VPR算法：LiDARs、相机和cross-model方法。

### LiDAR Based Methods

基于LiDAR的VPR算法一般基于点云关键点检测和匹配。

> Image representations beyond histograms of gradients: The role of gestalt descriptors.
>
> Place recognition using keypoint voting in large 3d lidar datasets.
>
> Large scale place recognition in 2d lidar scans using geometrical landmark relations.
>
> Segmatch: Segment based place recognition in 3d point clouds.
>
> Locus: Lidar-based place recognition using spatiotemporal higher-order pooling.
>
> Local descriptor for robust place recognition using lidar intensity.
>
> Robust place recognition using an imaging lidar.
>
> Bvmatch: Lidar-based place recognition using bird’s-eye view images.
>
> Scan context: Egocentric spatial descriptor for place recognition within 3d point cloud map.
>
> Locnet: Global localization in 3d point clouds for mobile vehicles.
>
> 3d lidar-based global localization using siamese neural network.
>
> Rinet: Efficient 3d lidar-based place recognition using rotation invariant neural network.
>
> Lidar-based initial global localization using two-dimensional (2d) submap projection image (spi).
>
> Sticky-localization: Robust end-to-end relocalization on point clouds using graph neural networks.
>
> Semantic graph based place recognition for 3d point clouds.

### Camera Based Methods

基于视觉的VPR算法主要依赖于提取图像特征并借此计算不同视角下图像的相似度。

> Cross-view image matching for geo-localization in urban environments.
>
> Cvmnet: Cross-view matching network for image-based ground-to-aerial geolocalization.
>
> Netvlad: Cnn architecture for weakly supervised place recognition.
>
> Lending orientation to neural networks for cross-view geo-localization.
>
> Revisiting street-to-aerial view image geo-localization and orientation estimation.
>
> Optimal feature transport for cross-view image geo-localization.
>
> Soft exemplar highlighting for cross-view image-based geo-localization.
>
> Vigor: Cross-view image geo-localization beyond one-to-one retrieval.
>
> Spatial-aware feature aggregation for image based cross-view geo-localization.
>
> Cross-view geo-localization with layer-to-layer transformer.
>
> Transgeo: Transformer is all you need for cross-view image geo-localization.

### Cross-Modal Methods

这类方法是一般基于开源地图（比如卫星地图和OSM）来进行LiDAR定位。

> Openstreetmap-based lidar global localization in urban environment without a prior lidar map.
>
> Get to the point: Learning lidar place recognition and metric localisation using overhead imagery.

## Metric Map Localization

同样分为三类讨论Metric Map Localization算法：LiDARs、相机和cross-model方法。

### LiDAR Based Methods

> Mapbased precision vehicle localization in urban environments.
>
> Lidar scan feature for localization with highly precise 3-d map.
>
> Ground-edge-based lidar localization without a reflectivity calibration for autonomous driving.
>
> Robust lidar localization for autonomous driving in rain.
>
> Fast lidar localization using multiresolution gaussian mixture maps.
>
> Autonomous vehicle self-localization based on multilayer 2d vector map and multi-channel lidar.
>
> Lidar-based lane marking detection for vehicle positioning in an hd map.
>
> Lidar-based road signs detection for vehicle localization in an hd map.
>
> Pole-curb fusion based robust and efficient autonomous vehicle localization system with branch-and-bound global optimization and local grid map method.
>
> Lol: Lidar-only odometry and localization in 3d point cloud maps.
>
> Collaborative semantic perception and relative localization based on map matching.
>
> Overlapnet: a siamese network for computing lidar scan similarity with applications to loop closing and localization.
>
> L3net: Towards learning based lidar localization for autonomous driving.
>
> Attentionbased vehicle self-localization with hd feature maps.

### Camera Based Methods

基于视觉的VPR算法一般基于视觉特征或视觉landmarks的提取和匹配。

> Metric localization using google street view.
>
> Leveraging the osm building data to enhance the localization of an urban vehicle.
>
> Map-based probabilistic visual self-localization.
>
> Monocular camera localization in 3d lidar maps.
>
> Satellite image-based localization via learned embeddings.
>
> Pole-based localization for autonomous vehicles in urban scenarios.
>
> Cross-view matching for vehicle localization by learning geographically local representations.
>
> Grad-cam: Visual explanations from deep networks via gradient-based localization.

### Cross-Modal Methods

在基于相机的地图（比如卫星地图或OSM地图）使用LiDAR点云定位：

> Global localization on openstreetmap using 4-bit semantic descriptors.
>
> Lidar-osm-based vehicle localization in gps-denied environments by using constrained particle filter.
>
> Any way you look at it: Semantic crossview localization and mapping with lidar.
>
> Rsl-net: Localising in satellite images from a radar on the ground.
>
> Selfsupervised localisation between range sensors and overhead imagery.

在基于LiDAR的地图中使用相机数据定位，一般使用双目相机，因为它可以将2D数据转换到3D中：

> 3d point cloud map based vehicle localization using stereo camera.
>
> Stereo camera localization in 3d lidar maps.

在LiDAR地图中使用单目相机定位更具有挑战，因为它不具备任何深度信息和3D信息。

> Cmrnet: Camera to lidar-map registration.
>
> Cmrnet++: Map and camera agnostic monocular visual localization in lidar maps.

## Evaluation and Discussion

### Datasets

CVUSA、CVACT、KITTI、KITTI-360

### Metrics

对于场景识别：

Recall@1%: represents the percentage of cases in which the correct query sample is ranked within top 1 percentile of possible samples.&#x20;

Recall@1: represents the percentage of cases in which the correct query sample is ranked first among possible samples.&#x20;

F1 Max Score: measures the accuracy of the predicted samples using $$F_1=2*\frac{r*p}{r+p}$$

对于metric map localization：

Metric error: reflects the error accumulation or drift of the localization using $$E_m=\frac{\sum^N_{i=1}|p_i-\hat{p_i}|}{N}$$ where $$p_i$$ and $$\hat{p_i}$$ are the predicted and ground truth pose.

### Place recognition

<figure><img src="../../../.gitbook/assets/image (581).png" alt=""><figcaption></figcaption></figure>

> \[32] Cvmnet: Cross-view matching network for image-based ground-to-aerial geolocalization.
>
> \[43] Lending orientation to neural networks for cross-view geo-localization.
>
> \[86] Revisiting street-to-aerial view image geo-localization and orientation estimation.
>
> \[57] Optimal feature transport for cross-view image geo-localization.
>
> \[30] Soft exemplar highlighting for cross-view image-based geo-localization.
>
> \[55] Spatial-aware feature aggregation for image based cross-view geo-localization.
>
> \[77] Cross-view geo-localization with layer-to-layer transformer.
>
> \[85] Transgeo: Transformer is all you need for cross-view image geo-localization.

<figure><img src="../../../.gitbook/assets/image (626).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (573).png" alt=""><figcaption></figcaption></figure>

### Metric Map Localization

<figure><img src="../../../.gitbook/assets/image (607).png" alt=""><figcaption></figcaption></figure>

> \[6] Map-based probabilistic visual self-localization.
>
> \[76] Global localization on openstreetmap using 4-bit semantic descriptors.
>
> \[21] Lidar-osm-based vehicle localization in gps-denied environments by using constrained particle filter.
>
> \[34] Satellite image-based localization via learned embeddings.
>
> \[47] Any way you look at it: Semantic crossview localization and mapping with lidar.
>
> \[10] Cmrnet: Camera to lidar-map registration.
>
> \[9] Cmrnet++: Map and camera agnostic monocular visual localization in lidar maps.
>
> \[36] Stereo camera localization in 3d lidar maps.
>
> \[88] Multimodal localization: Stereo over lidar map.
>
> \[12] Overlapnet: a siamese network for computing lidar scan similarity with applications to loop closing and localization.

## Conclusion

We found that cameras can be very effective and accurate in solving the place recognition task, using deep learning mainly, making it possible to find the initial position of a vehicle in a pre-built map much more efficiently. For the metric map localization stage, point cloud maps are still essential in order to produce the most accurate results, regardless of which sensor was equipped onto the vehicle. However, the crossmodal method using stereo camera sensors and LiDAR point cloud maps seems to produce the most promising results in terms of metric map localization performance. In addition, this combination can lead to a drastic cost reduction in production and increase the accessibility of such vehicles to the general public by making it easier and cheaper to produce smart vehicles capable of accurately localizing themselves in prebuilt visual maps.
