---
description: 'HyperMap: Compressed 3D Map for Monocular Camera Registration'
---

# \[ICRA 2021] HyperMap

## Abstract

We address the problem of image registration to a compressed 3D map. While this is most often performed by comparing LiDAR scans to the point cloud based map, it depends on an expensive LiDAR sensor at run time and the large point cloud based map creates overhead in data storage and transmission. Recently, efforts have been underway to replace the expensive LiDAR sensor with cheaper cameras and perform 2D-3D localization. In contrast to the previous work that learns relative pose by comparing projected depth and camera images, we propose HyperMap, a paradigm shift from online depth map feature extraction to offline 3D map feature computation for the 2D-3D camera registration task through end-to-end training. In the proposed pipeline, we first perform offline 3D sparse convolution to extract and compress the voxelwise hypercolumn features for the whole map. Then at run-time, we project and decode the compressed map features to the rough initial camera pose to form a virtual feature image. A Convolutional Neural Network (CNN) is then used to predict the relative pose between the camera image and the virtual feature image. In addition, we propose an efficient occlusion handling layer, specifically designed for large point clouds, to remove occluded points in projection. Our experiments on synthetic and real datasets show that, by moving the feature computation load offline and compressing, we reduced map size by 87 − 94% while maintaining comparable or better accuracy.

## Introduction

In this paper, we propose a novel strategy to learn on-map convolutional features to compress the map and preserve the registration performance. Given a noisy initial camera pose, our method predicts the relative 6 degree-of-freedom (DoF) pose of the camera by comparing the image captured by the camera to a virtual feature image created by projecting a 3D feature map to 2D.

如CMRNet之类的算法是先将点云地图投影到虚拟视角下，再将虚拟的深度图输入CNN得到特征图，作者将这一流程称为early projection。作者在本文中提出late projection，即预先计算并压缩体素化点云地图的3D特征，然后对特征进行投影。本文用稀疏的3D卷积来提取高精度点云地图的特征。

The primary contributions of this paper is as follows: we propose “late projection” in contrast to “early projection” for the 2D-3D registration task. Our late projection strategy precomputes and compresses the 3D map features offline before online projection, which we refer to as a “HyperMap” due to the use of hypercolumn features. The proposed HyperMap outperforms the baseline in map size significantly while maintaining comparable or slightly better performance. Although we focus 2D-3D registration in this paper, we believe that the concept of “late projection” can be extended to and potentially benefit other map-related tasks.

In addition, we propose an efficient occlusion-handling layer that enables backpropagation from a projected feature image to 3D convolutional layers on a large-scale sparse point cloud map. This occlusion-handling layer is crucial to the scalability of our proposed HyperMap.

## Problem Formulation

In this paper, we propose the use of “late projection”, i.e. the computation of 3D features directly on the map in offline fashion before projecting into the 2D space to generate the projective virtual feature image.

<figure><img src="../../../.gitbook/assets/image (967).png" alt=""><figcaption></figcaption></figure>

The major challenges associated with “late projection” can be summarized as the following: Although we can reduce the map resolution by representing the local map statistics by features, the highdimensional features might reversely increase the total map size and takes more time to load and process; Due to the sparsity of the LiDAR map, points occluded in RGB images might appear in the projective feature images. Thus, we need a differentiable occlusion handling layer for the large-scale sparse point clouds. We use 3D feature compression and an occlusion handling layer to overcome the above challenge.

Our problem formulation is as follows. Let $$\theta_{init}$$ represent the 6-DoF initial camera pose, $$\theta_{gt}$$ represent the ground truth pose, $$I$$ represent the camera image and $$M$$ represent the 3D voxelized feature map. Assuming the camera intrinsics are fixed, our goal is to estimate the relative pose $$\Delta \theta$$ that aligns our initial estimate $$\theta_{init}$$ to $$\theta_{gt}$$. We can formulate this as an estimation problem where we seek to estimate the weights $$w$$ of a network $$\mathcal{G}$$, that predicts $$\Delta \theta$$ from $$\theta_{init}$$, $$I$$, and $$M$$. Let $$\circ$$ be the pose composition operator:

<figure><img src="../../../.gitbook/assets/image (989).png" alt=""><figcaption></figcaption></figure>

where $$\pi(\cdot)$$ is the 3D to 2D perspective projection function.

## Map Feature Extraction

In order to capture features with different frequencies, we apply the hypercolumn concept to 3D feature extraction. We use stride 2 for the first 3D convolutional block and 1 for all the other blocks. The receptive field of each layer expands as more convolutional layers are applied, and the feature dimension also increases correspondingly. Afterwards, we combine the multiple activations to form a hypercolumn feature vector for each occupied voxel to preserve both the precision of earlier layers and the capacity of later layers. The final voxel resolution was thus reduced by ratio two due to the stride 2 in the first block.

At training time, we first voxelize the whole raw point cloud map, crop the local map region using the initial pose, extract 3D features in the map coordinate frame, and then transform the cropped feature map to the camera coordinate frame. Afterwards, n layers of 3D sparse convolutional filters are applied to the voxelized local map, and the feature output from the n convolutional layers are concatenated to form a high-dimensional hypercolumn feature vector.

For a voxel $$v_i$$ in the map, the corresponding hypercolumn feature vector is first compressed to a lower dimension feature $$f_i \in R^m$$ (dimension 72 → 16) using another 3D sparse convolutional layer. After trained end-to-end, we apply K-means algorithm to all the $$f_i$$ in the map to obtain k centroids, and compute the cluster index $$d_i$$ of each voxel (As shown in Figure 3b, we use k = 16 in our experiments, so we only need 4 bits to represent the centroid index, $$d_i \in$$ 0, 1, 2, ..., 15). We then project the cluster index di to form a 2D virtual feature image, and recover the original feature $$f_i$$ from $$d_i$$ using the corresponding K-means centroids. Notice that map feature projection required retrieving the map feature data from the storage and thus projecting $$d_i$$ is cheaper and faster than projecting $$f_i$$ due to its small size.

In the map projection step, we project the voxel grids to form a depth map and concatenate the depth map to fi as an additional channel, so the final projective virtual feature image has m + 1-dimensions. This feature precomputation step reduces the required voxel resolution while preventing the performance drop. We use kernel size 3 for all the 3D sparse convolutional layers.

## Occlusion Handling

We use the voxel size to approximate the occupied neighborhood of the map points in 3D space, and projection of the occupied neighborhood should only contain the projections of the map points that are closer to the camera than the voxel center (with smaller depth value). This means that if a projective map point has some nearby pixels with smaller depth, it is likely that this voxel is occluded. We use efficient maxpooling filters to simulate the 2D occupied neighborhood. In order to apply the maxpooling layers, we first make the projective depth negative and set the empty pixels to the maximum negative depth value. The pixels with smaller depth values originally would be larger after this transformation, and thus will be kept after the maxpooling operation. Afterwards, we recover the original image by setting the empty pixels back to zero and inverting the sign of the depth map. The output, maxpooled depth map, is noted as $$M_r(p)$$ with kernel size r at pixel position p.

Let D(p) and R(p) be the depth map and its corresponding occlusion filter kernel size map, and f be the focal length. The map of the occlusion filter kernel size (in pixel) can be computed from the fixed voxel size in the map:&#x20;

<figure><img src="../../../.gitbook/assets/image (982).png" alt=""><figcaption></figcaption></figure>

Afterwards, we find the pyramid level with smallest $$M_r(p)$$ among all levels for each pixel, denoted as:

<figure><img src="../../../.gitbook/assets/image (988).png" alt=""><figcaption></figcaption></figure>

If rmin is larger than the R(p) at this pixel, it means that this pixel is occluded by a nearby pixel with smaller depth value and the corresponding feature value should be set to zero. Let F(p) be the virtual feature image. The final virtual feature image is computed by:

<figure><img src="../../../.gitbook/assets/image (973).png" alt=""><figcaption></figcaption></figure>

We choose δ = 0.5 so the occlusion filter is only effective when the occluded points are far away from the visible point. If several layers in $$M_r(p)$$ have the same pixel value, which happens when all the maxpooling layer outputs are dominated by a close-by nearer point, we pick the smallest r among them.

<figure><img src="../../../.gitbook/assets/image (950).png" alt=""><figcaption></figcaption></figure>

## &#x20;Camera Pose Prediction

CMRNet.

We add one additional tanh layer as an output layer to constrain the range of predicted translation and rotation.

## Experiments

We choose CMRNet as our baseline and compare with it in 0.1m, 0.2m and 0.4m voxel resolutions. We use n = 4, m = 16, k = 16, and a five-level occlusion pyramid (maxpooling kernel r = 3, 5, 11, 15, 23) in all experiments.

### Data Preparation

CARLA, KITTI, Argoverse

We downsampled the maps using voxel resolutions 0.1m, 0.2m, and 0.4m for the baseline experiments, and used the 0.2m resolution as the input of our HyperMap. The final HyperMap resolution is 0.4m. To simulate erroneous initial pose, we added translation noise within \[-2m, +2m] in xyz directions, and rotation noise of \[-10◦, +10◦] about xyz axes applied in xyz order following CMR. The initial poses were generated online in training time and fixed in test time.

### Implementation Details

We split the training process into two stages. First, we cropped the local map around initial camera pose with radius 50m and voxelized it and then applied the 3D sparse convolutional layer to the local voxelized map to extract map features. Afterwards, the extracted map feature was projected to form a virtual feature image for pose prediction and initial training. Second, after the map feature is welltrained, we applied the pretrained sparse convolutional layers to the whole voxelized map to get the map features fi, using K-means to get the centroid index di for each voxel, and only store the di in the map for the map size comparison. Afterwards, we fixed the map features, only refining the pose prediction network until convergence. The refinement step helped to compensate the compression error induced by Kmeans.&#x20;

We train all the models using learning rate $$10^{-4}$$ and batch size 40 with Adam optimizer. The occlusion handling layer takes about 1ms and the pose prediction takes about 14ms on an Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz machine with GeForce GTX TITAN Xp GPU for KITTI odometry dataset.

### Performance

The map size is computed as the total storage required to store all the 3-dimensional indice (2-byte integer coordinates) of the occupied voxels (map points) and the corresponding voxel features (4-bit for the 16 K-means centroid indices). The baseline map only contains the voxel indices and no feature.

<figure><img src="../../../.gitbook/assets/image (972).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (952).png" alt=""><figcaption></figcaption></figure>
