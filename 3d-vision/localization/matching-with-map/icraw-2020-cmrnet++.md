---
description: 'CMRNet++: Map and Camera Agnostic Monocular Visual Localization in LiDAR Maps'
---

# \[ICRAW 2020] CMRNet++

## Abstract

Localization is a critically essential and crucial enabler of autonomous robots. While deep learning has made significant strides in many computer vision tasks, it is still yet to make a sizeable impact on improving capabilities of metric visual localization. One of the major hindrances has been the inability of existing Convolutional Neural Network (CNN)based pose regression methods to generalize to previously unseen places. Our recently introduced CMRNet effectively addresses this limitation by enabling map independent monocular localization in LiDAR-maps. In this paper, we now take it a step further by introducing CMRNet++, which is a significantly more robust model that not only generalizes to new places effectively, but is also independent of the camera parameters. We enable this capability by combining deep learning with geometric techniques, and by moving the metric reasoning outside the learning process. In this way, the weights of the network are not tied to a specific camera.

## Introduction

In this paper, we present our CMRNet++ approach for camera to LiDAR-map registration. We build upon our previously proposed CMRNet model, which was inspired by camerato-LiDAR pose calibration techniques. CMRNet localizes independent of the map, and we now further improve it to also be independent of the camera intrinsics. Unlike existing stateof-the-art CNN-based approaches for pose regression, CMRNet does not learn the map, instead it learns to match images to a pre-existing map. Consequently, CMRNet can be used in any environment for which a LiDAR-map is available. However, since the output of CMRNet is metric (a 6-DoF rigid body transformation from an initial pose), the weights of the network are tied to the intrinsic parameters of the camera used for collecting the training data. In this work, we mitigate this problem by decoupling the localization, by first employing a pixel to 3D point matching step, followed by a pose regression step. While the network is independent of the intrinsic parameters of the camera, they are still required for the second step.

## Technical Approach

We extend CMRNet by decoupling the localization into two steps: pixel to 3D point matching, followed by pose regression. In the first step, the CNN only focuses on matching at the pixel-level instead of metric basis, which makes the network independent of the intrinsic parameters of the camera. These parameters are instead employed in the second step, where traditional computer vision methods are exploited to estimate the pose of the camera, given the matches from the first step.

<figure><img src="../../../.gitbook/assets/image (763).png" alt=""><figcaption></figcaption></figure>

### Matching Step

For the matching step, we generate a synthesized depth image, which we refer to as a LiDAR-image, by projecting the map into a virtual image plane placed at Hinit , a rough pose estimate obtained, e.g., from a GNSS. The projection uses the camera intrinsics. To deal with occlusions in point clouds, we employ a z-buffer technique followed by an occlusion estimation filter. Once the inputs to the network (camera and LiDAR images) have been obtained, for every 3D point in the LiDAR-image, CMRNet++ estimates the pixel of the RGB image that represents the same world point.

The architecture of CMRNet++ is based on PWC-Net. The output of CMRNet++ is a dense feature map, which is 1/4-th the input resolution and consists of two channels that represent, for every pixel in the LiDAR-image, the displacement (u and v) of the pixel in the RGB image from the same world point.

In order to train CMRNet++, we first need to generate the ground truth pixel displacement $$\triangle P$$ of the LiDAR-image w.r.t. RGB image. To accomplish this, we first compute the coordinates of the map’s points in the $$H_{init}$$ reference frame using Equation (1) and the pixel position of their projection in the LiDAR-image exploiting the intrinsic matrix $$K$$ of the camera from Equation (2).

<figure><img src="../../../.gitbook/assets/image (712).png" alt=""><figcaption></figcaption></figure>

We keep track of indices of valid points in an array $$VI$$. This is done by excluding indices of points whose projection lies behind or outside the image plane, as well as points marked occluded by the occlusion estimation filter. Subsequently, we generate the sparse LiDAR-image $$D$$ and project the points of the map into a virtual image plane placed at the ground truth pose $$H_{GT}$$ . We then store the pixels’ position of these projections as

<figure><img src="../../../.gitbook/assets/image (744).png" alt=""><figcaption></figcaption></figure>

Finally, we compute the displacement ground truths $$\triangle P$$ by comparing the projections in the two image planes as

<figure><img src="../../../.gitbook/assets/image (717).png" alt=""><figcaption></figcaption></figure>

For every pixel $$[u, v]$$ without an associated 3D point, we set $$D_{u,v} = 0$$ and $$\triangle P_{u,v} = [0, 0]$$. Moreover, we generate a mask of valid pixels as $$mask_{u,v} = 1~\mbox{if}~ D_{u,v} > 0, 0$$ otherwise. We use a loss function that minimizes the sum of the regression component $$L_{reg}$$ and the smoothness component $$L_{smooth}$$, to train our network. The regression loss defined in Equation (6) penalizes pixel displacements predicted by the network $$\hat{\triangle P}$$ that differs from their respective ground truth displacements $$\triangle P$$ on valid pixels. While the smoothness loss $$L_{smooth}$$ enforces the displacement of pixels without a ground truth to be similar to the ones in the neighboring pixels.

<figure><img src="../../../.gitbook/assets/image (724).png" alt=""><figcaption></figcaption></figure>

where $$\rho$$ is the generalized Charbonnier function $$\rho(x) = (x^2 + \epsilon^2)^α , \epsilon = 10^{−9}, \alpha = 0.25$$.

### Localization Step

Once CMRNet++ has been trained, we have the map, i.e., a set of 3D points $$P$$ whose coordinates are known, altogether with their projection in the LiDAR-image $$D$$ and a set $$p$$ of matching points in the RGB image that is predicted by the CNN given as

<figure><img src="../../../.gitbook/assets/image (776).png" alt=""><figcaption></figcaption></figure>

Estimating the pose of the camera given a set of 2D-3D correspondences and the camera intrinsics is known as the Perspective-n-Points (PnP) problem. We use the EPnP algorithm within a RANdom SAmple Consensus (RANSAC) scheme to solve this, with a maximum of 1000 iterations and an inlier threshold value of 2 pixels.

### Iterative Refinement

Similar to CMRNet, we employ an iterative refinement technique where we train different instances of CMRNet++, each specialized in handling different initial error ranges. During inference, we feed the RGB and LiDAR-image to the network trained with the highest error range, and we generate a new LiDAR-image by projecting the map in the predicted pose. The latter is then fed to the second instance of CMRNet++ that is trained with a lower error range. This process can be repeated multiple times, iteratively improving the estimated localization accuracy.

### Training Details

We train each instance of CMRNet++ from scratch for 300 epochs with a batch size of 40 using two NVIDIA Tesla P100. The weights of the network were updated with the ADAM optimizer with an initial learning rate of $$1.5 \times 10^{−4}$$ and a weight decay of $$5 \times 10^{−6}$$. We halved the learning rate after 20 and 40 epochs.

## Experimental Results

<figure><img src="../../../.gitbook/assets/image (706).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (796).png" alt=""><figcaption></figcaption></figure>
