---
description: 'NeuMap: Neural Coordinate Mapping by Auto-Transdecoder for Camera Localization'
---

# \[arxiv 2022] NeuMap

Abstract

This paper presents an end-to-end neural mapping method for camera localization, encoding a whole scene into a grid of latent codes, with which a Transformer-based auto-decoder regresses 3D coordinates of query pixels. State-of-the-art camera localization methods require each scene to be stored as a 3D point cloud with per-point features, which takes several gigabytes of storage per scene. While compression is possible, the performance drops significantly at high compression rates. NeuMap achieves extremely high compression rates with minimal performance drop by using 1) learnable latent codes to store scene information and 2) a scene-agnostic Transformer-based auto-decoder to infer coordinates for a query pixel. The scene-agnostic network design also learns robust matching priors by training with large-scale data, and further allows us to just optimize the codes quickly for a new scene while fixing the network weights.

## Introduction

We design our method to enjoy the benefits of compact scene representation of SCR methods and the robust performance of FM methods. Similar to FM methods, we also focus on a sparse set of robustly learned features to deal with large viewpoint and illumination changes. On the other hand, we exploit similar ideas to SCR methods to regress the 3D scene coordinates of these sparse features in the query images with a compact map representation. Our method, dubbed neural coordinate mapping (NeuMap), first extracts robust features from images and then applies a transformer-based auto-decoder (i.e., auto-transdecoder) to learn: 1) a grid of scene codes encoding the scene information (including 3D scene coordinates and feature information) and 2) the mapping from query image feature points to 3D scene coordinates. At test time, given a query image, after extracting image features of its key-points, the auto-transdecoder regresses their 3D coordinates via cross attention between image features and latent codes. In our method, the robust feature extactor and the auto-transdecoder are scene-agnostic, where only latent codes are scene specific. This design enables the scene-agnostic parameters to learn matching priors across scenes while maintaining a small data size. To handle large scenes, we divide the scene into smaller sub-regions and process them independently while applying a network pruning technique to drop redundant codes.

## Method

Given a scene represented by a set of reference images $$\{I_n\}$$ with known camera calibrations $$\{T_n\}$$ expressed in a (scene-specific) coordinate frame, we devise a technique that encodes these images into a compact scene representation $$S$$. We employ this scene representation to perform visual localization, a task of predicting the camera $$T_q$$ of a query image $$I_q$$ that was never seen before – with a vantage point and/or appearance different from the one in the reference image set.

### Overview

<figure><img src="../../../.gitbook/assets/image (73).png" alt=""><figcaption></figcaption></figure>

We achieve visual localization by solving the proxy task of scene coordinate regression on sparse features, where given a set of 2D key-point $$\{k_i\}$$ extracted from $$I_q$$, we predict its corresponding 3D scene coordinate $$\{K_i\}$$. As shown in Figure 2, our method extracts 2D key-points {ki} following HLoc with a pre-trained backbone, R2D2. In order to determine their scene coordinates, we first compute a feature map of the image Iq with a trainable CNN backbone $$F$$, which is a ResNet18 and bilinearly interpolates the feature, and then solve the scene coordinate by a decoder $$D$$ as,

<figure><img src="../../../.gitbook/assets/image (62) (1).png" alt=""><figcaption></figcaption></figure>

Here, $$S$$ is the learned scene representation. Finally, we obtain camera localization by solving a perspective-n-point (PnP) problem with 2D-3D correspondences from $$\{k_i ↔ K_i\}$$.

### Sparse Scene Coordinate Regression

#### Scene Representation

Given a set of reference images, we extract 2D key-points by R2D2 and compute their corresponding 3D scene coordinates by COLMAP. Some of the key-points are not successfully triangulated, so their 3D coordinates are invalid. We cache 3D scene points in a sparse grid with voxels of uniform side length $$l$$ as shown in Figure 2. We denote the $$v$$-th voxel as $$V_v$$. We then model the reference scene as a set of latent codes $$S=\{S_v\}$$ – one per voxel – and modify Equation 1 to reflect this design choice as:

<figure><img src="../../../.gitbook/assets/image (125).png" alt=""><figcaption></figcaption></figure>

Here, the scalar parameter $$c^v_i$$ is the confidence that the keypoint $$k_i$$ is in the voxel $$V_v$$, and $$K^v_i$$ is the scene coordinate under the voxel attached coordinate frame, i.e., $$K_i = K^v_i + O_v$$ and $$O_v$$ is the mean of all the coordinates in $$V_v$$.

This formulation simultaneously solves a classification and regression problem: 1) classifying whether a 2D keypoint belongs to a 3D voxel and 2) regressing its 3D position locally within the voxel.

#### Sparity

Inspired by network pruning, we apply code pruning to remove redundant codes. Specifically, we multiply each voxel code in $$S_v$$ with a scaling factor $$w_v$$, and jointly learn the codes with these scaling factors, with L1 loss imposed on $$w_v$$. After finishing training, we prune codes whose weights are below a threshold and finetune the remaining codes.

#### Scene Coordinate Regression

We use a set of per-voxel latent codes $$S_v$$ to facilitate the learning of scene coordinate regression. The decoder $$D$$ is a stacked transformer to regress the scene coordinates of the 2D image key-points. We include $$T$$ transformer blocks, (6 blocks in our implementation), defined by the inductive relationship (see Figure 2) as

<figure><img src="../../../.gitbook/assets/image (131).png" alt=""><figcaption></figcaption></figure>

where the feature $$f^{(1)}_i =F(I_q, k_i)$$, $$w^t_v$$ is the scaling factor enforcing sparsity. Each transformer block contains a set of codes $$s^t_v$$ , and the final per voxel codes are $$S_v=\{s^t_v , 1≤t≤T\}$$. The function $$CrAtt(·, ·)$$ is classical cross-attention from transformers

<figure><img src="../../../.gitbook/assets/image (114).png" alt=""><figcaption></figcaption></figure>

At the end of the stacked transformer, we apply another MLP to compute the scene coordinate and confidence as,

<figure><img src="../../../.gitbook/assets/image (72).png" alt=""><figcaption></figcaption></figure>

### Training

<figure><img src="../../../.gitbook/assets/image (71).png" alt=""><figcaption></figcaption></figure>

The scene coordinate loss is defined as,

<figure><img src="../../../.gitbook/assets/image (129).png" alt=""><figcaption></figcaption></figure>

The second term is a classification loss, i.e., a binary cross entropy, for the confidence $$c^v_i$$ ,

<figure><img src="../../../.gitbook/assets/image (67) (1).png" alt=""><figcaption></figcaption></figure>

The third term enforces sparsity and produces a compressed representation, which is defined as,

<figure><img src="../../../.gitbook/assets/image (97).png" alt=""><figcaption></figcaption></figure>

#### Training strategy

We learn the decoder $$D$$, the CNN backbone $$F$$, and the scene representation $$S$$ with voxel sampling. At each iteration, we randomly choose $$B$$ voxels, where $$B$$ is the batch size. Each voxel $$V_v$$ has a set of reference images $$I_v$$, each of which contains at least 20 scene points in $$V_v$$. We then sample one training image for each voxel and optimize $$D, F,$$ and the scene codes of sampled voxels by minimizing the training loss in Equation 8. We sample voxels without replacement, so all scene codes are updated once at each epoch.

Similarly to network pruning, we minimize the training loss to convergence, set sparsity factors $$w^t_v$$ whose values are below a certain threshold to zero, and fine-tune our model while keeping $$w^t_v$$ frozen.

#### Scene adaption

Given a new scene (i.e., not in the training data), we simply optimize the scene code $$S$$, while keeping decoder $$D$$ and CNN backbone $$F$$ fixed. In this way, our scene representation $$S$$ is scene specific, but the decoder $$D$$ and feature extractor $$F$$ are scene agnostic.

### Inference

Given a query image $$I_q$$, we use an existing deep image retrieval method to retrieve the most similar reference images, which activate a subset of voxels; see Figure 2. A voxel $$V_v$$ is activated if one of the retrieved reference images contains at least 20 scene points in $$V_v$$. For large-scale scenes, we typically get around 100-200 voxels, while for small-scale scenes, we consider all the voxels without image retrieval. We then extract a set of 2D key-points $$\{k_i\}$$ within $$I_q$$, and for each of them, regress their per-voxel confidence and positions via Equation 2. We discard points with confidence $$c<0.5$$. All the remaining points are used to compute the camera pose with the PnP algorithm combined with RANSAC, implemented in Pycolmap.

## Experiments

### Implementation details

We use ResNet-18 to extract image features and train models with 8 V100 GPUs with a batch size of 256. We set the initial learning rate to 0.002 to train the scene-agnostic parameters and 0.0001 to train the scene-specific codes. After every 30 epochs, we multiply the learning rate by 0.5. We train the model for 200 epochs in the first stage and fine-tune the model for 100 epochs in the second stage. For training data generation, we use r2d2 to extract key-points and run triangulation to get coordinates in Cambridge, Aachen, and NAVER. We use depth images to obtain 3D coordinates in 7scenes and ScanNet.

### Main results with code-pruning

<figure><img src="../../../.gitbook/assets/image (123).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (122).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (70).png" alt=""><figcaption></figcaption></figure>

### Main results without code-pruning

<figure><img src="../../../.gitbook/assets/image (92).png" alt=""><figcaption></figcaption></figure>

### Ablation studies

<figure><img src="../../../.gitbook/assets/image (134).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (110).png" alt=""><figcaption></figcaption></figure>
