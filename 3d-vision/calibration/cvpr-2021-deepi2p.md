---
description: 'DeepI2P: Image-to-Point Cloud Registration via Deep Classification'
---

# \[CVPR 2021] DeepI2P

{% embed url="https://github.com/lijx10/DeepI2P" %}

## Abstract

This paper presents DeepI2P: a novel approach for cross-modality registration between an image and a point cloud. Given an image (e.g. from a rgb-camera) and a general point cloud (e.g. from a 3D Lidar scanner) captured at different locations in the same scene, our method estimates the relative rigid transformation between the coordinate frames of the camera and Lidar. Learning common feature descriptors to establish correspondences for the registration is inherently challenging due to the lack of appearance and geometric correlations across the two modalities. We circumvent the difficulty by converting the registration problem into a classification and inverse camera projection optimization problem. A classification neural network is designed to label whether the projection of each point in the point cloud is within or beyond the camera frustum. These labeled points are subsequently passed into a novel inverse camera projection solver to estimate the relative pose.

## Introduction

Image-to-point cloud registration refers to the process of finding the rigid transformation, i.e., rotation and translation that aligns the projections of the 3D point cloud to the image. This process is equivalent to finding the pose, i.e., extrinsic parameters of the imaging device with respect to the reference frame of the 3D point cloud.

Cross-modality image-to-point cloud registration can be used to alleviate the aforementioned problems from the same modality registration methods. Specifically, a 3D point cloud-based map can be acquired once with Lidars, and then pose estimation can be deployed with images taken from cameras that are relatively low-maintenance and less costly on a large fleet of robots and mobile devices. Moreover, maps acquired directly with Lidars circumvents the hassle of SfM, and are largely invariant to seasonal/illumination changes.

<figure><img src="../../.gitbook/assets/image (774).png" alt=""><figcaption></figcaption></figure>

In this paper, we propose the DeepI2P: a novel approach for cross-modal registration of an image and a point cloud without explicit feature descriptors as illustrated in Fig. 1. Our method requires lesser storage memory, i.e., O(3N) for the reference point cloud since we do not rely on feature descriptors to establish correspondences. Furthermore, the images captured by cameras can be directly utilized without SfM. We solve the cross-modal image-to-point cloud registration problem in two stages. In the first stage, we design a two-branch neural network that takes the image and point cloud as inputs, and outputs a label for every point that indicates whether the projection of this point is within or beyond the image frustum. The second stage is formulated as an unconstrained continuous optimization problem. The objective is to find the optimal camera pose, i.e., the rigid transformation with respect to the reference frame of the point cloud, such that 3D points labeled as within the camera frustum is correctly projected into the image. Standard solvers such as the Gauss-Newton algorithm can be used to solve our camera pose optimization problem.

The main contributions of this paper are listed as follow:

* We circumvent the challenging need to learn cross-modal feature descriptor for registration by casting the problem into a two-stage classification and optimization framework.
* A two-branch neural network with attention modules to enhance cross-modality fusion is designed to learn labels of whether a 3D point is within or beyond the camera frustum.
* The inverse camera projection optimization is proposed to solve for the camera pose with the classification labels of the 3D points.
* Our method and the experimental results show a proof-of-concept that cross-modal registration can be achieved with deep classification.

## Overview of DeepI2P

We denote an image as $$I \in R^{3\times W \times H}$$ , where W and H are the image width and height, and a point cloud as $$P =\{P_1, P_2, ... , P_N | P_n \in R^3\}$$. The cross-modal image-topoint cloud registration problem is to solve for the rotation matrix $$R \in SO(3)$$ and translation vector $$t \in R^3$$ between the coordinate frames of the camera and point cloud. Establishing crossmodal point-to-pixel correspondence is non-trivial. This is because the points in the $$R^3$$ space shares very little appearance and geometric correlations with the image in the $$P^2$$ space.

To this end, we propose a two-stage “Frustum classification + Inverse camera projection” pipeline. The first stage classifies each point in the point cloud into within or beyond the camera frustum. We call this the frustum classification, which is done easily by a deep network. In the second stage, we show that it is sufficient to solve the pose between camera and point cloud using only the frustum classification result. This is the inverse camera projection problem. In our supplementary materials, we propose another cross-modality registration method “Grid classification + PnP” as our baseline for experimental comparison. In the grid classification, the image is divided into a tessellation of smaller regular grids, and we predict the cell each 3D point projects into. The pose estimation problem can then be solved by applying RANSAC-based PnP to the grid classification output.

## Classification

The input to the network is a pair of image $$I$$ and point cloud $$P$$ , and the output is a per-point classification for $$P$$ . There are two classification branches: frustum and grid classification. The frustum classification assign a label to each point, $$L_c = \{l^c_1, l^c_2, ..., l^c_N \}$$, where $$l^c_n \in \{0, 1\}$$. $$l^c_n = 0$$ if the point $$P_n$$ is projected to outside the image $$I$$, and vice versa. Refer to the supplementary for the details of the grid classification branch used in our baseline.

### Our Network Design

<figure><img src="../../.gitbook/assets/image (789).png" alt=""><figcaption></figcaption></figure>

Our per-point classification network consists of four parts: point cloud encoder, point cloud decoder, image encoder and image-point cloud attention fusion. The point cloud encoder/decoder follows the design of SO-Net and PointNet++, while the image encoder is a ResNet-34. The classified points are then used in our inverse camera projection optimization to solve for the unknown camera pose.

#### Point Cloud Encoder

Given an input point cloud denoted as $$P \in R^{3\times N}$$ , a set of nodes $$\mathfrak{P}^{(1)} \in R^{3\times M_1}$$ is sampled by Farthest Point Sampling (FPS). A point-to-node grouping is performed to obtain $$M_1$$ clusters of points. Each cluster is processed by a PointNet to get $$M_1$$ feature vectors of length $$C_1$$, respectively, i.e. $$P^{(1)}\in R^{C_1\times M_1}$$ . The point-to-node grouping is adaptive to the density of points. This is beneficial especially for point clouds from Lidar scans, where points are sparse at far range and dense at near range. The above sampling-grouping-PointNet operation is performed again to obtain another set of feature vectors $$P^{(2)}\in R^{C_2\times M_2}$$ . Finally, a PointNet is applied to obtain the global point cloud feature vector $$P^{(3)} \times  R^{C_3\times 1}$$.

#### Image-Point Cloud Attention Fusion

The goal of the classification is to determine whether a point projects to the image plane (frustum classification) and which region it falls into (grid classification). Hence, it is intuitive that the classification requires fusion of information from both modalities. To this end, we design an Attention Fusion module to combine the image and point cloud information. The input to the Attention Fusion module consists of three parts: a set of node features $$P_{att}(P^{(1)} or P^{(2)})$$, a set of image features $$I_{att} \in R^{C_{img}\times H_{att} \times W_{att}} (I^{(1)} or I^{(2)})$$, and the global image feature vector $$I^{(3)}$$. As shown in Fig. 2, the image global feature is stacked and concatenated with the node features $$P_{att}$$, and fed into a shared MLP to get the attention score $$S_{att} \in R^{H_{att}W_{att}\times M}$$ . $$S_{att}$$ provides a weighting of the image features $$I_{att}$$ for $$M$$ nodes. The weighted image features are obtained by multiplying $$I_{att}$$ and $$S_{att}$$. The weighted image features can now be concatenated with the node features in the point cloud decoder.

#### Point Cloud Decoder

The decoder takes the image and point cloud features as inputs, and outputs the per-point classification result. In general, it follows the interpolation idea of PointNet++. At the beginning of the decoder, the global image feature $$I^{(3)}$$ and global point cloud feature $$P^{(3)}$$ are stacked $$M_2$$ times, so that they can be concatenated with the node features $$P^{(2)}$$ and the Attention Fusion output  $$\tilde{I}^{(2)}$$. The concatenated $$[I^{(3)}, \tilde{I}^{(2)}, P^{(3)}, P^{(2)}]$$ is processed by a shared MLP to get $$M_2$$ feature vectors denoted as $$\tilde{P}^{(2)} \in R^{C_2\times M_2}$$ . We perform interpolation to get $$\tilde{P}^{(2)}_{(itp)}\in R^{C_2\times M_1}$$ , where the $$M_2$$ features are upsampled to $$M_1 \ge M_2$$ features. Note that $$P^{(2)}, \tilde{P}^{(2)}$$ are associated with node coordinates $$\mathfrak{P}^{(2)}\in R^{3\times M_2}$$ . The interpolation is based on k-nearest neighbors between node coordinates $$\mathfrak{P}^{(1)} \in R^{3\times M_1}$$ , where $$M_1 \ge M_2$$. For each $$C_2$$ channel, the interpolation is denoted as:

<figure><img src="../../.gitbook/assets/image (111).png" alt=""><figcaption></figcaption></figure>

and $$\mathfrak{P}^{(2)}_j$$ is one of the k-nearest neighbors of $$\mathfrak{P}^{(1)}_i$$ in $$\mathfrak{P}^{(2)}$$. We get $$\tilde{P}^{(2)}_{(itp)} \in R^{C_2\times M_1}$$ with the concatenate-sharedMLPinterpolation process. Similarly, we obtain $$\tilde{P}^{(1)}_{(itp)} \in R^{C_1\times N}$$ after another round of operations. Lastly, we obtain the final output $$(2 + HW/(32 \times 32)) \times N$$ , which can be reorganized into the frustum prediction scores $$2 \times N$$ and grid prediction scores $$(HW/(32 \times 32)) \times N$$ .

### Training Pipeline

The generation of the frustum labels is simply a camera projection problem. During training, we are given the camera intrinsic matrix $$K \in R^{3\times 3}$$ and the pose $$G \in SE(4)$$ between the camera and point cloud. The 3D transformation of a point $$P_i \in R^3$$ from the point cloud coordinate frame to the camera coordinate frame is given by:

<figure><img src="../../.gitbook/assets/image (99).png" alt=""><figcaption></figcaption></figure>

and the point $$\tilde{P}^′_i$$ is projected into the image coordinate:

<figure><img src="../../.gitbook/assets/image (133).png" alt=""><figcaption></figcaption></figure>

Note that homogeneous coordinate is represented by a tilde symbol, e.g., $$\tilde{P}'_i$$ is the homogeneous representation of $$P'_i$$. The inhomogeneous coordinate of the image point is:

<figure><img src="../../.gitbook/assets/image (75).png" alt=""><figcaption></figcaption></figure>

#### Frustum Classification

For a given camera pose G, we define the function:

<figure><img src="../../.gitbook/assets/image (102).png" alt=""><figcaption></figcaption></figure>

which assigns a label of 1 to a point $$P_i$$ that projects within the image, and 0 otherwise. Now the frustum classification labels are generated as $$l^c_i = f(P_i; G, K, H, W )$$, where G is known during training. In the Oxford Robotcar and KITTI datasets, we randomly select a pair of image and raw point cloud $$(I, P_{raw})$$, and compute the relative pose from the GPS/INS readings as the ground truth pose Gcp. We use $$(I, P_{raw})$$ with a relative distance within a specified interval in our training data. However, we observe that the rotations in $$G^c_p$$ are close to zero from the two datasets since the cars used to collect the data are mostly undergoing pure translations. To avoid overfitting to such scenario, we apply randomly generated rotations $$G_r$$ onto the raw point cloud to get the final point cloud $$P = G_r P_{raw}$$ in the training data. Furthermore, the ground truth pose is now given by $$G = G^c_p G^{−1}_r$$.

#### Training Procedure

The frustum classification training procedure is summarized as:

1. Select a pair of image and point cloud $$(I, P_{raw})$$ with relative pose $$G^p_c$$ .
2. Generate 3D random transformation $$G_r$$, and apply it to get $$P = G_r P_{raw}$$ and $$G = G^c_p G^{−1}_r$$.
3. Get the ground truth per-point frustum labels $$l^c_i$$ according to Eq. 5.
4. Feed $$(I, P)$$ into the network illustrated in Fig. 2.
5. Frustum prediction $$\hat{L}_c = \{\hat{l}^c_1, ..., \hat{l}^c_n\}, l^c_i \in \{0, 1\}$$.
6. Apply cross entropy loss for the classification tasks to train the network.

## Pose Optimization

Formally, the pose optimization problem is to solve for $$\hat{G}$$, given the point cloud P, frustum predictions $$\hat{L}^c = \{l^c_1, ..., l^c_N\}, l^c_i \in \{0, 1\}$$, and camera intrinsic matrix K. In this section, we describe our inverse camera projection solver to solve for \$$\hat{G}.

### Inverse Camera Projection

The inverse camera projection problem is the other way around, i.e, determine the optimal pose $$\hat{G}$$ that satisfies a given $$\hat{L}^c$$. It can be written more formally as:

<figure><img src="../../.gitbook/assets/image (86).png" alt=""><figcaption></figcaption></figure>

Intuitively, we seek to find the optimal pose $$\hat{G}$$ such that all 3D points with label $$l^c_i = 1$$ from the network are projected into the image, and vice versa. However, a naive search of the optimal pose in the SE(3) space is intractable. To mitigate this problem, we relax the cost as a function of the distance from the projection of a point to the image boundary, i.e., a $$H \times W$$ rectangle.

#### Frustum Prediction Equals to 1

Let us consider a point $$P_i$$ with the prediction $$l^c_i = 1$$. We define cost function:

<figure><img src="../../.gitbook/assets/image (78).png" alt=""><figcaption></figcaption></figure>

that penalizes a pose G which causes $$p^′_{x_i}$$ of the projected point $$p^′_i = [p^′_{x_i} , p^′_{y_i} ]^T$$ (c.f. Eq. 4) to fall outside the borders of the image width. Specifically, the cost is zero when $$p^′_{x_i}$$ is within the image width, and negatively proportional to the distance to the closest border along the image x-axis otherwise. A cost $$g(p^′_{y_i}; H)$$ can be analogously defined along image y-axis. In addition, cost function $$h(·)$$ is defined to avoid the ambiguity of $$P^′_i$$ falling behind the camera:

<figure><img src="../../.gitbook/assets/image (94).png" alt=""><figcaption></figcaption></figure>

where $$\alpha$$ is a hyper-parameter that balances the weighting between $$g(·)$$ and $$h(·)$$.

#### Frustum Prediction Equals to 0

We now consider a point $$P_i$$ with prediction $$l^c_i = 0$$. The cost defined along the image x-axis is given by:

<figure><img src="../../.gitbook/assets/image (74).png" alt=""><figcaption></figcaption></figure>

Similarly, an analogous cost $$u(p^′_{y_i} ; H)$$ along the y-axis can be defined. Furthermore, an indicator function:

<figure><img src="../../.gitbook/assets/image (107).png" alt=""><figcaption></figcaption></figure>

is required to achieve the target of zero cost when $$p^′_i$$ is outside the $$H \times W$$ image or $$P^′_i$$ is behind the camera (i.e. $$z^′_i < 0$$).

#### Cost Function

Finally, the cost function for a single point $$P_i$$ is given by:

<figure><img src="../../.gitbook/assets/image (64).png" alt=""><figcaption></figcaption></figure>

$$p^′_{x_i} , p^′_{y_i} , z_i$$ are functions of $$G$$ according to Eq. 2, 3 and 4. Image height $$H$$, width $$W$$ and camera intrinsics $$K$$ are known. Now the optimization problem in Eq. 6 becomes:

<figure><img src="../../.gitbook/assets/image (82).png" alt=""><figcaption></figcaption></figure>

$$G \in SE(3)$$ is an over-parameterization that can cause problems in the unconstrained continuous optimization. To this end, we use the Lie-algebra representation $$\xi \in se(3)$$ for the minimal parameterization of $$G \in SE(3)$$. The exponential map $$G = exp_{se(3)}(\xi)$$ converts $$se(3) \rightarrow SE(3)$$, while the log map $$\xi = log_{SE(3)}(G)$$ converts $$SE(3) \rightarrow se(3)$$. We define the se(3) concatenation operator $$\circ : se(3) × se(3) \rightarrow se(3)$$ as:

<figure><img src="../../.gitbook/assets/image (69).png" alt=""><figcaption></figcaption></figure>

and the cost function in Eq. 12 can be re-written with the proper exponential or log map modifications into:

<figure><img src="../../.gitbook/assets/image (106).png" alt=""><figcaption></figcaption></figure>

#### Gauss-Newton Optimization

Eq. 15 is a typical least squares optimization problem that can be solved by the Gauss-Newton method. During iteration i with the current solution $$\xi^{(i)}$$, the increment $$\delta \xi^{(i)}$$ is estimated by a Gauss-Newton second-order approximation:

<figure><img src="../../.gitbook/assets/image (68).png" alt=""><figcaption></figcaption></figure>

and the update is given by $$\xi^{(i+1)} = \delta \xi^{(i)} \circ \xi^{(i)}$$. Finally the inverse camera projection problem is solved by performing the exponential map $$\hat{G}=exp_{se(3)}(\hat{\xi})$$.

## Experiments

#### Inverse Camera Projection

The initial guess G(0) in our proposed inverse camera projection (c.f. Section 5.1) is critical since the solver for Eq. 15 is an iterative approach. To alleviate the initialization problem, we perform the optimization 60 times with randomly generated initialization G(0), and select the solution with the lowest cost. In addition, the 6DoF search space is too large for random initialization. We mitigate this problem by leveraging on the fact that our datasets are from ground vehicles to perform random initialization in 2D instead. Specifically, R(0) is initialized as a random rotation around the up-axis, and t(0) as a random translation in the x-y horizontal plane. Our algorithm is implemented with Ceres.

<figure><img src="../../.gitbook/assets/image (124).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (112).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (80).png" alt=""><figcaption></figcaption></figure>
