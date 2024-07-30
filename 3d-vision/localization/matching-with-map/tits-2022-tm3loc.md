---
description: >-
  TM3Loc: Tightly-Coupled Monocular Map Matching for High Precision Vehicle
  Localization
---

# \[TITS 2022] TM3Loc

## Abstract

Vision-based map-matching with HD map for high precision vehicle localization has gained great attention for its low-cost and ease of deployment. However, its localization performance is still unsatisfactory in accuracy and robustness in numerous real applications due to the sparsity and noise of the perceived HD map landmarks. This article proposes the tightly-coupled monocular map-matching localization algorithm (TM3Loc) for monocular-based vehicle localization. TM3Loc introduces semantic chamfer matching (SCM) to model monocular map-matching problem and combines visual features with SCM in a tightly-coupled manner. By applying the sliding window-based optimization technique, the historical visual features and HD map constraints are also introduced, such that the vehicle poses are estimated with an abundance of visual features and multi-frame HD map landmark features, rather than with single-frame HD map observations in previous works. Experiments are conducted on large scale dataset of 15 km long in total. The results show that TM3Loc is able to achieve high precision localization performance using a low-cost monocular camera, largely exceeding the performance of the previous state-of-the-art methods, thereby promoting the development of autonomous driving.

## Introduction

Existing high-precision localization techniques based on differential RTK (D-RTK), such as GNSS, can theoretically achieve centimeter-level localization precision. However, in real applications such as urban scenarios, a large localization deviation is often seen when the surrounding buildings and trees block the GNSS signal, making using the GNSS sensor alone becomes insufficient \[4,5].

> \[4] A. Kumar, T. Oishi, S. Ono, A. Banno, and K. Ikeuchi, “Global coordinate adjustment of 3D survey models in world geodetic system under unstable GPS condition,” in Proc. 20th ITS World Congr. Tokyo, Jan. 2013, p. 10.&#x20;
>
> \[5] A. Aggarwal, “Machine vision based self-position estimation of mobile robots,” Int. J. Electron. Commun. Eng. Technol., vol. 6, no. 10, pp. 20–29, 2015.

In recent years, since its first inception in late 2010, High-Definition Maps (HD Maps) has gained tremendous popularity in the intelligent vehicle industry, mainly because it carries road elements with a much higher level of details compared to the traditional navigation maps \[6]. Several map building companies have engaged in constructing their HD map databases on a large scale with a series of production and publication standards, such as Navigation Data Standard (NDS) and Local Dynamic Map (LDM). The mainstream approach to building HD Maps is through Mobile Mapping System (MMS) equipped with high precision sensors including LiDAR, RTK and IMU at centimeter-level precision. The resultant map then consists of fine localization features that can support intelligent vehicles’ positioning and trajectory planning. The localization feature in the HD map can be divided into a) dense point cloud feature and b) sparse landmark feature. The point cloud feature consists of the original point cloud scanned by 3D LiDAR sensor \[7], which maintains the raw geometric information of the point cloud. State-of-the-art map-based localization methods use point cloud HD map to accurately estimate vehicle pose within a maximum error of 0.2 m \[8]–\[10]. However, equipping intelligent vehicles (IVs) with LiDAR sensors will significantly and undesirably increase the overall sensor cost and the subsequent vehicle production cost. Furthermore, the huge data size of the point cloud map increases the difficulty in implementing the HD map on IVs. Compared to the landmarks in the point cloud map, the lightweight HD map landmark features are more flexible and easy to use. The HD map landmarks consist of static vectorized semantic landmarks (such as lane lines, poles, and traffic signs), which are much more lightweight than those in the original point cloud map. Matching the HD map landmark features with the images from the low-cost camera is an engineering- and commercial-friendly solution for mass-produced vehicles. As a result, researchers have been investing great efforts in matching these HD map landmarks with features in the camera images. The basic idea behind this approach is to detect semantic landmarks of HD map in the camera image. The vehicle pose can then be obtained by aligning the detected landmarks in the image with their corresponding 3D landmarks in the HD map.

> \[7] K. Yoneda, H. Tehrani, T. Ogawa, N. Hukuyama, and S. Mita, “LiDAR scan feature for localization with highly precise 3-D map,” in Proc. IEEE Intell. Vehicles Symp., Jun. 2014, pp. 1345–1350.&#x20;
>
> \[8] G. Wan et al., “Robust and precise vehicle localization based on multi- sensor fusion in diverse city scenes,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2018, pp. 4670–4677.
>
> \[9] X. Zuo, P. Geneva, Y. Yang, W. Ye, Y. Liu, and G. Huang, “Visual- inertial localization with prior LiDAR map constraints,” IEEE Robot. Autom. Lett., vol. 4, no. 4, pp. 3394–3401, Oct. 2019.&#x20;
>
> \[10] W. Ding, S. Hou, H. Gao, G. Wan, and S. Song, “LiDAR inertial odometry aided robust LiDAR localization system in changing city scenes,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2020, pp. 4322–4328.

We propose TM3Loc, a novel localization algorithm for improving the performance of the monocular map-matching process. First, to deal with the map-matching problem in the image plane, we adopt the semantic chamfer matching (SCM) algorithm, inspired by the traditional image alignment method, chamfer matching. In TM3Loc, we used SCM as a general cost model to match different landmarks. However, instead of optimizing the cost model using global search or automatic derivatives, we derived an analytical derivation of chamfer matching cost with respect to the 6-DoF pose on se(3) to ensure efficient optimization. Besides, to tackle the inevitable noise of HD map landmark feature detection, an effective outlier rejection strategy is also proposed. With the improved SCM implementation, the monocular map-matching problem can be efficiently and robustly solved in a unified form with various landmark shapes, thereby avoiding the inaccuracy brought by any prior assumption of the target shapes.

Secondly, to deal with the sparsity of HD map landmark features, we introduce the visual features from monocular images in the map-matching process. we introduce a tightly-coupled sliding window-based optimization strategy to fuse the map-matching and visual feature localization con- straints. Similar idea is also utilized in our previous work \[26]. The basic concept is to optimize the poses with the minimiza- tion target consisting of both visual landmark feature residuals and HD map landmark feature residuals in a certain length of past frames. As a result, the localization estimator can provide the global pose of the current frame even when the current HD map observations are insufficient. Moreover, with the aid of the more abundant and accurate constraints of visual feature landmarks, the system can be more robust against the HD map observation noise and offer a more accurate localization result. However, the calculation of SCM for multi-frames in the tightly-coupled sliding window-based optimization is generally time-consuming, making it challenging to meet the real-time performance. As a further improvement of tightly-coupled strategy in \[26], a linearization approximation algorithm that simplifies the SCM residual is also proposed to accelerate the overall optimization process. With this algorithm, the tightly-coupled sliding window-based optimization can be solved in real-time with a negligible performance drop.

> \[26] T. Wen, Z. Xiao, B. Wijaya, K. Jiang, M. Yang, and D. Yang, “High pre- cision vehicle localization based on tightly-coupled visual odometry and vector HD map,” in Proc. IEEE Intell. Vehicles Symp. (IV), Oct. 2020, pp. 672–679.

The contributions of this article are summarized as follows:

1. We adopted the semantic chamfer matching (SCM) as the monocular map matching model and derived its analytical derivatives with respect to the 6-DoF camera pose. Also, an outlier association rejection strategy is proposed, thereby allowing both efficient and robust map-matching optimization.
2. tightly-coupled sliding window-based optimization algorithm to fuse visual feature and HD map land- mark features is proposed. Moreover, a linearization approximation algorithm is proposed to accelerate the calculation of SCM during the optimization to ensure the real-time performance of the whole system.
3. large scale dataset with HD map landmarks on KAIST Urban Dataset and Shougang Park is built for evaluating the map matching localization algorithms. Experiments are conducted on the proposed dataset, and the results have demonstrated the robustness and high precision of our self-localization algorithm.

## Method

### Problem Formulation

We start by considering the on-board camera pose as the equivalent vehicle pose for this study, as the camera is fixed to the vehicle. As such, the 6-DoF pose of camera frame $$C_t$$ at time t in global frame $$G$$ is defined as $$^Gx_{C_t} = \{ ^Gt_{C_t},{}^GR_{C_t} \}$$, where $${}^Gt_{C_t} \in R^3$$ represents the translation vector from the origin of $$G$$ to the origin of $$C_t$$ , and $${}^GR_{C_t} \in SO(3)$$ represents the $$3 \times 3$$ rotation matrix from frame $$G$$ to frame $$C_t$$ . The problem of vehicle localization can be defined as the camera pose $${}^Gx_{C_t}$$ estimation relative to the HD map, given the monocular image with HD map landmark observations up to time $$t$$.

### HD Map Landmark

This study utilizes the lane boundaries and poles for localization because these elements provide the basic localization cues and are commonplace in structured road scenarios. The HD map is denoted as $$M =\{M_i\}$$. Each landmark $$M_i$$ with its semantic category si is modeled as a series of control 3D points $$\{m_{i,j} \in R^3\}_{j=1:N_i}$$ sampled uniformly in the 3D space for a unified representation, with $$N_i$$ as the total number of control points of landmark $$M_i$$ .

In this study, we design a network based on FCN to detect lane boundaries and poles. Thus, this study adopts the semantic segmentation of lane lines and poles as the observation since it provides a straightforward approach to precisely describe the diverse shapes of HD map landmarks.

### Semantic Chamfer Matching

We start by formulating the map matching problem in a single frame HD map observation. Given the initial camera pose $${}^Gx_{C_t}$$ , the 3D vector HD map landmarks points $$m_{i,j} \in M_i$$ can be projected into the image space. The map-matching problem is to find an optimal camera pose $${}^Gx^{∗}_{C_t}$$ which can minimize the cost model $$d$$ between the projected HD map landmark points and their corresponding observations, as formulated in (1):

<figure><img src="../../../.gitbook/assets/image (63).png" alt=""><figcaption></figcaption></figure>

where $$z_{m_{i,j}}$$ is the observation of landmark point $$m_{i, j}$$ in the image $$I_t$$ ,and $$m^{I_t}_{i, j}$$ represent the 2D projection result of $$m_{i, j}$$ given the camera pose $${}^Gx_{C_t}$$ , i.e.:

<figure><img src="../../../.gitbook/assets/image (67).png" alt=""><figcaption></figcaption></figure>

Suppose the images have been undistorted beforehand, we adopt the pinhole model $$\pi(\cdot): R^3 \rightarrow R^2$$ as the camera model:

<figure><img src="../../../.gitbook/assets/image (48) (1).png" alt=""><figcaption></figcaption></figure>

where $$K$$ is the intrinsic matrix.

Given the segmentation result $$S_t$$ of frame $$t$$, the cost model of HD map control points $$m_{i, j}$$ with the detection result is formulated as:

<figure><img src="../../../.gitbook/assets/image (65).png" alt=""><figcaption></figcaption></figure>

The SCM algorithm first generates the distance images $$\{D_{t_s}, s\in S\}$$ for each kind of HD map landmarks from the image segmentation result $$S_t$$ , with $$S$$ as the set of semantic categories of all kinds of HD map landmark features. In this work, $$S =\{LaneBoundary, Pole\}$$. The distance image is formed by augmenting each pixel with its distance to the nearest non-zero pixel. Thus it is essentially a lookup table for querying the nearest distance for all pixels in the image space. Given a set of projected HD map control points $$m^{I_t}_{i, j} = (u,v)$$ with type $$s_i$$ , the nearest distance can be approximated by bi-linear interpolation in $$D^{s_i}_t$$ as shown in (5):

<figure><img src="../../../.gitbook/assets/image (62).png" alt=""><figcaption></figcaption></figure>

where  $$\overline{u} = \lfloor u \rfloor$$and $$\delta u = u − \overline{u}$$ and the same for $$\overline{v}$$ and $$\delta v$$. $$D^{s_i}_t(\cdot, \cdot)$$ represents the corresponding pixel value in $$D^{s_i}_t$$.

In our implementation, we first separate the segmentation result $$S_t$$ into $$\{S_t^s\}$$ with different semantic categories, and the distance images $$\{D_t^s\}$$ are subsequently computed by the efficient two-pass algorithm \[Parametric correspondence and chamfer matching: Two New techniques for image matching] using the L2 norm with the input of $$\{S_t^s\}$$ respectively.

To tackle the noisy image perception,  a gating operation is applied as the outlier rejection strategy to the distance image $$D_t^s$$, i.e.:

<figure><img src="../../../.gitbook/assets/image (60).png" alt=""><figcaption></figcaption></figure>

where $$T$$ is the gating threshold. In our implementation, $$T$$ is set as 20. By using this strategy, the value of areas with distance larger than $$T$$ will be constant and has zero gradient. As a result, the projected HD map points lying within these areas will not be counted in the optimization.

### Tightly-coupled Map-Matching

Since visual feature landmarks only provide constraints in the local coordinate, the full state vector is composed of $$K$$ camera pose states $${}^{C_0}x_C = \{{}^{C_0}x_{C_k} \}_{k=1:K}$$ in the first camera frame $$C_0$$. As the final output of camera pose is in the global frame $$G$$, a global to local 6-DoF transformation state $${}^Gx_{C_0}$$ is also defined in $$X$$ . The states of visual feature landmarks appearing in the sliding window are also included in the inverse depth form. $$\lambda_i$$ is the inverse depth of visual feature landmark $$c_i$$ related to the frame with its first observation. It is important to note that the states of HD map landmarks are not estimated since their prior information is sufficiently accurate. The full state vector $$X$$ is defined as:

<figure><img src="../../../.gitbook/assets/image (13) (2).png" alt=""><figcaption></figcaption></figure>

The system is to find the optimal state vector $$X$$ by minimizes the Mahalanobis distance of all measurement residuals $$r(X)$$ in the sliding window:

<figure><img src="../../../.gitbook/assets/image (11) (2).png" alt=""><figcaption></figcaption></figure>

where $$r_C(z^k_{c_i}, X)$$ is the visual landmark residual and $$r_M(z^k_{m_i},X)$$ is the HD map landmark residual. $$C$$ and $$M′$$ are the sets of 3D visual landmarks and HD map landmarks observed in the sliding window. $$\{r_\rho, H_\rho\}$$ is the prior information derived from marginalization during the sliding windowbased optimization. $$\rho(·)$$ is the Huber loss function, a robustify function that makes system robust to outlier noise, as defined in (10):

<figure><img src="../../../.gitbook/assets/image (12) (2).png" alt=""><figcaption></figcaption></figure>

with $$\delta$$ as the parameter that can be adjusted for different levels of outlier suppression strength. To optimize the (9), the Levenberg-Marquardt (LM) algorithm is utilized:

<figure><img src="../../../.gitbook/assets/image (5) (2) (1).png" alt=""><figcaption></figcaption></figure>

where $$J$$ is the jacobian matrix of $$r(X )$$ w.r.t. the state vector $$X$$ . The algorithm iteratively solves for the $$\triangle x$$,and $$X$$ is updated from $$k$$ step to $$k + 1$$ step as follows:

<figure><img src="../../../.gitbook/assets/image (6) (2).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (10) (2).png" alt=""><figcaption></figcaption></figure>

#### Visual Landmark Residual

In our implementation, the visual feature points are detected using Shi-Tomasi algorithm, and tracked using the optical flow. The inverse depth model is adopted to describe the 3D visual landmarks. The observation of visual landmarks is defined in the normalized image plane, which is obtained by applying the inverse camera projection $$\pi^{-1}$$ to “lift” the pixels of observed visual landmarks in the image plane to the camera coordinate with the depth of 1. Considering the visual landmark $$c_i$$ firstly observed in frame $$C_{k_0}$$ , the residual of its observation in frame $$C_k$$ can be defined as:

<figure><img src="../../../.gitbook/assets/image (7) (2).png" alt=""><figcaption></figcaption></figure>

where $$z^k_{c_i}$$ and $$z^{k_0}_{c_i}$$ represent the normalized observations of visual feature $$c_i$$ in frame $$C_k$$ and $$C_{k_0}$$ . The Jacobian matrices $$J_{c_i} ({}^{C_0}x_{C_k} )$$ and $$J_{c_i} ({}^{C_0} x_{C_{k_0}} )$$ of $$r_{c_i} (z^k_{c_i} , X )$$ w.r.t. the camera pose $${}^{C_0} x_{C_k}$$ and $${}^{C_0}x_{C_{k_0}}$$ are derived on their $$se(3)$$ Lie Algebra manifold. With $$z$$ denoting the depth of $$p^{C_k}_{c_i}$$, then

<figure><img src="../../../.gitbook/assets/image (4) (2).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (5) (2).png" alt=""><figcaption></figcaption></figure>

where $$[·]_×$$ denotes the skew-symmetric matrix transformation. The Jacobian matrix $$J_{c_i} (λ_i )$$ w.r.t the inverse depth $$λ_i$$ is:

<figure><img src="../../../.gitbook/assets/image (8) (2).png" alt=""><figcaption></figcaption></figure>

#### HD Map Landmark Residual

First, the sampled control points of the HD map landmarks are transformed from the global coordinate $$G$$ into the local coordinate $$C_0$$ by $${}^Gx_{C_0}$$ , and then projected into the image plane with local camera pose $${}^{C_0} x_{C_k}$$ . For a sample point $$m_{i, j}$$ of HD map landmark $$M_i$$ , given its corresponding observation $$z^k_{m_{i,j}}$$ in frame $$C_k$$, the residual is defined as:

<figure><img src="../../../.gitbook/assets/image (2) (2).png" alt=""><figcaption></figcaption></figure>

The jacobian matrix $$J_{m_{i, j}} ({}^{C_0} x_{C_k} )$$ and $$J_{m_{i, j}} ({}^G x_{C_0} )$$ w.r.t $${}^{C_0} x_{C_k}$$ and $${}^G x_{C_0}$$ are:

<figure><img src="../../../.gitbook/assets/image (3) (2).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (9) (2).png" alt=""><figcaption></figcaption></figure>

and $$\frac{\delta D^{s_i}_k}{\delta m^{I_{k}}_{i,j}}$$ is approximated as the pixel gradient of $$D^{s_i}_k$$ at $$m^{I_k}_{i, j}$$ , i.e.:

<figure><img src="../../../.gitbook/assets/image (15) (2).png" alt=""><figcaption></figcaption></figure>

In our implementation, the number of sample points of HD map landmark features is around 150 for single frame. As a result, when performing the LM optimization, the HD map landmark residuals and jacobians needs around 150K times calculation for each LM optimization update step, making naive SCM not suitable for tightly-coupled sliding windowbased optimization.

#### Linear Approximated Residual

Next, we introduce the linearization approximation algorithm for accelerating the HD map residual calculation of SCM. Given the set $$M^k$$ of all sample points of HD map landmarks at frame $$k$$,when applying LM optimization algorithm in (11), the corresponding block of HD map residuals is:

<figure><img src="../../../.gitbook/assets/image (29) (4).png" alt=""><figcaption></figcaption></figure>

where $$H_{M^k}$$ and $$b_{M^k}$$ are calculated as:

<figure><img src="../../../.gitbook/assets/image (48).png" alt=""><figcaption></figcaption></figure>

Since $$H_{M^k}$$ is a symmetric matrix, one can perform Cholesky decomposition as $$H_{M^k} = J^T_{M^k} J_{M^k}$$ . By further introducing $$r_{M^k} = {(J^\dagger_{M^k})}^T b_{M^k}$$ , (23) can be transformed as:

<figure><img src="../../../.gitbook/assets/image (75).png" alt=""><figcaption></figcaption></figure>

This transformation indicates that the overall HD map residuals at frame $$k$$ are equivalent to one single residual block $$r$$ satisfying:

<figure><img src="../../../.gitbook/assets/image (1) (1) (1).png" alt=""><figcaption></figcaption></figure>

Normally, $$J_{M^k}$$ is relative to $${}^{C_0} x_{C_k}$$ and $${}^G x_{C_0}$$ . As a result, it should be re-calculated after each round of updates in the LM optimization. However, if $${}^{C_0} x_{C_k}$$ and $${}^G x_{C_0}$$ have been optimized several times in the previous sliding window-based optimizations, we can assume that they are already close to the local optimum and therefore will change only a little after each round of update. Under this assumption, the proposed algorithm is to replace $$J_{M^k}$$ by a constant jacobian  $$\overline{J}_{M^k}$$ that is jacobian of the HD map residuals at the initial value $$({}^{C_0̄} \overline{x}_{C_k}$$ , $${}^G \overline{x}_{C_0})$$ before optimization, deriving the linear approximated residual $$r_{LA}$$ as:

<figure><img src="../../../.gitbook/assets/image (2) (3).png" alt=""><figcaption></figcaption></figure>

This approximation avoids the jacobian re-calculation of HD map residuals at each LM optimization round, accelerating the overall state estimation. Notice that the approximation can only be reasonable when the states at frame $$k$$ are lying within the neighborhood of local optimum. To ensure the approximation is not applied on frames that have not been well solved, the algorithm introduces a variable $$n_k$$ for each frame k to record the lifetime of frame $$k$$ in the sliding window. Only frames with $$n_k$$ larger than a threshold $$N_T$$ will be approximated.

In the sliding window-based optimization, the $$K$$ camera poses from the past frames are selected as keyframes. In our implementation strategy, the latest frame is added as a new keyframe when it has enough visual parallax with the second latest keyframe in the sliding window, leading to the removal of the oldest keyframe in the sliding window; otherwise, the second latest keyframe will be discarded. With this strategy, the feature landmarks in the sliding window can be observed by the keyframes with enough parallax so that their 3D positions can be estimated more accurately.

In the cost model (4) of SCM, the data association result is strongly related to the initial guess of the state poses. Thus, in order to reduce the false data association in SCM, a good initial guess of the camera pose is required. Therefore, in our implementation, we have proposed the initial guess generation strategy with the aid of visual features. The strategy predicts the initial guess with the current frame observations of the visual feature landmarks, of which their 3D positions have already been estimated in the sliding window process. The results of this prediction serve as the initial guess for the SCM, thereby helping improve the accuracy in generating the final data association for optimization. The whole tightly-coupled sliding window-based map matching pipeline is outlined in Algorithm 1.

<figure><img src="../../../.gitbook/assets/image (174).png" alt=""><figcaption></figcaption></figure>
