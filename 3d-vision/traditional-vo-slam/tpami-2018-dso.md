---
description: Direct Sparse Odometry
---

# \[TPAMI 2018] DSO

{% embed url="https://github.com/JakobEngel/dso" %}

## Abstract

Direct Sparse Odometry (DSO) is a visual odometry method based on a novel, highly accurate sparse and direct structure and motion formulation. It combines a fully direct probabilistic model (minimizing a photometric error) with consistent, joint optimization of all model parameters, including geometry-represented as inverse depth in a reference frame-and camera motion. This is achieved in real time by omitting the smoothness prior used in other direct methods and instead sampling pixels evenly throughout the images. Since our method does not depend on keypoint detectors or descriptors, it can naturally sample pixels from across all image regions that have intensity gradient, including edges or smooth intensity variations on essentially featureless walls. The proposed model integrates a full photometric calibration, accounting for exposure time, lens vignetting, and non-linear response functions.

## Introduction

Underlying all formulations is a probabilistic model that takes noisy measurements Y as input and computes an estimator X for the unknown, hidden model parameters (3D world model and camera motion). Typically a Maximum Likelihood approach is used, which finds the model parameters that maximize the probability of obtaining the actual measurements, i.e., $$X^*:=argmax_XP(Y|X)$$.

In this paper we propose a sparse and direct approach to monocular visual odometry. To our knowledge, it is the only fully direct method that jointly optimizes the full likelihood for all involved model parameters, including camera poses, camera intrinsics, and geometry parameters (inverse depth values).

Optimization is performed in a sliding window, where old camera poses as well as points that leave the field of view of the camera are marginalized. In contrast to existing approaches, our method further takes full advantage of photometric camera calibration, including lens attenuation, gamma correction, and known exposure times. This integrated photometric calibration further increases accuracy and robustness.

## Direct Sparse Model

Our direct sparse odometry is based on continuous optimization of the photometric error over a window of recent frames, taking into account a photometrically calibrated model for image formation. In contrast to existing direct methods, we jointly optimize for all involved parameters (camera intrinsics, camera extrinsics, and inverse depth values), effectively performing the photometric equivalent of windowed sparse bundle adjustment. We keep the geometry representation employed by other direct approaches, i.e., 3D points are represented as inverse depth in a reference frame (and thus have one degree of freedom).

### Notation

Camera poses are represented as transformation matrices $$T_i\in SE(3)$$, transforming a point from the world frame into the camera frame. Linearized pose-increments will be expressed as Lie-algebra elements $$x_i\in se(3)$$, which-with a slight abuse of notation-we directly write as vectors $$x_i \in X^6$$. We further define the commonly used operator $$\boxplus: se(3)\times SE(3)\rightarrow SE(3)$$ using a left-multiplicative formulation, i.e.,

<figure><img src="../../.gitbook/assets/image (802).png" alt=""><figcaption></figcaption></figure>

### Calibration

#### Geometric Camera Calibration

For simplicity, we formulate our method for the well-known pinhole camera model—radial distortion is removed in a preprocessing step. Throughout this paper, we will denote projection by $$\Pi_c: R^3\rightarrow \Omega$$ and back-projection with $$\Pi^{-1}_c: \Omega \times R \rightarrow R^3$$, where c denotes the intrinsic camera parameters (for the pinhole model these are the focal length and the principal point).

### Photometric Camera Calibration

We use the image formation model used in \[A photometrically calibrated benchmark for monocular visual odometry], which accounts for a non-linear response function $$G: R\rightarrow [0,255]$$, as well as lens attenuation (vignetting) $$V: \Omega \rightarrow [0,1]$$. Fig. 3 shows an example calibration from the TUM monoVO dataset. The combined model is then given by

<figure><img src="../../.gitbook/assets/image (792).png" alt=""><figcaption></figcaption></figure>

where $$B_i$$ and $$I_i$$ are the irradiance and the observed pixel intensity in frame $$i$$, and $$t_i$$ is the exposure time. The model is applied by photometrically correcting each video frame as very first step, by computing

<figure><img src="../../.gitbook/assets/image (723).png" alt=""><figcaption></figcaption></figure>

In the remainder of this paper, $$I_i$$ will always refer to the photometrically corrected image $$I'_i$$, except where otherwise stated.

<figure><img src="../../.gitbook/assets/image (800).png" alt=""><figcaption></figcaption></figure>

### Model Formulation

<figure><img src="../../.gitbook/assets/image (782).png" alt=""><figcaption></figcaption></figure>

We define the photometric error of a point $$p \in \Omega_i$$ in reference frame $$I_i$$, observed in a target frame $$I_j$$, as the weighted SSD over a small neighborhood of pixels. Our experiments have shown that eight pixels, arranged in a slightly spread pattern (see Fig. 4) give a good trade-off between computations required for evaluation, robustness to motion blur, and providing sufficient information. Note that in terms of the contained information, evaluating the SSD over such a small neighborhood of pixels is similar to adding first- and second-order irradiance derivative constancy terms (in addition to irradiance constancy) for the central pixel. Let

<figure><img src="../../.gitbook/assets/image (803).png" alt=""><figcaption></figcaption></figure>

where $$N_p$$ is the set of pixels included in the SSD; $$t_i,t_j$$ the exposure times of the images $$I_i,I_j$$; and $$||\cdot||_\gamma$$ the Huber norm. Further, $$p'$$ stands for the projected point position of $$p$$ with inverse depth $$d_p$$, given by

<figure><img src="../../.gitbook/assets/image (738).png" alt=""><figcaption></figcaption></figure>

with

<figure><img src="../../.gitbook/assets/image (746).png" alt=""><figcaption></figcaption></figure>

In order to allow our method to operate on sequences without known exposure times, we include an additional affine brightness transfer function given by $$e^{-a_i}(I_i-b_i)$$.Note that in contrast to most previous formulations, the scalar factor $$e^{-a_i}$$ is parametrized logarithmically. This both prevents it from becoming negative and avoids numerical issues arising from multiplicative (i.e., exponentially increasing) drift.

In addition to using robust Huber penalties, we apply a gradient-dependent weighting $$w_p$$ given by

<figure><img src="../../.gitbook/assets/image (801).png" alt=""><figcaption></figcaption></figure>

which down-weights pixels with high gradient. This weighting function can be probabilistically interpreted as adding small, independent geometric noise on the projected point position $$p'$$, and immediately marginalizing it-approximating small geometric error. To summarize, the error $$E_{p_j}$$ depends on the following variables: (1) the point’s inverse depth $$d_p$$, (2) the camera intrinsics c, (3) the poses of the involved frames $$T_i,T_j$$, and (4) their brightness transfer function parameters $$a_i,b_i,a_j,b_j$$.

The full photometric error over all frames and points is given by

<figure><img src="../../.gitbook/assets/image (780).png" alt=""><figcaption></figcaption></figure>

where $$i$$ runs over all frames $$F$$ , $$p$$ over all points $$P_i$$ in frame $$i$$, and $$j$$ over all frames $$obs(p)$$ in which the point $$p$$ is visible.

<figure><img src="../../.gitbook/assets/image (798).png" alt=""><figcaption></figcaption></figure>

Fig. 5 shows the resulting factor graph: The only difference to the classical reprojection error is the additional dependency of each residual on the pose of the host frame, i.e., each term depends on two frames instead of only one. While this adds off-diagonal entries to the pose-pose block of the Hessian, it does not affect the sparsity pattern after application of the Schur complement to marginalize point parameters. The resulting system can thus be solved analogously to the indirect formulation. Note that the Jacobians with respect to the two frames’ poses are linearly related by the adjoint of their relative pose. In practice, this factor can then be pulled out of the sum when computing the Hessian or its Schur complement, greatly reducing the additional computations caused by more variable dependencies.

If exposure times are known, we further add a prior pulling the affine brightness transfer function to zero

<figure><img src="../../.gitbook/assets/image (778).png" alt=""><figcaption></figcaption></figure>

If no photometric calibration is available, we set $$t_i=1$$ and $$\lambda_a=\lambda_b=0$$, as in this case they need to model the (unknown) changing exposure time of the camera. As a side-note it should be mentioned that the ML estimator for a multiplicative factor $$a^*=argmax_a \sum_i (ax_i-y_i)^2$$is biased if both $$x_i$$ and $$y_i$$ contain noisy measurements; causing a to drift in the unconstrained case $$\lambda_a=0$$. While this generally has little effect on the estimated poses, it may introduce a bias if the scene contains only few, weak intensity variations.

**Point Dimensionality.** In the proposed direct model, a point is parametrized by only one parameter (the inverse depth in the reference frame), in contrast to three unknowns as in the indirect model. In addition to a reduced number of parameters, this naturally enables an inverse depth parametrization, which-in a Gaussian frameworkis better suited to represent uncertainty from stereo-based depth estimation, in particular for far-away points.

**Consistency**. Strictly speaking, the proposed direct sparse model does allow to use some observations (pixel values) multiple times, while others are not used at all. This is because-even though our point selection strategy attempts to avoid this by equally distributing points in space-we allow point observations to overlap, and thus depend on the same pixel value(s). This particularly happens in scenes with little texture, where all points have to be chosen from a small subset of textured image regions. We however argue that this has negligible effect in practice, and-if desired-can be avoided by removing (or downweighting) observations that use the same pixel value.

### Windowed Optimization

We optimize the total error (8) in a sliding window using the Gauss-Newton algorithm, which gives a good trade-off between speed and flexibility.

For ease of notation, we extend the $$\boxplus$$ operator as defined in (1) to all optimized parameters—for parameters other than SE(3) poses it denotes conventional addition. We will use $$\zeta \in SE(3)^n \times R^m$$ to denote all optimized variables, including camera poses, affine brightness parameters, inverse depth values, and camera intrinsics.

Marginalizing a residual that depends on a parameter in $$\zeta$$ will fix the tangent space in which any future information (delta-updates) on that parameter is accumulated. We will denote the evaluation point for this tangent space with $$\zeta_0$$, and the accumulated delta-updates by $$x \in se(3)^n \times R^m$$. The current state estimate is hence given by $$\zeta=x\boxplus \zeta_0$$.

**Gauss-Newton Optimization.** We compute the Gauss-Newton system as

<figure><img src="../../.gitbook/assets/image (756).png" alt=""><figcaption></figcaption></figure>

where $$W\in R^{n \times n}$$ is the diagonal matrix containing the weights, $$r\in R^n$$ is the stacked residual vector, and $$J \in R^{n\times d}$$ is the Jacobian of $$r$$.

Note that each point contributes $$|N_p|=8$$ residuals to the energy. For notational simplicity, we will in the following consider only a single residual $$r_k$$, and the associated row of the Jacobian $$J_k$$. During optimization—as well as when marginalizing—residuals are always evaluated at the current state estimate, i.e.,

<figure><img src="../../.gitbook/assets/image (781).png" alt=""><figcaption></figcaption></figure>

where $$(T_i,T_j,d,c,a_i,a_j,b_i,b_j):=x\boxplus \zeta_0$$ are the current state variables the residual depends on. The Jacobian $$J_k$$ is evaluated with respect to an additive increment to x, i.e.,

<figure><img src="../../.gitbook/assets/image (748).png" alt=""><figcaption></figcaption></figure>

It can be decomposed as

<figure><img src="../../.gitbook/assets/image (709).png" alt=""><figcaption></figcaption></figure>

where $$\delta_{geo}$$ denotes the “geometric” parameters $$(T_i,T_j,d,c)$$, and $$\delta_{photo}$$ denotes the “photometric” parameters $$(a_i,a_j,b_i,b_j)$$. We employ two approximations, described below.

First, both $$J_{photo}$$ and $$J_{geo}$$ are evaluated at $$x=0$$. This technique is called “First Estimate Jacobians”, and is required to maintain consistency of the system and prevent the accumulation of spurious information. In particular, in the presence of non-linear null-spaces in the energy (in our formulation absolute pose and scale), adding linearizations around different evaluation points eliminates these and thus slowly corrupts the system. In practice, this approximation is very good, since $$J_{photo}$$, $$J_{geo}$$ are smooth compared to the size of the increment $$x$$. In contrast, $$J_I$$ is much less smooth, but does not affect the null-spaces. Thus, it is evaluated at the current value for $$x$$, i.e., at the same point as the residual $$r_k$$. We use centred differences to compute the image derivatives at integer positions, which are then bilinearly interpolated.

Second, $$J_{geo}$$ is assumed to be the same for all residuals belonging to the same point, and evaluated only for the center pixel. Again, this approximation is very good in practice. While it significantly reduces the required computations, we have not observed a notable effect on accuracy for any of the used datasets.

From the resulting linear system, an increment is computed as $$\delta=H^{-1}b$$ and added to the current state

<figure><img src="../../.gitbook/assets/image (760).png" alt=""><figcaption></figcaption></figure>

Note that due to the First Estimate Jacobian approximation, a multiplicative formulation (replacing $$(\delta+x)\boxplus \zeta_0$$ with $$\delta \boxplus (x \boxplus \delta_0)$$ in (12)) results in the exact same Jacobian, thus a multiplicative update step $$x_{new}\leftarrow log(\delta \boxplus e^x)$$ is equally valid.

After each update step, we update $$\zeta_0$$ for all variables that are not part of the marginalization term, using $$\zeta^{new}_0 \leftarrow x \boxplus \zeta_0$$ and $$x \leftarrow 0$$. In practice, this includes all depth values, as well as the pose of the newest keyframe. Each time a new keyframe is added, we perform up to 6 Gauss-Newton iterations, breaking early if $$\delta$$ is sufficiently small. We found that-since we never start far-away from the minimum—a Levenberg-Marquardt dampening (which slows down convergence) is not required.

**Marginalization**. When the active set of variables becomes too large, old variables are removed by marginalization using the Schur complement. We drop any residual terms that would affect the sparsity pattern of H: When marginalizing frame i, we first marginalize all points in $$P_i$$, as well as points that have not been observed in the last two keyframes. Remaining observations of active points in frame i are dropped from the system.

Marginalization proceeds as follows: Let $$E'$$ denote the part of the energy containing all residuals that depend on state variables to be marginalized. We first compute a Gauss-Newton approximation of $$E'$$ around the current state estimate $$\zeta=x \boxplus \zeta_0$$. This gives

<figure><img src="../../.gitbook/assets/image (761).png" alt=""><figcaption></figcaption></figure>

where $$x_0$$ denotes the current value (evaluation point for $$r$$)of $$x$$. The constants $$c,c'$$ can be dropped, and $$H,b$$ are defined as in (10), (11), (12), and (13). This is a quadratic function on $$x$$, and we can apply the Schur complement to marginalize a subset of variables. Written as a linear system, it becomes

<figure><img src="../../.gitbook/assets/image (719).png" alt=""><figcaption></figcaption></figure>

where $$\beta$$ denotes the block of variables we would like to marginalize, and $$\alpha$$ the block of variables we would like to keep. Applying the Schur complement yields $$\hat{H_{\alpha\alpha}}x_\alpha=\hat{b'_{\alpha}}$$, with

<figure><img src="../../.gitbook/assets/image (728).png" alt=""><figcaption></figcaption></figure>

The residual energy on xxa can hence be written as

<figure><img src="../../.gitbook/assets/image (768).png" alt=""><figcaption></figcaption></figure>

This is a quadratic function on x and can be trivially added to the full photometric error $$E_{photo}$$ during all subsequent optimization and marginalization operations, replacing the corresponding non-linear terms. Note that this requires the tangent space for $$\zeta_0$$ to remain the same for all variables that appear in $$E'$$ during all subsequent optimization and marginalization steps.

## Visual Odometry Front-end

The front end is the part of the algorithm that

* determines the sets $$F,P_i$$ and $$obs(p)$$ that make up the error terms of $$E_{photo}$$. It decides which points and frames are used, and in which frames a point is visible-in particular, this includes outlier removal and occlusion detection.
* provides initializations for new parameters, required for optimizing the highly non-convex energy function $$E_{photo}$$. As a rule of thumb, a linearization of the image $$I$$ is only valid in a 1-2 pixel radius; hence all parameters involved in computing $$p'$$ should be initialized sufficiently accurately for $$p'$$ to be off by no more than 1-2 pixels.
* decides when a point/frame should be marginalized.

### Frame Management

Our method always keeps a window of up to $$N_f$$ active keyframes (we use $$N_f$$). Every new frame is initially tracked with respect to these reference frames (Step 1). It is then either discarded or used to create a new keyframe (Step 2). Once a new keyframe-and respective new pointsare created, the total photometric error (8) is optimized. Afterwards, we marginalize one or more frames (Step 3).

#### Step 1: Initial Frame Tracking.&#x20;

When a new keyframe is created, all active points are projected into it and slightly dilated, creating a semi-dense depth map. New frames are tracked with respect to only this frame using conventional two-frame direct image alignment, a multi-scale image pyramid and a constant motion model to initialize. Note that when down-scaling the images, a pixel is assigned a depth value if at least one of the source pixels has a depth value, significantly increasing the density on coarser resolutions.

If the final RMSE for a frame is more than twice that of the frame before, we assume that direct image alignment failed and attempt to recover by initializing with up to 27 different small rotations in different directions. This recovery-tracking is done on the coarsest pyramid level only, and takes approximately 0.5 ms per try. Note that this RANSAC-like procedure is only rarely invoked, such as when the camera moves very quickly or shakily. Tightly integrating an IMU would likely render this unnecessary.

#### Step 2: Keyframe Creation.&#x20;

Similar to ORB-SLAM, our strategy is to initially take many keyframes (around 5-10 keyframes per second), and sparsify them afterwards by early marginalizing redundant keyframes. We combine three criteria to determine if a new keyframe is required:

1. New keyframes need to be created as the field of view changes. We measure this by the mean square optical flow (from the last keyframe to the latest frame) $$f:=(\frac{1}{n}\sum^n_{i=1} {||p-p'||}^2)^{\frac{1}{2}}$$ during initial coarse tracking.
2. Camera translation causes occlusions and disocclusions, which requires more keyframes to be taken (even though $$f$$ may be small). This is measured by the mean flow without rotation, i.e., $$f_t:=(\frac{1}{n}\sum^n_{i=1} {||p-p'_t||}^2)^{\frac{1}{2}}$$, where $$p_t$$ is the warped point position with $$R=I_{3 \times 3}$$.
3. If the camera exposure time changes significantly, a new keyframe should be taken. This is measured by the relative brightness factor between two frames $$a:=|log(e^{a_j-a_i}t_jt^{-1}_j)|$$.

These three quantities can be obtained easily as a by-product of initial alignment. Finally, a new keyframe is taken if $$w_f f+w_{f_t}f_t+w_aa>T_{kf}$$, where $$w_f,w_{f_t},w_a$$ provide a relative weighting of these three indicators, and $$T_{kf}=1$$ by default.

#### Step 3: Keyframe Marginalization.&#x20;

Our marginalization strategy is as follows (let $$I_1,...,I_n$$ be the set of active keyframes, with $$I_n$$ being the newest and In being the oldest):

1. We always keep the latest two keyframes ($$I_1$$ and $$I_2$$).
2. Frames with less than 5 percent of their points visible in $$I_1$$ are marginalized.
3. If more than $$N_f$$ frames are active, we marginalize the one (excluding $$I_1$$ and $$I_2$$) which maximizes a “distance score” $$s(I_i)$$, computed as below, where $$d(i,j)$$ is the euclidean distance between keyframes $$I_i$$ and $$I_j$$, and $$\epsilon$$ a small constant. This scoring function is heuristically designed to keep active keyframes well-distributed in 3D space, with more keyframes close to the most recent one.

<figure><img src="../../.gitbook/assets/image (716).png" alt=""><figcaption></figcaption></figure>

A keyframe is marginalized by first marginalizing all points represented in it, and then the frame itself, using the marginalization procedure. To preserve the sparsity structure of the Hessian, all observations of still existing points in the frame are dropped from the system. While this is clearly suboptimal (in practice about half of all residuals are dropped for this reason), it allows to efficiently optimize the energy function.

### Point Management

We aim at always keeping a fixed number Np of active points (we use $$N_p=2000$$), equally distributed across space and active frames, in the optimization. In a first step, we identify $$N_p$$ candidate points in each new keyframe (Step 1). Candidate points are not immediately added into the optimization, but instead are tracked individually in subsequent frames, generating a coarse depth value which will serve as initialization (Step 2). When new points need to be added to the optimization, we choose a number of candidate points (from across all frames in the optimization window) to be activated, i.e., added into the optimization (Step 3). Note that we choose $$N_p$$ candidates in each frame, however only keep $$N_p$$ active points across all active frames combined. This assures that we always have sufficient candidates to activate, even though some may become invalid as they leave the field of view or are identified as outliers.

#### Step 1: Candidate Point Selection.&#x20;

Our point selection strategy aims at selecting points that are (1) well-distributed in the image and (2) have sufficiently high image gradient magnitude with respect to their immediate surroundings. We obtain a region-adaptive gradient threshold by splitting the image into $$32 \times 32$$ blocks. For each block, we then compute the threshold as $$\overline{g}+g_{th}$$, where $$\overline{g}$$ is the median absolute gradient over all pixels in that block, and $$g_{th}$$ a global constant (we use $$g_{th}=7$$).

To obtain an equal distribution of points throughout the image, we split it into $$d\times d$$ blocks, and from each block select the pixel with largest gradient if it surpasses the region-adaptive threshold. Otherwise, we do not select a pixel from that block. We found that it is often beneficial to also include some points with weaker gradient from regions where no high-gradient points are present, capturing information from weak intensity variations originating. To achieve this, we repeat this procedure twice more, with decreased gradient threshold and block-size $$2d$$ and $$4d$$, respectively. The block-size $$d$$ is continuously adapted such that this procedure generates the desired amount of points (if too many points were created it is increased for the next frame, otherwise it is decreased). Note that for for candidate point selection, we use the raw images prior to photometric correction.

#### Step 2: Candidate Point Tracking.&#x20;

Point candidates are tracked in subsequent frames using a discrete search along the epipolar line, minimizing the photometric error (4). From the best match we compute a depth and associated variance, which is used to constrain the search interval for the subsequent frame. Note that the computed depth only serves as initialization once the point is activated.

#### Step 3: Candidate Point Activation.&#x20;

After a set of old points is marginalized, new point candidates are activated to replace them. Again, we aim at maintaining a uniform spatial distribution across the image. To this end, we first project all active points onto the most recent keyframe. We then activate candidate points which—also projected into this keyframe—maximize the distance to any existing point (requiring larger distance for candidates created during the second or third block-run).&#x20;

#### Outlier and Occlusion Detection.&#x20;

Since the available image data generally contains much more information than can be used in real time, we attempt to identify and remove potential outliers as early as possible. First, when searching along the epipolar line during candidate tracking, points for which the minimum is not sufficiently distinct are permanently discarded, greatly reducing the number of false matches in repetitive areas. Second, point observations for which the photometric error (4) surpasses a threshold are removed. The threshold is continuously adapted with respect to the median residual in the respective frame. For “bad” frames (e.g., frames that contain a lot of motion blur), the threshold will be higher, such that not all observations are removed. For good frames, in turn, the threshold will be lower, as we can afford to be more strict.

## Results

<figure><img src="../../.gitbook/assets/image (732).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (797).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (739).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (791).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (733).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (729).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (794).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (759).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (707).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (795).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (751).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (721).png" alt=""><figcaption></figcaption></figure>
