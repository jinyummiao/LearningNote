---
description: >-
  ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial, and
  Multimap SLAM
---

# \[TRO 2021] ORB-SLAM3

## Abstract

This article presents ORB-SLAM3, the first system able to perform visual, visual-inertial and multimap SLAM with monocular, stereo and RGB-D cameras, using pin-hole and fisheye lens models. The first main novelty is a tightly integrated visualinertial SLAM system that fully relies on maximum a posteriori (MAP) estimation, even during IMU initialization, resulting in real-time robust operation in small and large, indoor and outdoor environments, being two to ten times more accurate than previous approaches. The second main novelty is a multiple map system relying on a new place recognition method with improved recall that lets ORB-SLAM3 survive to long periods of poor visual information: when it gets lost, it starts a new map that will be seamlessly merged with previous maps when revisiting them. Compared with visual odometry systems that only use information from the last few seconds, ORB-SLAM3 is the first system able to reuse in all the algorithm stages all previous information from high parallax co-visible keyframes, even if they are widely separated in time or come from previous mapping sessions, boosting accuracy.&#x20;

## Introduction

Modern systems rely on maximum a posteriori (MAP) estimation which, in the case of visual sensors, corresponds to bundle adjustment (BA), either geometric BA that minimizes feature reprojection error, in feature-based methods, or photometric BA that minimizes the photometric error of a set of selected pixels, in direct methods.

The goal of visual SLAM is to use the sensors on-board a mobile agent to build a map of the environment and compute in real time the pose of the agent in that map. In contrast, VO systems put their focus on computing the agent’s ego-motion and not on building a map.

The big advantage of a SLAM map is that it allows matching and using in BA previous observations performing three types of data association:

* **Short-term data association**: matching map elements obtained during the last few seconds. This is the only data association type used by most VO systems, which forget environment elements once they get out of view, resulting in continuous estimation drift even when the system moves in the same area.
* **Mid-term data association**: matching map elements that are close to the camera whose accumulated drift is still small. These can be matched and used in BA in the same way than short-term observations and allow to reach zero drift when the systems move in mapped areas. They are the key to the better accuracy obtained by our system compared against VO systems with loop detection.
* **Long-term data association**: matching observations with elements in previously visited areas using a place recognition technique, regardless of the accumulated drift (loop detection), the current area being previously mapped in a disconnected map (map merging), or the tracking being lost (relocalization). Long-term matching allows to reset the drift and to correct the map using pose-graph (PG) optimization or, more accurately, using BA. This is the key to SLAM accuracy in medium and large loopy environments.

In this work, we build on ORB-SLAM and ORB-SLAM visual–inertial, the first visual and visual–inertial systems able to take full profit of short-term, mid-term, and longterm data association, reaching zero drift in mapped areas. Here, we go one step further providing multimap data association, which allows us to match and use in BA map elements coming from previous mapping sessions, achieving the true goal of a SLAM system: building a map that can be used later to provide accurate localization.

<figure><img src="../../.gitbook/assets/image (949).png" alt=""><figcaption></figcaption></figure>

The main novelties of ORB-SLAM3 are as follows:

* **A monocular and stereo visual–inertial SLAM system** that fully relies on MAP estimation, even during the inertial measurement unit (IMU) initialization phase. The initialization method proposed was previously presented in \[Inertial-only optimization for visual-inertial initialization]. Here, we add its integration with ORB-SLAM visual–inertial and a thorough evaluation in public datasets. Our results show that the monocular and stereo visual–inertial systems are extremely robust and significantly more accurate than other visual–inertial approaches, even in sequences without loops.
* **Improved-recall place recognition.** We propose a novel place recognition algorithm, in which candidate keyframes are first checked for geometrical consistency, and then for local consistency with three covisible keyframes, which in most occasions are already in the map. This strategy increases recall and densifies data association improving map accuracy, at the expense of a slightly higher computational cost.
* **ORB-SLAM Atlas.** The first complete multimap SLAM system able to handle visual and visual–inertial systems in monocular and stereo configurations. The Atlas can represent a set of disconnected maps and apply to them all the mapping operations smoothly: place recognition, camera relocalization, loop closure, and accurate seamless map merging. This allows to automatically use and combine maps built at different times, performing incremental multisession SLAM. A preliminary version of ORB-SLAM Atlas for visual sensors was presented in \[ORBSLAM-Atlas: A robust and accurate multi-map system]. Here we add the new place recognition system, the visual–inertial multimap system, and its evaluation on public datasets.
* **An abstract camera representation** making the SLAM code agnostic of the camera model used and allowing to add new models by providing their projection, unprojection, and Jacobian functions. We provide the implementations of pin-hole and fisheye models.

## System Overview

<figure><img src="../../.gitbook/assets/image (927).png" alt=""><figcaption></figcaption></figure>

* **Atlas** is a multimap representation composed of a set of disconnected maps. There is an active map where the tracking thread localizes the incoming frames and is continuously optimized and grown with new keyframes by the local mapping thread. We refer to the other maps in the Atlas as the nonactive maps. The system builds a unique DBoW2 database of keyframes that is used for relocalization, loop closing, and map merging.
* **Tracking thread** processes sensor information and computes the pose of the current frame with respect to the active map in real time, minimizing the reprojection error of the matched map features. It also decides whether the current frame becomes a keyframe. In visual–inertial mode, the body velocity and IMU biases are estimated by including the inertial residuals in the optimization. When tracking is lost, the tracking thread tries to relocalize the current frame in all the Atlas’ maps. If relocalized, tracking is resumed, switching the active map if needed. Otherwise, after a certain time, the active map is stored as nonactive, and a new active map is initialized from scratch.
* **Local mapping thread** adds keyframes and points to the active map, removes the redundant ones, and refines the map using visual or visual–inertial BA, operating in a local window of keyframes close to the current frame. Additionally, in the inertial case, the IMU parameters are initialized and refined by the mapping thread using our novel MAP-estimation technique.
* **Loop and map merging thread** detects common regions between the active map and the whole Atlas at keyframe rate. If the common area belongs to the active map, it performs loop correction; if it belongs to a different map, both maps are seamlessly merged into a single one, which becomes the active map. After a loop correction, a full BA is launched in an independent thread to further refine the map without affecting real-time performance.

## Camera Model

Our goal is to abstract the camera model from the whole SLAM pipeline by extracting all properties and functions related to the camera model (projection and unprojection functions, Jacobian, etc.) into separate modules. This allows our system to use any camera model by providing the corresponding camera module.

Rectify either the whole image or the feature coordinates to work in an ideal planar retina is problematic for fisheye lenses that can reach or surpass a field of view (FOV) of 180. Image rectification is not an option as objects in the periphery get enlarged and objects in the center lose resolution, hindering feature matching. Rectifying the feature coordinates requires using less than 180◦ FOV and causes trouble to many computer vision algorithms that assume uniform reprojection error along the image, which is far from true in rectified fisheye images.

### Relocalization

ORB-SLAM solves the relocalization problem by setting a perspective-n-points solver based on the ePnP algorithm, which assumes a calibrated pin-hole camera along all its formulation. To follow up with our approach, we need a PnP algorithm that works independently of the camera model used. For that reason, we have adopted maximum likelihood perspective-n-point algorithm \[MLPnP—A real-time maximum likelihood solution to the perspective-N-Point problem] that is completely decoupled from the camera model as it uses projective rays as input. The camera model just needs to provide an unprojection function passing from pixels to projection rays, to be able to use relocalization.

### Nonrectified Stereo SLAM

our system does not rely on image rectification, considering the stereo rig as two monocular cameras having the following:

1. a constant relative SE(3) transformation between them;
2. optionally, a common image region that observes the same portion of the scene.

These constrains allow us to effectively estimate the scale of the map by introducing that information when triangulating new landmarks and in the BA optimization. Following up with this idea, our SLAM pipeline estimates a 6 DoF rigid body pose, whose reference system can be located in one of the cameras or in the IMU sensor, and represents the cameras with respect to the rigid body pose.

If both cameras have an overlapping area in which we have stereo observations, we can triangulate true scale landmarks the first time they are seen. The rest of both images still has a lot of relevant information that is used as monocular information in the SLAM pipeline. Features first seen in these areas are triangulated from multiple views, as in the monocular case.

## Visual-Inertial SLAM

### Fundamentals

These are the body pose $$T_i=[R_i,p_i]\in SE(3)$$ and velocity $$v_i$$ in the world frame, and the gyroscope and accelerometer biases, $$b^g_i$$ and $$b^a_i$$, which are assumed to evolve according to a Brownian motion. This leads to the state vector

<figure><img src="../../.gitbook/assets/image (959).png" alt=""><figcaption></figcaption></figure>

For visual–inertial SLAM, we preintegrate IMU measurements between consecutive visual frames, $$i$$ and $$i+1$$, following the theory developed in \[Visual-inertial-aided navigation for highdynamic motion in built environments without initial conditions] and formulated on manifolds in \[On-manifold preintegration for real-time visual-inertial odometry]. We obtain preintegrated rotation, velocity, and position measurements, denoted as $$\Delta R_{i,i+1}, \Delta v_{i,i+1}$$ and $$\Delta p_{i,i+1}$$,as well as a covariance matrix $$\sum_{\mathcal{I}_{i,i+1}}$$ for the whole measurement vector. Given these preintegrated terms and states $$S_i$$ and $$S_{i+1}$$, we adopt the definition of inertial residual $$r_{\mathcal{I}_{i,i+1}}$$ from \[On-manifold preintegration for real-time visual-inertial odometry]

<figure><img src="../../.gitbook/assets/image (799).png" alt=""><figcaption></figcaption></figure>

where $$Log:SO(3)\rightarrow R^3$$ maps from the Lie group to the vector space. Together with inertial residuals, we also use reprojection errors $$r_{ij}$$ between frame $$i$$ and 3-D point $$j$$ at position $$x_j$$

<figure><img src="../../.gitbook/assets/image (708).png" alt=""><figcaption></figcaption></figure>

where $$\Pi: R^3 \rightarrow R^n$$ is the projection function for the corresponding camera model, $$u_{ij}$$ is the observation of point $$j$$ at image $$i$$, having a covariance matrix $$\sum_{ij}$$, $$T_{CB} \in SE(3)$$ stands for the rigid transformation from body-IMU to camera (left or right), known from calibration, and $$\oplus$$ is the transformation operation of $$SE(3)$$ group over $$R^3$$ elements.

Combining inertial and visual residual terms, visual–inertial SLAM can be posed as a keyframe-based minimization problem. Given a set of $$k+1$$ keyframes and its state $$\overline{S}_k\doteq \{S_0,...,S_k\}$$ and a set of $$l$$ 3-D points and its state $$X\doteq \{x_0,...,x_{l-1}\}$$, the visual–inertial optimization problem can be stated as follows:

<figure><img src="../../.gitbook/assets/image (758).png" alt=""><figcaption></figcaption></figure>

where $$K^j$$ is the set of keyframes observing 3-D point $$j$$.This optimization may be outlined as the factor-graph shown in Fig. 2(a). Note that for reprojection error, we use a robust Huber kernel $$\rho_{Hub}$$ to reduce the influence of spurious matchings, while for inertial residuals, it is not needed since miss-associations do not exist. This optimization needs to be adapted for efficiency during tracking and mapping, but, more importantly, it requires good initial seeds to converge to accurate solutions.

<figure><img src="../../.gitbook/assets/image (714).png" alt=""><figcaption></figcaption></figure>

### IMU Initilization

The goal of this step is to obtain good initial values for the inertial variables: body velocities, gravity direction, and IMU biases. In this work, we propose a fast and accurate initialization method based on the following three key insights.

1. Pure monocular SLAM can provide very accurate initial maps, whose main problem is that scale is unknown. Solving first the vision-only problem will enhance IMU initialization.
2. Scale converges much faster when it is explicitly represented as an optimization variable, instead of using the implicit representation of BA.
3. Ignoring sensor uncertainties during IMU initialization produces large unpredictable errors.

So, taking properly into account sensor uncertainties, we state the IMU initialization as a MAP estimation problem, split into the following three steps.

**1.Vision-Only MAP Estimation**: We initialize pure monocular ORB-SLAM and run it during 2 s, inserting keyframes at 4 Hz. After this period, we have an up-toscale map composed of k =10 camera poses and hundreds of points, which is optimized using visual-only BA \[Fig. 2(b)]. These poses are transformed to body reference, obtaining the trajectory $$\overline{T}_{0:k}={[R,\overline{p}]}_{0:k}$$ where the bar denotes up-to-scale variables in the monocular case.

**2.Inertial-Only MAP Estimation**: In this step, we aim to obtain the optimal estimation of the inertial variables, in the sense of MAP estimation, using only $$\overline{T}_{0:k}$$ and inertial measurements between these keyframes. These inertial variables may be stacked in the inertial-only state vector

<figure><img src="../../.gitbook/assets/image (711).png" alt=""><figcaption></figcaption></figure>

where $$s \in R^+$$ is the scale factor of the vision-only solution; $$R_{wg} \in SO(3)$$ is a rotation matrix used to compute gravity vector g in the world reference as $$g=R_{wg}g_I$$, where $$g_I={(0,0,G)}^T$$ and G is the gravity magnitude; $$b=(b^a,b^g) \in R^6$$ are the accelerometer and gyroscope biases assumed to be constant during initialization; and $$\overline{v}_{0:k} \in R^3$$ is the up-to-scale body velocities from first to last keyframe, initially estimated from $$\overline{T}_{0:k}$$. At this point, we are only considering the set of inertial measurements $$I_{0:k}\doteq\{I_{0,1},...,I_{k-1,k}\}$$. Thus, we can state an MAP estimation problem, where the posterior distribution to be maximized is

<figure><img src="../../.gitbook/assets/image (737).png" alt=""><figcaption></figcaption></figure>

Considering independence of measurements, the inertial-only MAP estimation problem can be written as

<figure><img src="../../.gitbook/assets/image (793).png" alt=""><figcaption></figcaption></figure>

Taking negative logarithm and assuming Gaussian error for IMU preintegration and prior distribution, this finally results in the optimization problem

<figure><img src="../../.gitbook/assets/image (766).png" alt=""><figcaption></figcaption></figure>

This optimization, represented in Fig. 2(c), differs from (4) in not including visual residuals, as the up-to-scale trajectory estimated by visual SLAM is taken as constant, and adding a prior residual that forces IMU biases to be close to zero. Covariance matrix $$\Sigma_b$$ represents prior knowledge about the range of values IMU biases may take. Details for preintegration of IMU covariance $$\Sigma_{I_{i-1,i}}$$ can be found at \[On-manifold preintegration for real-time visual-inertial odometry].

Since rotation around gravity direction does not suppose a change in gravity, this update is parameterized with two angles

<figure><img src="../../.gitbook/assets/image (725).png" alt=""><figcaption></figcaption></figure>

with Exp() being the exponential map from $$R^3$$ to SO(3). To guarantee that scale factor remains positive during optimization, we define its update as

<figure><img src="../../.gitbook/assets/image (767).png" alt=""><figcaption></figcaption></figure>

Once the inertial-only optimization is finished, the frame poses and velocities and the 3-D map points are scaled with the estimated scale factor and rotated to align the z-axis with the estimated gravity direction. Biases are updated and IMU preintegration is repeated, aiming to reduce future linearization errors.

**3.Visual–Inertial MAP Estimation**: Once we have a good estimation for inertial and visual parameters, we can perform a joint visual–inertial optimization for further refining the solution. This optimization may be represented as Fig. 2(a) but having common biases for all keyframes and including the same prior information for biases than in the inertial-only step.

This initialization is very efficient, achieving 5% scale error with trajectories of 2 s. To improve the initial estimation, visual–inertial BA is performed 5 and 15 s after initialization, converging to 1% scale error. After these BAs, we say that the map is mature, meaning that scale, IMU parameters, and gravity directions are already accurately estimated.

In some specific cases, when slow motion does not provide good observability of the inertial parameters, initialization may fail to converge to accurate solutions in just 15 s. To get robustness against this situation, we propose a novel scale refinement technique based on a modified inertial-only optimization, where all inserted keyframes are included, but scale and gravity directions are the only parameters to be estimated \[Fig. 2(d)]. Note that, in that case, the assumption of constant biases would not be correct. Instead, we use the values estimated from mapping, and we fix them. This optimization, which is very computationally efficient, is performed in the local mapping thread every 10 s until the map has more than 100 keyframes or more than 75 s have passed since initialization.

Finally, we have easily extended our monocular-inertial initialization to stereo-inertial by fixing the scale factor to one and taking it out from the inertial-only optimization variables, enhancing its convergence.

### Tracking and Mapping

For tracking and mapping, we adopt the schemes proposed in \[Visual-inertial monocular SLAM with map reuse]. Tracking solves a simplified visual–inertial optimization where only the states of the last two frames are optimized, while map points remain fixed.&#x20;

For mapping, trying to solve the whole optimization from (4) would be intractable for large maps. We use as optimizable variables a sliding window of keyframes and their points, including also observations to these points from covisible keyframes but keeping their pose fixed.

### Robustness to Tracking Loss

Our visual–inertial system enters into visually lost state when less than 15 point maps are tracked and achieves robustness in the following two stages:

1. **Short-term lost**: The current body state is estimated from IMU readings, and map points are projected in the estimated camera pose and searched for matches within a large image window. The resulting matches are included in visual–inertial optimization. In most cases, this allows to recover visual tracking. Otherwise, after 5 s, we pass to the next stage.
2. **Long-term lost**: A new visual–inertial map is initialized as explained above, and it becomes the active map.

If the system gets lost within 15 s after IMU initialization, the map is discarded. This prevents to accumulate inaccurate and meaningless maps.

## Map Merging and Loop Closing

In this work, we propose a new place recognition algorithm with improved recall for long-term and multimap data association. Whenever the mapping thread creates a new keyframe, place recognition is launched trying to detect matches with any of the keyframes already in the Atlas. If the matching keyframe found belongs to the active map, a loop closure is performed. Otherwise, it is a multimap data association, and then the active and the matching maps are merged. As a second novelty in our approach, once the relative pose between the new keyframe and the matching map is estimated, we define a local window with the matching keyframe and its neighbors in the covisibility graph. In this window, we intensively search for mid-term data associations, improving the accuracy of loop closing and map merging.

### Place Recognition

The steps of our place recognition algorithm are as follows:

1. **DBoW2 candidate keyframes**. We query the Atlas DBoW2 database with the active keyframe $$K_a$$ to retrieve the three most similar keyframes, excluding keyframes covisible with $$K_a$$. We refer to each matching candidate for place recognition as $$K_m$$.
2. **Local window.** For each $$K_m$$, we define a local window that includes $$K_m$$, its best covisible keyframes, and the map points observed by all of them. The DBoW2 direct index provides a set of putative matches between keypoints in $$K_a$$ and in the local window keyframes. For each of these 2D–2D matches, we have also the 3D–3D match available between their corresponding map points.
3. **3D aligning transformation**. We compute using RANSAC the transformation Tam that better aligns the map points in $$K_m$$ local window with those of $$K_{am}$$.In pure monocular, or in monocular-inertial when the map is still not mature, we compute $$T_{am}\in Sim(3)$$; otherwise, $$T_{am} \in SE(3)$$. In both cases, we use Horn algorithm using a minimal set of three 3D–3D matches to find each hypothesis for $$T_{am}$$. The putative matches that, after transforming the map point in $$K_a$$ by $$T_{am}$$, achieve a reprojection error in $$K_a$$ below a threshold give a positive vote to the hypothesis. The hypothesis with more votes is selected, provided the number is over a threshold.
4. **Guided matching refinement**. All the map points in the local window are transformed with $$T_{am}$$ to find more matches with the keypoints in $$K_a$$. The search is also reversed, finding matches for $$K_a$$ map points in all the keyframes of the local window. Using all the matchings found, $$T_{am}$$ is refined by nonlinear optimization, where the goal function is the bidirectional reprojection error, using Huber influence function to provide robustness to spurious matches. If the number of inliers after the optimization is over a threshold, a second iteration of guided matching and nonlinear refinement is launched, using a smaller image search window.
5. **Verification in three covisible keyframes**. To verify place recognition, we search in the active part of the map two keyframes covisible with $$K_a$$ where the number of matches with points in the local window is over a threshold. If they are not found, the validation is further tried with the new incoming keyframes, without requiring the bag-of-words to fire again. The validation continues until three keyframes verify $$T_{am}$$ or two consecutive new keyframes fail to verify it.
6. **VI gravity direction verification**. In the visual–inertial case, if the active map is mature, we have estimated $$T_{am} \in SE(3)$$. We further check whether the pitch and roll angles are below a threshold to definitively accept the place recognition hypothesis.

### Visual Map Merging

When a successful place recognition produces multimap data association between keyframe $$K_a$$ in the active map $$M_a$$, and a matching keyframe $$K_m$$ from a different map stored in the Atlas $$M_m$$, with an aligning transformation Tam, we launch a map merging operation. In the process, special care must be taken to ensure that the information in $$M_m$$ can be promptly reused by the tracking thread to avoid map duplication. For this, we propose to bring the $$M_a$$ map into $$M_m$$ reference. As $$M_a$$ may contain many elements and merging them might take a long time, merging is split into two steps. First, the merge is performed in a welding window defined by the neighbors of $$K_a$$ and $$K_m$$ in the covisibility graph, and in a second stage, the correction is propagated to the rest of the merged map by a PG optimization.

1. **Welding window assembly.** The welding window includes $$K_a$$ and its covisible keyframes, $$K_m$$ and its covisible keyframes, and all the map points observed by them. Before their inclusion in the welding window, the keyframes and map points belonging to $$M_a$$ are transformed by $$T_{ma}$$ to align them with respect to $$M_m$$.
2. **Merging maps.** Maps $$M_a$$ and $$M_m$$ are fused together to become the new active map. To remove duplicated points, matches are actively searched for $$M_a$$ points in the $$M_m$$ keyframes. For each match, the point from $$M_a$$ is removed, and the point in $$M_m$$ is kept accumulating all the observations of the removed point. The covisibility and essential graphs are updated by the addition of edges connecting keyframes from $$M_m$$ and $$M_a$$, thanks to the new mid-term point associations found.
3. **Welding bundle adjustment**. A local BA is performed optimizing all the keyframes from $$M_a$$ and $$M_m$$ in the welding window along with the map points which are observed by them \[Fig. 3(a)]. To fix gauge freedom, the keyframes of $$M_m$$ not belonging to the welding window but observing any of the local map points are included in the BA with their poses fixed. Once the optimization finishes, all the keyframes included in the welding area can be used for camera tracking, achieving fast and accurate reuse of map $$M_m$$.
4. **Essential-graph optimization**. A PG optimization is performed using the essential graph of the whole merged map, keeping fixed the keyframes in the welding area. This optimization propagates corrections from the welding window to the rest of the map.

### Visual–Inertial Map Merging

The visual–inertial merging algorithm follows similar steps than the pure visual case. Steps 1) and 3) are modified to better exploit the inertial information.

1. **VI welding window assembly**: If the active map is mature, we apply the available $$T_{ma}\in SE(3)$$ to map $$M_a$$ before its inclusion in the welding window. If the active map is not mature, we align $$M_a$$ using the available $$T_{ma}\in Sim(3)$$.
2. **VI welding bundle adjustment:** Poses, velocities, and biases of keyframes $$K_a$$ and $$K_m$$ and their five last temporal keyframes are included as optimizable. These variables are related by IMU preintegration terms, as shown in Fig. 3(b). For $$M_m$$, the keyframe immediately before the local window is included but fixed, while, for $$M_a$$, the similar keyframe is included, but its pose remains optimizable. All map points seen by the above-mentioned keyframes are optimized, together with poses from $$K_m$$ and $$K_a$$ covisible keyframes. All keyframes and points are related by means of reprojection error.

### Loop Closing

Loop closing correction algorithm is analogous to map merging but in a situation where both keyframes matched by place recognition belong to the active map. A welding window is assembled from the matched keyframes, and point duplicates are detected and fused creating new links in the covisibility and essential graphs. The next step is a PG optimization to propagate the loop correction to the rest of the map. The final step is a global BA to find the MAP estimate after considering the loop closure mid-term and long-term matches. In the visual–inertial case, the global BA is only performed if the number of keyframes is below a threshold to avoid a huge computational cost.

## Experimental Results

All experiments have been run on an Intel Core i7-7700 CPU, at 3.6 GHz, with 32 GB memory, using only CPU.

### Single-Session SLAM on EuRoC

<figure><img src="../../.gitbook/assets/image (727).png" alt=""><figcaption></figcaption></figure>

### Visual–Inertial SLAM on TUM-VI Benchmark

<figure><img src="../../.gitbook/assets/image (745).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (771).png" alt=""><figcaption></figcaption></figure>

### Multisession SLAM

<figure><img src="../../.gitbook/assets/image (765).png" alt=""><figcaption></figcaption></figure>

### Computing Time

<figure><img src="../../.gitbook/assets/image (764).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (788).png" alt=""><figcaption></figcaption></figure>
