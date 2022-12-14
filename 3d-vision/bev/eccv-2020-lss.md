---
description: >-
  Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly
  Unprojecting to 3D
---

# \[ECCV 2020] LSS

{% embed url="https://nv-tlabs.github.io/lift-splat-shoot" %}

## Abstract

The goal of perception for autonomous vehicles is to extract semantic representations from multiple sensors and fuse these representations into a single “bird’s-eye-view” coordinate frame for consumption by motion planning. We propose a new end-to-end architecture that directly extracts a bird’s-eye-view representation of a scene given image data from an arbitrary number of cameras. The core idea behind our approach is to “lift” each image individually into a frustum of features for each camera, then “splat” all frustums into a rasterized bird’s-eyeview grid. By training on the entire camera rig, we provide evidence that our model is able to learn not only how to represent images but how to fuse predictions from all cameras into a single cohesive representation of the scene while being robust to calibration error. On standard bird’seye-view tasks such as object segmentation and map segmentation, our model outperforms all baselines and prior work. In pursuit of the goal of learning dense representations for motion planning, we show that the representations inferred by our model enable interpretable end-to-end motion planning by “shooting” template trajectories into a bird’s-eyeview cost map output by our network.

<figure><img src="../../.gitbook/assets/image (135).png" alt=""><figcaption></figcaption></figure>

## Method

Formally, we are given $$n$$ images $$\{X_k ∈ R^{3×H×W}\}_n$$ each with an extrinsic matrix $$E_k ∈ R^{3×4}$$ and an intrinsic matrix $$I_k ∈ R^{3×3}$$, and we seek to find a rasterized representation of the scene in the BEV coordinate frame $$y ∈ R^{C×X×Y}$$ . The extrinsic and intrinsic matrices together define the mapping from reference coordinates $$(x, y, z)$$ to local pixel coordinates $$(h, w, d)$$ for each of the $$n$$ cameras. We do not require access to any depth sensor during training or testing.

### Lift: Latent Depth Distribution

The first stage of our model operates on each image in the camera rig in isolation. The purpose of this stage is to “lift” each image from a local 2-dimensional coordinate system to a 3-dimensional frame that is shared across all cameras.

<figure><img src="../../.gitbook/assets/image (61).png" alt=""><figcaption></figcaption></figure>

Let $$X ∈ R^{3×H×W}$$ be an image with extrinsics $$E$$ and intrinsics $$I$$, and let $$p$$ be a pixel in the image with image coordinates $$(h, w)$$. We associate $$|D|$$ points $$\{(h, w, d) ∈ R^3 | d ∈ D\}$$ to each pixel where $$D$$ is a set of discrete depths, for instance defined by $$\{d_0 + ∆, ..., d_0 + |D|∆\}$$. Note that there are no learnable parameters in this transformation. We simply create a large point cloud for a given image of size $$D · H · W$$ .

The context vector for each point in the point cloud is parameterized to match a notion of attention and discrete depth inference. At pixel $$p$$, the network predicts a context $$c ∈ R^C$$ and a distribution over depth $$α ∈ \triangle^{|D|−1}$$ for every pixel. The feature $$c_d ∈ R^C$$ associated to point $$p_d$$ is then defined as the context vector for pixel p scaled by $$α_d$$:

<figure><img src="../../.gitbook/assets/image (65).png" alt=""><figcaption></figcaption></figure>

Our network is therefore in theory capable of choosing between placing context from the image in a specific location of the bird’s-eyeview representation versus spreading the context across the entire ray of space, for instance if the depth is ambiguous.

In summary, ideally, we would like to generate a function $$g_c : (x, y, z) ∈ R^3 → c ∈ R^C$$ for each image that can be queried at any spatial location and return a context vector. To take advantage of discrete convolutions, we choose to discretize space. For cameras, the volume of space visible to the camera corresponds to a frustum.

### Splat: Pillar Pooling

“Pillars” are voxels with infinite height.

We assign every point to its nearest pillar and perform sum pooling to create a $$C × H × W$$ tensor that can be processed by a standard CNN for bird’s-eye-view inference. The overall lift-splat architecture is outlined in Figure 4.

<figure><img src="../../.gitbook/assets/image (137).png" alt=""><figcaption></figcaption></figure>

Just as OFT uses integral images to speed up their pooling step, we apply an analagous technique to speed up sum pooling. Efficiency is crucial for training our model given the size of the point clouds generated. Instead of padding each pillar then performing sum pooling, we avoid padding by using packing and leveraging a “cumsum trick” for sum pooling. This operation has an analytic gradient that can be calculated efficiently to speed up autograd.

### Shoot: Motion Planning

Key aspect of our Lift-Splat model is that it enables end-to-end cost map learning for motion planning from camera-only input. At test time, planning using the inferred cost map can be achieved by “shooting” different trajectories, scoring their cost, then acting according to lowest cost trajectory.&#x20;

We frame “planning” as predicting a distribution over K template trajectories for the ego vehicle

<figure><img src="../../.gitbook/assets/image (130).png" alt=""><figcaption></figcaption></figure>

conditioned on sensor observations $$p(τ |o)$$. Instead of the hard-margin loss proposed in NMP, we frame planning as classification over a set of $$K$$ template trajectories. To leverage the cost-volume nature of the planning problem, we enforce the distribution over K template trajectories to take the following form

<figure><img src="../../.gitbook/assets/image (66).png" alt=""><figcaption></figcaption></figure>

where $$c_o(x, y)$$ is defined by indexing into the cost map predicted given observations \$$\$$o at location $$x, y$$ and can therefore be trained end-to-end from data by optimizing for the log probability of expert trajectories. For labels, given a ground-truth trajectory, we compute the nearest neighbor in L2 distance to the template trajectories $$T$$ then train with the cross entropy loss. This definition of $$p(τ_i|o)$$ enables us to learn an interpretable spatial cost function without defining a hard-margin loss as in NMP.

n practice, we determine the set of template trajectories by running K-Means on a large number of expert trajectories. The set of template trajectories used for “shooting” onto the cost map that we use in our experiments are visualized in Figure 5.

<figure><img src="../../.gitbook/assets/image (83).png" alt=""><figcaption></figcaption></figure>

## Implementation

### Architecture Details

Our model has two large network backbones. One of the backbones operates on each image individually in order to featurize the point cloud generated from each image. The other backbone operates on the point cloud once it is splatted into pillars in the reference frame. The two networks are joined by our lift-splat layer.

For the network that operates on each image in isolation, we leverage layers from an EfficientNet-B0 pretrained on Imagenet.

For our bird’s-eye-view network, we use a combination of ResNet blocks. Specifically, after a convolution with kernel 7 and stride 2 followed by batchnorm and ReLU, we pass through the first 3 meta-layers of ResNet-18 to get 3 bird’s-eye-view representations at different resolutions $$x_1, x_2, x_3$$. We then upsample $$x_3$$ by a scale factor of 4, concatenate with $$x_1$$, apply a resnet block, and finally upsample by 2 to return to the resolution of the original input bird’s-eye-view pseudo image. We count 14.3M trainable parameters in our final network.

First, there is the size of the input images $$H × W$$ . In all experiments below, we resize and crop input images to size $$128 × 352$$ and adjust extrinsics and intrinsics accordingly. Another important hyperparameter of network is the size the resolution of the bird’s-eye-view grid $$X × Y$$. In our experiments, we set bins in both $$x$$ and $$y$$ from -50 meters to 50 meters with cells of size 0.5 meters × 0.5 meters. The resultant grid is therefore $$200×200$$. Finally, there’s the choice of $$D$$ that determines the resolution of depth predicted by the network. We restrict $$D$$ between 4.0 meters and 45.0 meters spaced by 1.0 meters. With these hyperparameters and architectural design choices, the forward pass of the model runs at 35 hz on a Titan V GPU.

### Frustum Pooling Cumulative Sum Trick

We choose sum pooling across pillars in Section 3 as opposed to max pooling because our “cumulative sum” trick saves us from excessive memory usage due to padding. The “cumulative sum trick” is the observation that sum pooling can be performed by sorting all points according to bin id, performing a cumulative sum over all features, then subtracting the cumulative sum values at the boundaries of the bin sections. Instead of relying on autograd to backprop through all three steps, the analytic gradient for the module as a whole can be derived, speeding up training by 2x. We call the layer “Frustum Pooling” because it handles converting the frustums produced by n images into a fixed dimensional $$C × H × W$$ tensor independent of the number of cameras $$n$$.

## Experiments and Results

For all object segmentation tasks, we train with binary cross entropy with positive weight 1.0. For the lane segmentation, we set positive weight to 5.0 and for road segmentation we use positive weight 1.0. In all cases, we train for 300k steps using Adam with learning rate $$1e − 3$$ and weight decay $$1e − 7.$$ We use the PyTorch framework.

<figure><img src="../../.gitbook/assets/image (85).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (127).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (87).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (95).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (113).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (100).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (117).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (128).png" alt=""><figcaption></figcaption></figure>
