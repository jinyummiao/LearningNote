---
description: Digging into Self-Supervised Monocular Depth Prediction
---

# \[ICCV 2019] MonoDepth2

{% embed url="https://www.github.com/nianticlabs/monodepth2" %}

## Abstract

In particular, we propose (i) a minimum reprojection loss, designed to robustly handle occlusions, (ii) a full-resolution multi-scale sampling method that reduces visual artifacts, and (iii) an auto-masking loss to ignore training pixels that violate camera motion assumptions.

## Introduction

We propose three architectural and loss innovations that combined, lead to large improvements in monocular depth estimation when training with monocular video, stereo pairs, or both: (1) A novel appearance matching loss to address the problem of occluded pixels that occur when using monocular supervision. (2) A novel and simple auto-masking approach to ignore pixels where no relative camera motion is observed in monocular training. (3) A multi-scale appearance matching loss that performs all image sampling at the input resolution, leading to a reduction in depth artifacts.

## Method

Here, we describe our depth prediction network that takes a single color input $$I_t$$ and produces a depth map $$D_t$$.

### Self-Supervised Training

We also formulate our problem as the minimization of a photometric reprojection error at training time. We express the relative pose for each source view $$I_{t'}$$ , with respect to the target image $$I_t$$’s pose, as $$T_{t\rightarrow t'}$$ . We predict a dense depth map $$D_t$$ that minimizes the photometric reprojection error $$L_p$$, where

<figure><img src="../../.gitbook/assets/image (805).png" alt=""><figcaption></figcaption></figure>

Here pe is a photometric reconstruction error, e.g. the L1 distance in pixel space; proj() are the resulting 2D coordinates of the projected depths $$D_t$$ in $$I_{t'}$$ and <> the sampling operator. For simplicity of notation we assume the pre-computed intrinsics K of all the views are identical, though they can be different. We use bilinear sampling to sample the source images, which is locally sub-differentiable, and we use L1 and SSIM to make our photometric error function pe, i.e.

<figure><img src="../../.gitbook/assets/image (772).png" alt=""><figcaption></figcaption></figure>

where $$\alpha=0.85$$. We use edge-aware smoothness

<figure><img src="../../.gitbook/assets/image (753).png" alt=""><figcaption></figcaption></figure>

where $$d^*_t=d_t / \overline{d_t}$$ is the mean-normalized inverse depth to discourage shrinking of the estimated depth.

In stereo training, our source image $$I_{t'}$$ is the second view in the stereo pair to $$I_t$$, which has known relative pose. While relative poses are not known in advance for monocular sequences, it is possible to train a second pose estimation network to predict the relative poses $$T_{t\rightarrow t'}$$ used in the projection function proj. During training, we solve for camera pose and depth simultaneously, to minimize $$L_p$$. For monocular training, we use the two frames temporally adjacent to $$I_t$$ as our source frames, i.e. $$I_{t'}\in \{I_{t-1},I_{t+1}\}$$. In mixed training (MS), $$I_{t'}$$ includes the temporally adjacent frames and the opposite stereo view.

### Improved Self-Supervised Depth Estimation

<figure><img src="../../.gitbook/assets/image (755).png" alt=""><figcaption></figcaption></figure>

#### Per-Pixel Minimum Reprojection Loss

At each pixel, instead of averaging the photometric error over all source images, we simply use the minimum. Our final per-pixel photometric loss is therefore

<figure><img src="../../.gitbook/assets/image (779).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (747).png" alt=""><figcaption></figcaption></figure>

#### Auto-Masking Stationary Pixels

Our second contribution: a simple auto-masking method that filters out pixels which do not change appearance from one frame to the next in the sequence. This has the effect of letting the network ignore objects which move at the same velocity as the camera, and even to ignore whole frames in monocular videos when the camera stops moving.

We also apply a per-pixel mask $$\mu$$ to the loss, selectively weighting pixels. However in contrast to prior work, our mask is binary, so $$\mu \in \{0,1\}$$, and is computed automatically on the forward pass of the network, instead of being learned or estimated from object motion. We observe that pixels which remain the same between adjacent frames in the sequence often indicate a static camera, an object moving at equivalent relative translation to the camera, or a low texture region. We therefore set $$\mu$$ to only include the loss of pixels where the reprojection error of the warped image $$I_{t\rightarrow t'}$$ is lower than that of the original, unwarped source image $$I_{t'}$$, i.e.

<figure><img src="../../.gitbook/assets/image (731).png" alt=""><figcaption></figcaption></figure>

where \[ ] is the Iverson bracket.

<figure><img src="../../.gitbook/assets/image (710).png" alt=""><figcaption></figcaption></figure>

#### Multi-scale Estimation

Instead of computing the photometric error on the ambiguous low-resolution images, we first upsample the lower resolution depth maps (from the intermediate layers) to the input image resolution, and then reproject, resample, and compute the error pe at this higher input resolution. This procedure is similar to matching patches, as low-resolution disparity values will be responsible for warping an entire ‘patch’ of pixels in the high resolution image. This effectively constrains the depth maps at each scale to work toward the same objective i.e. reconstructing the high resolution input target image as accurately as possible.

#### Final Training Loss

We combine our per-pixel smoothness and masked photometric losses as $$L=\mu L_p+\lambda L_s$$, and average over each pixel, scale, and batch.

### Additional Considerations

We use a ResNet18 as our encoder, which contains 11M parameters. We start with weights pretrained on ImageNet, and show that this improves accuracy for our compact model compared to training from scratch. Our depth decoder is with sigmoids at the output and ELU nonlinearities elsewhere. We convert the sigmoid output σ to depth with $$D=1/(a\sigma+b)$$, where a and b are chosen to constrain D between 0.1 and 100 units. We make use of reflection padding, in place of zero padding, in the decoder, returning the value of the closest border pixels in the source image when samples land outside of the image boundaries. We found that this significantly reduces the border artifacts found in existing approaches.

For pose estimation, we predict the rotation using an axis-angle representation, and scale the rotation and translation outputs by 0.01. For monocular training, we use a sequence length of three frames, while our pose network is formed from a ResNet18, modified to accept a pair of color images (or six channels) as input and to predict a single 6-DoF relative pose. We perform horizontal flips and the following training augmentations, with 50% chance: random brightness, contrast, saturation, and hue jitter with respective ranges of ±0.2, ±0.2, ±0.2, and ±0.1. Importantly, the color augmentations are only applied to the images which are fed to the networks, not to those used to compute $$L_p$$. All three images fed to the pose and depth networks are augmented with the same parameters.

Our models are implemented in PyTorch, trained for 20 epochs using Adam, with a batch size of 12 and an input/output resolution of 640 × 192 unless otherwise specified. We use a learning rate of $$10^{-4}$$ for the first 15 epochs which is then dropped to $$10^{-5}$$ for the remainder. This was chosen using a dedicated validation set of 10% of the data. The smoothness term $$\lambda$$ is set to 0.001.

## Experiments

<figure><img src="../../.gitbook/assets/image (790).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (770).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (734).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (784).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (741).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (715).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (785).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (730).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (740).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (752).png" alt=""><figcaption></figcaption></figure>

Post-processing is a technique to improve test time results on stereo-trained monocular depth estimation methods by running each test image through the network twice, once unflipped and then flipped. The two predictions are then masked and averaged. This has been shown to bring significant gains in accuracy for stereo results, at the expense of requiring two forward-passes through the network at test time

<figure><img src="../../.gitbook/assets/image (749).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (783).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (736).png" alt=""><figcaption></figcaption></figure>
