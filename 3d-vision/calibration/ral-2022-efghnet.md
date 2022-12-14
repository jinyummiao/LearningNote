---
description: >-
  EFGHNet: A Versatile Image-to-Point Cloud Registration Network for Extreme
  Outdoor Environment
---

# \[RAL 2022] EFGHNet

{% embed url="https://github.com/yurimjeon1892/EFGH" %}

## Abstract

We present an accurate and robust image-to-point cloud registration method that is viable in urban and off-road environments. Existing image-to-point cloud registration methods have focused on vehicle platforms along paved roads. Therefore, image-to-point cloud registration on UGV platforms for off-road driving remains an open question. Our objective is to find a versatile solution for image-to-point cloud registration. We present a method that stably estimates a precise transformation between an image and a point cloud using a two-phase method that aligns the two input data in the virtual reference coordinate system (virtualalignment) and then compares and matches the data to complete the registration (compare-and-match). Our main contribution is the introduction of divide-and-conquer strategies to image-to-point cloud registration. The virtual-alignment phase effectively reduces relative pose differences without cross-modality comparison. The compare-and-match phase divides the process of matching the image and point cloud into the rotation and translation steps. By breaking down the registration problem, it is possible to develop algorithms that can robustly operate in various environments.

## Introduction

<figure><img src="../../.gitbook/assets/image (98).png" alt=""><figcaption></figcaption></figure>

In this study, we propose an image-to-point cloud registration method called EFGHNet Fig. 1. We build a two-phase method that aligns the two input data in the virtual reference coordinate system (virtual-alignment) and then compares and matches the data to complete the registration (compare-and-match). The virtual-alignment phase exploits the unique features of data. The horizon is used to align the image to the virtual reference coordinate system, that is, the image is aligned by matching an estimated horizon to a standard basis vector $$e_2 =[0, 1, 0] (T_H)$$. Similarly, a ground normal vector is used to align the point cloud. The point cloud is firstly aligned by matching an estimated ground normal vector to a standard basis vector $$e_3 =[0, 0, 1] (T_E)$$. The compare-and-match phase compares the two data in the virtual reference coordinate system. It is assumed that the forward axis of the image is $$e_1 =[1, 0, 0]$$. The area of overlap of the point cloud with respect to the image is estimated, and the point cloud is rotated to match its forward axis to $$e_1 (T_F)$$. Next, the registration process is completed by estimating the displacement of the image and the point cloud and using it to match the origin of both coordinate systems ($$T_G$$).

The key contributions of our paper are:&#x20;

* We propose a novel image-to-point cloud registration network named EFGHNet, which provides a versatile solution to the image-to-point cloud registration problem.
* Our method has a two-phase structure using a divide-andconquer strategy, so it can reliably estimate transformations even with large differences in initial pose, and at the same time have high accuracy.
* We verify the performance of the proposed method in various situations through extensive experiments using four datasets in different platforms and environments.

## Methods

We define an image-to-point cloud registration problem as follows. The inputs are an image $$x^I_{in} \in R^{H\times W \times 3}$$ and a point cloud $$x^P_{in} \in R^{N×\times 3}$$, and the output is the transformation matrix $$T$$ . The overall registration process is divided into four steps, handled in subnetworks E3, Forward-axis, Gather, and Horizon. Each subnetwork estimates a transformation matrix, that is, $$T_E$$, $$T_F$$, $$T_G$$, and $$T_H$$ . The final result of the registration is the product of the matrices $$T = T_G · T_F · T_E$$, indicating the transformation from the point cloud sensor to the image sensor. In the following sections, $$p$$ refers to the prediction, $$y$$ refers to the ground truth, superscript $$I$$ denotes the image, and $$P$$ denotes the point cloud. The reference frame consists of $$e_1,e_2$$, and $$e_3$$.

### E3 Network

<figure><img src="../../.gitbook/assets/image (90).png" alt=""><figcaption></figcaption></figure>

The first step involves aligning the input point cloud to the virtual reference coordinate system Fig. 2. The E3 network estimates the ground normal vector $$p_{gn}$$ from the point cloud $$x^P_{in}$$. The output of the network $$T_E$$ is a transformation matrix that rotates $$p_{gn}$$ to fit the standard basis vector $$e_3 =[0, 0, 1]$$.

We implement the E3 network using the DownBCL block, which processes the point cloud and extracts features. By stacking this layer, the DownBCL block can learn point cloud information distributed over a large area. The extracted features of DownBCL are fed into the rotation head.&#x20;

We design a rotation head to estimate the rotation vector pr using the spherical regression framework, which is a general solution to the regression of the n-sphere problem. The rotation head makes two predictions: the absolute value $$p_{r_{abs}} \in R^{1 \times r_{dim}}$$ and sign value $$p_{r_{sgn}} \in R^{1 \times x^{r_{dim}}}$$ of $$p_r$$. The sign prediction is $$2^{r_{dim}}$$-dim, because the + and − signs of $$p_r$$ are encoded as a one-hot vector. In E3 network, the rotation head estimates the ground normal vector $$p_{gn} \in R^{1\times3}$$, where $$r_{dim} =3$$.

The loss function $$L_r$$ of the rotation head consists of two parts: the absolute and sign parts.

<figure><img src="../../.gitbook/assets/image (105).png" alt=""><figcaption></figcaption></figure>

where the loss for the absolute part is the cosine proximity loss and that for the sign part is the cross-entropy loss (denoted as CE).

### Horizon Network

<figure><img src="../../.gitbook/assets/image (126).png" alt=""><figcaption></figcaption></figure>

In the second step, the input image is aligned with the virtual reference coordinate system Fig. 3. The horizon network estimates the horizontal vector phv from image $$x^I_{in}$$.A horizontal vector is defined as a vector that is parallel to the horizon in the image. When the right-end pixel of the horizon is $$[u_r,v_r]$$ and the left-end is $$[u_l,v_l]$$, the horizon is defined as $$y_{hv} =[u_l − u_r,v_l − v_r, 0]$$. The output of the network $$T_H$$ rotates $$p_{hv}$$ to fit $$e_2 =[0, 1, 0]$$.&#x20;

The Horizon network consists of a VGG network used to extract features from xiIn and the rotation head to estimate the horizontal vector $$p_{hv} \in R^{1\times 3}$$, where $$r_{dim} =3$$ with the third element fixed at 0.

### Forward-Axis Network

In the third step, the Forward-axis network matches the forward axis of the point cloud with that of the image. It is assumed that the vector entering the image plane is the forward axis of the image coordinate $$e_1 =[1, 0, 0]$$, and that the estimation target is the forward axis of the point cloud pfwd. Output matrix $$T_F$$ rotates $$p_{fwd}$$ to fit $$e_1$$ to match the two forward axes.

The two inputs of the Forward-axis network are an image aligned to the horizon $$x^I_H = x^I_{in} · T_H$$, and a range map $$x^R \in R^{H\times \lambda W \times 4}$$ converted from the point cloud $$x^P_E = T_E · x^P_{in}$$,as shown in

<figure><img src="../../.gitbook/assets/image (108).png" alt=""><figcaption></figcaption></figure>

where $$VF^P_U$$ is the upper bound of the vertical field-of-view (FoV) of the point cloud, $$VF^P_L$$ is the lower bound, and $$H$$ and $$W$$ are the height and width of image $$x^I_{in}$$, respectively. In addition, $$\lambda$$ is the ratio of the horizontal FoV of the point cloud to that of the image, as shown in

<figure><img src="../../.gitbook/assets/image (103).png" alt=""><figcaption></figcaption></figure>

where $$HF^P_U$$ and $$HF^P_L$$ is the upper bound and lower bound of the horizontal FoV of the point cloud, and $$HF^I_U$$ and $$HF^I_L$$ is the upper and lower bound of the image. We assume that $$\lambda \ge 1$$because the horizontal FoV of the point cloud ($$=2\pi$$) is wider than the image.

<figure><img src="../../.gitbook/assets/image (120).png" alt=""><figcaption></figcaption></figure>

We design a method for predicting the correlation score map between two inputs Fig. 4, inspired by the cross-view localization method. The image $$x^I_H$$ and range map $$x^R$$ are processed using independent CNNs. We use the VGG network and add a simple convolution layer before and after the network to reshape the features. Each CNN generates a feature map from an image $$f^I \in R^{H′ \times W^′ \times C}$$ and a range map $$f^R \in R^{H' \times \lambda W' \times C}$$. The network computes the correlation score map $$p_{cs} \in R^{\lambda W'}$$ be

<figure><img src="../../.gitbook/assets/image (79).png" alt=""><figcaption></figcaption></figure>

where $$\%$$ is the modulo operation.

The forward axis of the point cloud can be calculated using the w-index of the highest correlation score as

<figure><img src="../../.gitbook/assets/image (93).png" alt=""><figcaption></figcaption></figure>

The ground truth $$y_{cs} \in R^{λW′}$$ is set to one for the pixel corresponding to the ground truth yrad value and n pixels around it, and all other pixels are set to zero. Binary cross-entropy loss and hard-negative mining are used for the training.

### Gather Network

<figure><img src="../../.gitbook/assets/image (96).png" alt=""><figcaption></figcaption></figure>

The fourth and final step is to gather the previous results and estimate the displacement of the image and point cloud Fig. 5. The two inputs of the Gather network are an image $$x^I_H$$ and depth image $$x^D \in R^{H \times W \times 4}$$ converted from the point cloud $$x^P_{EF}=T_F \cdot T_E \cdot x^P_{in}$$:

<figure><img src="../../.gitbook/assets/image (116).png" alt=""><figcaption></figcaption></figure>

where Kinit denotes the initial calibration matrix. The estimated translation vector pt is used to generate an output transformation matrix TG, which moves the origin of the point cloud coordinates (0,0,0) to pt to match the origin of the image coordinate system.&#x20;

The image $$x^I_H$$ is passed through an encoder-decoder network (CNN1) to predict the pseudo-depth image $$p_{di} \in R^{H \times W \times 1}$$ and depth mask $$p_{dm} \in R^{H \times W \times 1}$$. A pseudo-depth image is an image with depth value for each pixel in the image $$x^I_H$$ . The ground truth is a projection of the ground truth point cloud onto the image plane. A depth mask is a binary image for each pixel in the image $$x^I_H$$ that has a value of 1 if the point cloud is projected, and 0 otherwise. The feature map of the decoder and depth image $$x^D$$ are input into ResNet (CNN2), and the translation vector $$p_t \in R^{1\times 3}$$ is estimated. The loss function for the pseudo-depth image $$p_{di}$$ is the mean squared error, for the depth mask $$p_{dm}$$ is the cross-entropy error, and for $$p_t$$ is the L1-loss.

## Experiments

<figure><img src="../../.gitbook/assets/image (132).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (91).png" alt=""><figcaption></figcaption></figure>
