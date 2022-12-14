---
description: >-
  Deep Learning based Vehicle Position and Orientation Estimation  via Inverse
  Perspective Mapping Image
---

# \[IV 2019] BEV-IPM

## Abstract

In this paper, we present a method for estimating a position, size, and orientation using a single monocular image. The proposed method makes use of an inverse perspective mapping to effectively estimate the distance from the image. The proposed method consists of two stages: 1) cancel the pitch and roll motion of the camera using inertial measurement unit and project the corrected front view image onto the bird’s eye view using inverse perspective mapping. 2) detect the position, size, and orientation of the vehicle using a convolutional neural network. The camera motion cancellation process makes vanishing point to be located at the same point regardless of the ego vehicle attitude change. Through this process, the projected bird’s eye view image can be parallel and linear to the x-y plane of the vehicle coordinate system. The convolutional neural network predicts not only the position and size but also the orientation of the vehicle for the 3D localization. The predicted oriented bounding box from the bird’s eye view image is converted in the meter unit by the inverse projection matrix. Despite the conceptually simple architecture, the proposed method achieves promising performance compared to other image based approaches.

## Introduction

<figure><img src="../../.gitbook/assets/image (119).png" alt=""><figcaption></figcaption></figure>

In this paper, we proposed a conceptually simple and effective method that estimates the position, size, and orientation of the vehicle in a meter unit on the BEV image. The main idea is that distance information can be restored by projecting the front image onto the BEV image if the road plane is parallel to the vehicle coordinate system. As illustrated in Fig. 2, the proposed method consists of two stages: the motion cancellation & BEV projection stage and the vehicle detection & distance estimation stage. The first stage utilizes the extrinsic parameters of the camera and inertial measurement unit (IMU) information. This stage corrects the motion change of the ego vehicle so that the front view image can be projected onto the BEV image corresponding to the vehicle coordinate system. The vehicle detection network that has the pipeline of YOLOv3 takes the BEV image as input and predicts the oriented bounding box. The proposed oriented bounding box is designed for the 3D localization on the BEV image coordinate system and consists of x and y position, width and height, and orientation.

## Position and Orientation Estimation

The proposed architecture takes front view image as input and cancels the camera pitch and roll motion using IMU. Then projects the corrected front view image onto the BEV image using the IPM. The one-stage object detector based vehicle detection network takes the BEV image and predicts the oriented bounding box composed of the position, size, and orientation. Finally, convert the predicted detection results of pixel units in the BEV image coordinate system into the distance of m units in the vehicle coordinate system.

### Motion Cancellation and BEV Projection

#### Motion cancellation using IMU

The motion of the camera can be corrected by the extrinsic parameter of the camera and the rotation matrix of the IMU. The extrinsic parameter consists of the translation and rotation of the camera, and the rotation matrix consists of the pitch and roll angle. The rotation matrix obtained in the camera coordinate system is used to correct the motion of the camera as (1):

<figure><img src="../../.gitbook/assets/image (121).png" alt=""><figcaption></figcaption></figure>

Here, only the roll and pitch angles are used since the yaw angle indicates the heading of the ego vehicle and it should not be corrected. The Fig. 3 shows the example that the surrounding vehicle moves upward due to the ego vehicle pitch motion in the raw image. The movement caused by the pitch angle is corrected by the camera motion cancellation matrix. After correcting the roll and pitch motion, the vanishing point of the image can be located on the same point. Also, the specific pixel on the road always represents the same distance. Consequently, the distance can be estimated from the point of the vehicle that is in contact with the road.

<figure><img src="../../.gitbook/assets/image (76).png" alt=""><figcaption></figcaption></figure>

#### BEV projection

The proposed method detects the bottom box of the vehicle on the BEV image as illustrated in Fig. 4. The front view image can be projected into the BEV image by inverse perspective mapping (IPM).&#x20;

<figure><img src="../../.gitbook/assets/image (109).png" alt=""><figcaption></figcaption></figure>

Several assumptions are required to project the road surface in the front view image into the BEV image using IPM. First, the surface where the ego vehicle and the surrounding vehicle drive is located must be a planar surface since IPM is a plane to plane transformation. Second, the mounting position of the camera must be stationary. If the mounting position of the camera changes, the extrinsic parameters of the (1) also have to be changed. Third, the vehicle to be detected must be attached to the ground plane because only the points on the ground have distance information. The normal driving environment satisfies the above assumptions and the road of the front view image can be projected onto the BEV image using IPM.

As shown in Fig. 4, the relationship between two corresponding point Q(u,v) on front view image and point P(x,y) on BEV image can be defined by homography as (2):

<figure><img src="../../.gitbook/assets/image (115).png" alt=""><figcaption></figcaption></figure>

The homography can be obtained by 4 corresponding points of the front view and the BEV image and calculated experimentally using OpenCV. The homography is calculated only once since the perspective change by the vehicle motion is corrected by the camera motion cancellation.

### Vehicle Detection and Distance Estimation

#### Vehicle detection network

The vehicle detection network is designed based on the one stage detector YOLOv3 pipeline and modified to predict the orientation of the vehicle as Fig. 5. For the fast operation speed, we fix the image resolution to 320 by 320 and use the five anchors calculated by the k-mean clustering algorithm. Since most of the vehicle have medium size on the BEV image, detection layer on the middle uses three anchors while other layers have only one anchor each. Therefore, the big vehicle such as truck or bus is detected on the first detection layer and the small vehicle is detected on the last detection layer. In addition, the ground truth label is annotated assuming that the vehicle is heading for the same direction, the 5 anchors have a similar ratio of 1.8:1. The anchors with a similar ratio help the network to detect the size accurately.

<figure><img src="../../.gitbook/assets/image (136).png" alt=""><figcaption></figcaption></figure>

In the detection layer, the 1×1 convolutional filter predicts the probability of the object and class, offset of x, y, width, height, and orientation. The oriented bounding box is calculated using predicted offsets from the anchor as illustrated in Fig. 5. The orientation angle is predicted through hyperbolic tangent function to force the orientation output between −pi/2 and pi/2. The multi-task loss function is adapted from YOLOv3, but the orientation loss term (3) is added to measures the errors between the ground truth and the predicted orientation.

<figure><img src="../../.gitbook/assets/image (118).png" alt=""><figcaption></figcaption></figure>

#### Distance estimation

Since the detection network predicts the bottom box of the vehicle in contact with the road, the height of the detection result is zero. Therefore, the points on the road of the BEV image and vehicle coordinate system are defined by a one-to-one correspondence as (4).

<figure><img src="../../.gitbook/assets/image (77).png" alt=""><figcaption></figcaption></figure>

The matrix (4) is the product of the homography, motion cancellation matrix, intrinsic camera matrix, and transformation matrix of the camera. The point (x,y) on the BEV image coordinate system is converted into the point (X,Y) on the vehicle coordinate system using (4) as illustrated in Fig. 4.

## Experiments and Discussion

The proposed method is trained on the KITTI object detection dataset and evaluated on the KITTI raw dataset.

<figure><img src="../../.gitbook/assets/image (84).png" alt=""><figcaption></figcaption></figure>

### Distance Estimation Evaluation

We evaluate the distance estimation performance using the root mean square error (RMSE) and mean absolute percentage error (MAPE) as the metrics. Fig. 7 shows the estimated position and the error of the single target vehicle. The target vehicle for calculating RMSE for datasets 0015 is a forward vehicle of the same lane, and the target vehicle for datasets 0028 is a vehicle facing the opposite lane. The RMSE and MAPE are calculated between the box centers by the distance of the ground truth as (5) in the longitudinal and lateral direction.

<figure><img src="../../.gitbook/assets/image (89).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (81).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (104).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (63).png" alt=""><figcaption></figcaption></figure>

### BEV Object Detection Evaluation

The $$AP_{BEV}$$ is calculated using an oriented intersection over union (IoU) considering the direction of the vehicle.

<figure><img src="../../.gitbook/assets/image (88).png" alt=""><figcaption></figcaption></figure>
