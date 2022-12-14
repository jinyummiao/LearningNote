---
description: 'OpenSceneVLAD: Appearance Invariant, Open Set Scene Classification'
---

# \[ICRA 2022] OpenSceneVLAD

## Abstract

Scene classification is a well-established area of computer vision research that aims to classify a scene image into pre-defined categories such as playground, beach and airport. Recent work has focused on increasing the variety of pre-defined categories for classification, but so far failed to consider two major challenges: changes in scene appearance due to lighting and open set classification (the ability to classify unknown scene data as not belonging to the trained classes). Our first contribution, SceneVLAD, fuses scene classification and visual place recognition CNNs for appearance invariant scene classification that outperforms state-of-the-art scene classification by a mean F1 score of up to 0.1. Our second contribution, OpenSceneVLAD, extends the first to an open set classification scenario using intra-class splitting to achieve a mean increase in F1 scores of up to 0.06 compared to using state-of-the-art openmax layer.&#x20;

## Contributions

1. A visual localisation dataset we make publicly available covering a 20km traversal of Edinburgh in three different visual conditions with GPS data and labels for 4 scene classes. We also make available labels for 4 scene classes across three traversals of the Oxford RobotCar and Nordland datasets (https: //github.com/WHBSmith).
2. A combined scene classification and visual place recognition CNN ‘SceneVLAD’ trained for appearance invariant scene classification.
3. An investigation into the significance of open set scene classification as a problem and an extension ‘OpenSceneVLAD’ to our second contribution using intra-class splitting specifically for this.

## Method

### Scene Class Labelling

<figure><img src="../../.gitbook/assets/image (662).png" alt=""><figcaption></figcaption></figure>

### SceneVLAD: Appearance Invariant Scene Classification

#### Basic Idea

VPR特征应当可以提升场景分类的外观不变性。

#### Architecture

网络有两个输入。一个输入图像输入在场景分类任务上预训练的Places365或Places1365网络，冻结前16层，得到一个365维或1365维的输出。另一个输入图像输入NetVLAD，但是用一个1x1卷积层来代替PCA层，得到256维输出。将两个输出拼接起来，输入全连接层，得到4分类结果。SceneVLAD利用NetVLAD的外观不变性图像特征来学习场景分类。

<figure><img src="../../.gitbook/assets/image (633).png" alt=""><figcaption></figcaption></figure>

### OpenSceneVLAD： Open Set, Appearance Invariant Scene Classification

#### Basic Idea

Intra-class splitting was selected to extend SceneVLAD for open set scene classification (OpenSceneVLAD) because we hypothesised it could leverage assumptions about likely open set scene images from training images, such as position of the ground plane and orientation of ambient scenery for improved OSC performance.

#### Identify Atypical Class Examples

首先在N个类别上对SceneVLAD进行closed set classification的训练，场景图像为$$x_i$$​，对应的closed set标签为$$y_{i,cs}$$​。令每个类别中30%的图像无法被正确分类，或者分类的可信度很低，被识别为非典型样本。

#### Generate Open Set Labels

新生成一些标签$$y_{i,os}$$。对于每张图像$$x_i$$，如果被认为是非典型样本，则$$y_{i,os}=N+1$$​，否则$$y_{i,os}=y_{i,cs}$$​。对错误分类或分类不可信图像进行重新标注使得只用closed set images就可以得到open set。

#### Create OpenSceneVLAD and Re-train

<figure><img src="../../.gitbook/assets/image (601).png" alt=""><figcaption></figcaption></figure>

Two separate softmax layers are used as network outputs (Figure 4), one with N outputs trained for closed set regularization using cross-entropy loss $$L_{cs}$$ (Equation 3), the other with N + 1 outputs trained for OSC also using cross-entropy loss $$L_{os}$$ (Equation 2). OpenSceneVLAD is then trained using both losses with the corresponding labels of each image. Closed set regularization helps maintain a high closed-set accuracy by forcing the atypical samples to be correctly classified to their original classes.

#### Test

At test time the closed set regularization is removed and output from the remaining open set layer is used for open set classification.

#### Loss Functions

intra-class splitting是一个联合优化问题，包含两个loss，分别对应于open set层和closed set层：

<figure><img src="../../.gitbook/assets/image (933).png" alt=""><figcaption></figcaption></figure>

令B为minibatch size。$$1_{y_i\in y^{(n)}}$$​是指示函数，如果样本$$x_i$$​的标签$$y_i$$​属于类别$$y^{(n)}$$，则返回1，否则返回0。​$$L_{os}$$​是一个简单的$$N_{os}=N+1$$类交叉熵：

<figure><img src="../../.gitbook/assets/image (937).png" alt=""><figcaption></figcaption></figure>

$$L_{cs}$$是一个$$N_{cs}=N$$类交叉熵：

<figure><img src="../../.gitbook/assets/image (987).png" alt=""><figcaption></figcaption></figure>

## Experiments

<figure><img src="../../.gitbook/assets/image (956).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (953).png" alt=""><figcaption></figcaption></figure>
