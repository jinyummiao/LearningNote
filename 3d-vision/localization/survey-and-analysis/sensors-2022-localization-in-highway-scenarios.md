---
description: A Survey of Localization Methods for Autonomous Vehicles in Highway Scenarios
---

# \[Sensors 2022] Localization in highway scenarios

{% embed url="https://doi.org/10.3390/s22010247" %}

## Abstract

The localization problem on highways can be distilled into three main components. The first one consists of inferring on which road the vehicle is currently traveling. The second component consists of estimating the vehicle’s position in its lane. Finally, the third and last one aims at assessing on which lane the vehicle is currently driving.

## Introduction

The localization aspect on highways can be distilled in smaller components, that are **Road Level Localization (RLL)**: The road on which the vehicle travels; **Ego-Lane Level Localization (ELL)**: The position of the vehicle in the lane in terms of lateral and longitudinal position; and **Lane-Level Localization (LLL)**: The position of the host lane within the road (i.e., the lane on which the vehicle travels).

The accuracy of a standard GPS device is within 3 m with a 95% confidence

作者将高速道路上的定位算法分类为RLL、ELL和LLL三类，进行调研。

## Road Level Localization (RLL)

RLL旨在识别出车辆目前所处的道路。一般用Map-Matching方法来进行定位。

### Terminologies

<figure><img src="../../../.gitbook/assets/image (648).png" alt=""><figcaption><p>定义车辆轨迹</p></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (640).png" alt=""><figcaption><p>定义道路</p></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (638).png" alt=""><figcaption><p>定义map-matching算法</p></figcaption></figure>

### Online Map-Matching

Map-Matching包括离线和在线模式。

In online mode, the Map-Matching procedure is performed in a streaming fashion, meaning that for each point $$p_i$$, a Map-Matching is performed. Consequently, the procedure has to be adequate for real-time applications. In contrast, offline Map-Matching waits until the trajectory $$T_r$$ is completed in order to perform the Map-Matching on its entirety. Hence, this procedure is not concerned about the real-time requirements.

现有map-matching算法可分为两类：

<figure><img src="../../../.gitbook/assets/image (621).png" alt=""><figcaption></figcaption></figure>

### Deterministic Model Approach

这一类方法的关键在于如何定义轨迹和道路地图之间的closeness。

#### Geometric Algorithm

该类方法分为point-to-point、point-to-curve、curve-to-curve。The most elementary approach, the so-called point-to-point, matches each position sample to the nearest node in the map. The point-to-curve approach projects each position sample to the geometric-closest road. Lastly, curve-to-curve methods match the vehicle’s trajectory $$T_r$$ to the geometric-closest link in the road-network.

<figure><img src="../../../.gitbook/assets/image (653).png" alt=""><figcaption><p>可见，加入更多信息（比如朝向、路线的邻近性）会帮助效果</p></figcaption></figure>

类算法中最流行的closeness度量是Frechet距离。This distance can be illustrated by the following example: a person is walking on a certain curve, and a dog is walking on another one. We assume that both have free control over their speeds but are not allowed to go backward. In this example, the Fréchet distance of the curves is the minimal length of a leash between the person and the dog that is required to cover the curves from start to finish.

<figure><img src="../../../.gitbook/assets/image (571).png" alt=""><figcaption></figcaption></figure>

f,g是两个参数化的曲线，$$\alpha、\beta$$是从f、g重参数化得到的连续、单调递增的参数（在\[0,1]间）。由于该距离受到两曲线间最大距离的影响，所以轨迹中的outlier会导致错误的估计。

#### Pattern-based Algorithm

The assumption is that given a start and an endpoint, people tend to travel on the same trajectory. In that sense, giving a pair of a start and endpoint, and taking into account historical Map-Matching results, the method will find the most similar trajectories that the vehicle will travel on. Finally, the algorithm will decide on the optimal route based on a scoring function.

The main drawback of the pattern-based algorithms is the sparsity and disparity of the historical data. In that sense, the historical data may not cover all the new queried trajectories, which can lead to false Map-Matching results.

### Probabilistic Model Approach

#### Hidden Markov Model (HMM)

一般采用滑窗来降低HMM应用于定位的时间复杂度，各算法在算法结构和表征上没有太大区别，关键差异在于emission概率和transition概率的定义上。

HMM主要有两个缺点：The first one is the selection bias problem which is a side effect of the HMM when giving more weights to long disconnected segments of road. For instance, it is particularly troublesome in the case where a highway is close to a network of smaller roads: the HMM will give a higher weight to the highway and considerably smaller weights to the road network because of the transitions between the smaller paths and their possible large number, decreasing each other probabilities. As such, the highway may be preferred by the HMM even if the vehicle is closer to the road network. The second one is that these methods are not robust against missing trajectory samples. Indeed, the structure of the transition model in a HMM takes into account the connectivity, physical, and logic between two consecutive sets of route candidates. The discontinuity in position frames will jeopardize the travel possibility between these route candidates.&#x20;

#### Conditional Random Field (CRF)

CRF不需要马尔科夫独立性假设。理论上，CRF能够对多余两个状态之间的高维交互进行建模。Therefore, the CRF share the same inability as the HMMs to take into account contextual information. In addition, a learning procedure is required for the CRF to model the interactions between these states, which makes the CRF easy to utilize but heavy to structure.

#### Weighted Graph Technique

The matching process is performed through a weighted candidate graph. Lou et al. introduced the st-matching algorithm, in which the Weighted Graph Technique process is summarized as three steps: (1) Candidate Preparation: In this step, the candidate graph is initialized. Similar to most Map-Matching techniques, the candidates are selected based on a radius of measurements from the estimated position (GNSS position). (2) Spatial and Temporal Analysis: This step is composed of two components. First, similar to HMMbased method, an observation probability and a transition probability are emitted to each candidate. These two probabilities are inferred from a scoring function that takes into account the distance between the position and the candidate, in addition to the road topology. The second component is a temporal analysis in which the speed of the vehicle is compared with the typical speed constraints on each candidate path. The objective of the spatial and temporal analysis is to weigh edges in the graph. (3) Result Matching: In this last step, the path is inferred based on the constructed weighted graph.

这类算法一般都与Li的方法设计相似，区别主要在于时空分析的scoring function。

#### Particle Filter

该类算法考虑更多的轨迹信息，比如陀螺仪和加速度计数据。一般包含两种方法：线性的和非线性的。

线性的方法：For the linear filters, errors due to the imperfection of the model and sensors are represented by Gaussian white noises and are linearized using first order Taylor approximations. Based on the assumption of additive Gaussian white noises, the estimation of these error states can be obtained with an Extended Kalman Filter, for instance.

非线性的方法不需要线性化，一般用粒子滤波的方法给map-matching提供一个初值。For example, Toledo-Moreo et al. proposed a Lane-level localization Map-Matching where he introduced a Particle Filter to fuse sensors information in order to estimate the vehicle a priori position. In general, the Particle Filter is structured as follows. In the initial phase, $$N_p$$ particles are sampled. These particles represent the different hypothesis of the vehicle’s localization, and they all receive the same weight. For each particle, its associated weight is updated accordingly to its likelihood of existence, as soon as a new observation is received. Afterward, a resampling stage starts: particles with low weights are likely to be erased, and the ones with higher weights are used in a vehicle cinematic model in order to feed particles of the next cycle. 非线性算法间的差异在于粒子加权函数的设计。这些算法的主要缺陷在于他们使用了一个车辆动态模型，这一模型在数据采样率较小时将会失效。

#### Multiple Hypothesis Technique

The Multiple Hypothesis Technique, as the name suggests, holds a set of candidates or hypotheses during Map-Matching. The set of hypotheses is generally initialized based on a simple geometric metric. Afterward, the set of hypotheses keeps evolving as further observations are received. The evolving process consists of two processes, namely, hypothesis branching and hypothesis pruning. A hypothesis is branched or replaced when the vehicle travels the candidate and therefore arrives at a crossroad. The original parent hypothesis is then replaced by new child hypotheses. The new child hypotheses are an extension of the parent hypothesis by taking into account all the directions that the vehicle can take at the crossroad, which guarantees that there will be at least one hypothesis covering the correct candidate in which the vehicle will travel. Another advantage of the method is that some failures are intuitively spotted. If there are no hypotheses, it necessarily implies that a problem has occurred at some points. Hypothesis pruning consists of the elimination of the unrealistic hypothesis. The process is based on a pruning criterion: in the state-of-the-art methods, this pruning criteria differs from one author to another. However, the main idea is to model criteria that allow to keep the most likely hypothesis and simultaneously eliminate the most unlikely hypothesis.

### Conclusion of Road Level Localization

**Uncertainty-Proof** is the ability of the Map-Matching algorithm to take into account inherent uncertainties that come from the raw data;&#x20;

**Matching Break** describes the capability of the Map-Matching algorithm to propose a solution where there is a break in the GNSS data;&#x20;

**Integrity Indicator** is a trust indicator on the validity of the output of the Map-Matching algorithm, which can be relevant for the ambiguous cases;&#x20;

**Run Time** of the frameworks: in order to be used in an autonomous vehicle, the Map-Matching algorithm has to fulfill real-time requirements.

<figure><img src="../../../.gitbook/assets/image (591).png" alt=""><figcaption></figcaption></figure>

## Ego-Lane Level Localization (ELL)

The characteristics of these lane position-detection algorithm systems are distilled as follows:

**Lane-Departure-Warning Systems**: It is essential to accurately estimate the position of the vehicle with respect to the ego-lane marking.&#x20;

**Adaptive Cruise Control**: Measures such as the smoothness of the lane are crucial for this monitoring work.&#x20;

**Lane Keeping or centering**: The aim is to keep or center the vehicle in its host lane. A faultless estimation of the lateral position is required.

**Lane Change Assist**: It is mandatory to know the position of the ego-vehicle in its host lane. The lane change has to be done without any risk of colliding with an obstacle.

车道线检测方法一般分为模型驱动或完全端到端学习的方法。

<figure><img src="../../../.gitbook/assets/image (622).png" alt=""><figcaption></figcaption></figure>

模型驱动的方法将车道线检测方法分解为各个模块，每个模块可以独立地改进和测试。这种方法中每个模块的可解释性较好，模块的中间输出有助于查找系统失败的原因。但是由人工设计的中间输出是不够合适的。基于神经网络的端到端学习方法能够达到显著的准确性，但是存在泛化性差、可解释性差的问题，出现问题后难以找到确切原因。

### Model-Driven Approaches

模型驱动方法一般包含四个步骤，即预处理、车道特征检测、拟合和跟踪，并存在从较高层模块（比如拟合）到较低层模块（比如预处理）的反馈连接。

#### Pre-Processing

he objectives of pre-processing are to enhance features of interest, reduce clutter, and remove misleading artifacts. Thereafter, the cleaned image is used for feature extraction.

预处理方法一般旨在：处理光照相关的影响、丢弃图像中无关或误导性的部分。

#### Feature Extraction

特征提取对于下游的拟合环节非常重要。

一是根据图像的形状、颜色来辨识出车道线。这类基于梯度检测的方法都基于一个共同假设：车道线的外观与路面其他部分的外观易于分辨。

另一类方法是过滤掉不在垂直方法的边缘，这些滤波器称为steerable filters。很多滤波器都被用于检测车道线的片段，滤波器的核和类型需要调整。但是，相机的视角畸变会让这些调整后的核难以适用于整张图像，为了解决这一问题，可以将图像变换到另一个视角，称为inverse-perspective image，也称为bird's-eye view。这BEV视角下，车道线的宽度都是一致的，易于融合多种传感器信息，并且有助于拟合过程。

当使用lidar数据时，可以根据反射率将沥青路和车道线区分开来。

### Fitting Procedure

The main objective of the fitting procedure is to extract high-level representations of the path. This high-level representation is the sine qua non to a higher block of autonomous vehicles like decision-making and control. Thereby, the choice of the type of lane model is crucial.

The lane models can be clustered into three heterogeneous modeling techniques, namely parametric, semi-parametric, and non-parametric:

**Parametric model**: Methods that fall into this category make the strong assumption of a global lane shape (e.g., lines, curves, parabola). These models tend to fail when dealing with non-linear road and lane topologies (merging, splitting, and ending lanes). Indeed, the geometric restrictions imposed by the parametric model does not tolerate such scenarios. Concerning the fitting strategies, several regression techniques have been used (e.g., RANSAC, least-squares optimization, Hough transform, Kalman filter)

**Semi-parametric model**: Contrary to the parametric model, semi-parametric models do not assume a specific global geometry of the road. On the downside, the fitting model can over-fit or have unrealistic path curvature. The lane marking is parametrized by several control points. Different spline models with different control points have been used (e.g., Spline, B-spline, Cubic spline). The appearing complicatedness of these models is in choosing the best control points. Indeed, the number of these points affects the curve complexity. In addition to that, these points should be homogeneously distributed along the curve of the lane marking in order to prevent unrealistic curves.&#x20;

**Non-parametric model**: These models are the less conventional approach. The main needed prerequisite is continuous but not necessary differentiable. This model has more freedom to model the lane marking. Meanwhile, it is more prone to erroneous modeling, leading to unrealistic curves.

<figure><img src="../../../.gitbook/assets/image (594).png" alt=""><figcaption></figcaption></figure>

#### Tracking Procedure

The vast majority of lane marking detection system integrates tracking mechanics that use knowledge from the previous frame to improve the knowledge on the present frame.

使用这一技术有如下目标：improving the accuracy of correct detection, reducing the required computation, and correcting erroneous detections.

一般使用手段有两种：1. 使用前一帧的车道线检测结果；2. 使用跟踪系统来定义当前帧的ROIs。

这类方法基于模型能够体现相邻帧之间的运动、前一帧的车道线检测结果是正确的这两个假设。

### Learning Approach

基于CNN的车道线检测算法：

<figure><img src="../../../.gitbook/assets/image (642).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../../.gitbook/assets/image (649).png" alt=""><figcaption></figcaption></figure>

### Conclusion of Ego-Lane Level Localization

模型驱动的方法将车道线检测问题划分为一系列独立的模块，方法的可解释性强，并且允许对每个独立模块的改进和替换，易于寻找系统失败的原因。但是由于需要预设道路模型，所以泛化性差，在复杂场景表现不好。

学习的方法表现更好，更适合与模型难以定义或未知的场景。但是需要训练数据来离线训练，训练耗时长，且当训练场景和测试场景之间差异大时，难以保证良好表现。

## Lane-Level Localization (LLL)

In the broadest sense, Lane-Level localization is a meaningful concept that can refer to two distinct topics. The first goal is to determine the ego-lane, namely the lane on which the vehicle is currently traveling. Secondly, it may refer to the estimation of the lateral position of the vehicle inside the overall road.

LLL需要额外的信息来获得准确地位姿信息，单纯通过GPS精度不够（\~3m），一般有两种方法：1. 依靠高精度地图；2. 依靠路标。

### Map Aided Approaches

自动驾驶系统常用的三种地图尺度为：

**Macroscale maps** represent the road network with a metric accuracy. These maps are used for route-planning problems and high guidance routines. They provide the user meta information such as speed limitations or the number of lanes present on a given road. The road network is smoothed using clothoid curves, which can give a general intuition of the shape of the road.&#x20;

**Microscale maps** correspond to the most accurate maps. These maps have centimeters accuracy, representing the road network with dense information. Generally, lidars are used to gather maximum information. The fundamental benefit of these maps, which is their great information richness, is also their biggest disadvantage. Indeed, the density of information makes the handling of these maps difficult while trying to isolate points of interest, and keeping them updated is a laborious task.&#x20;

**Mesoscale maps** are a trade-off between the two aforementioned types of map. McMaster and Shea \[113] claimed that a map has to provide enough details about the environment without cluttering up the user with unneeded information. As such, this kind of map has more accurate information compared to macroscale maps while not burdening itself with precise information as done by the microscale maps.

智能交通领域，中等尺度的地图最常用，信息较为准确且不过分稠密，易于保存。

Generally speaking, the vision-based map matching localization is a process that aligns the perceived environment landmarks, such as lane lines, with the stored landmarks in the map.

### Landmark Approaches

In these approaches, relevant road level features are extracted from images. Once these features are extracted, they are fed into a high-level fusion framework that assesses the number of lanes and on which one the vehicle is travelling.

### Conclusion of Lane-Level Localization

In the broadest sense, Lane-Level localization is a meaningful concept that can be related to two different problematics. The two paradigms lead to the same knowledge, that is the position of the vehicle on the road, but differ in the methodology.

The first is the knowledge of the lateral position of the autonomous vehicle with respect to the road. The solutions of this problem yield a lateral position that is a real number, and are usually computed using a map-aided approach. In this paradigm, lanelevel Map-Matching algorithms are used to match the estimated position of an ego-vehicle, which can be estimated using Bayesian filters (e.g., Kalman filter) with the proprioceptive sensors. This estimated position is then matched with a map. Generally speaking, the type of map used for this kind of task is the mesoscale map using a lane-level Map-Matching algorithm. Contrary to the Map-Matching methods presented in Section 2, this kind of algorithms faces more difficulties in ambiguous cases. Typically, for highway scenarios with multi-lanes, strong ambiguities exist as all the lane marking shapes are identical. The second limitation of such a paradigm is in the type of map used. Indeed, these maps are relatively complex to build and cost-intensive, in addition to being difficult to use as they are oftentimes not open-source.

The second paradigm uses a different methodology in order to solve the LLL problematic. The methods that belong to this group of paradigm articulate the knowledge of LLL as a classification problem. To do so, these methods rely on the relevant features that are present in the road scene, especially lane markings and adjacent vehicles. These relevant features are first detected and then fused in high-level fusion frameworks that are essentially based on a graphical probabilistic model, namely, Bayesian Network or Hidden Markov Model. These probabilistic frameworks have the ability to take into consideration uncertainties of the detected relevant features. Contrary to the first paradigm, these methods rely solely on the exteroceptive sensors that are embedded in most of autonomous cars. Furthermore, they do not use expensive maps and thus are more flexible.

## Overall Conclusions

The task of localization is split into three components that are the Road Level Localization (RLL), Ego-Lane Level Localization (ELL) and Lane-Level Localization (LLL).

The Road Level Localization part aims at finding on which road the vehicle is currently traveling on. Techniques to perform such a task are named Map-Matching methods and can be divided into two categories that are the deterministic and probabilistic models. Without surprise, deterministic models offer lower computational demands at the cost of being less accurate than their probabilistic counterparts. Indeed, the probabilistic methods can keep multiple hypothesis or take into account the temporal dependencies of the estimation (e.g., the vehicle cannot switch roads between two timestamps), leading to more solid frameworks.

The second task is called the Ego-Lane Level Localization (ELL), namely the task of localizing oneself relatively to the ego lane. Two main approaches exist to tackle this problem, that are the model approach and the learning approach. In the first one, the estimation is conducted by splitting it into submodules that pre-process the sensors data, extract features, fit them to lane markings and finally track the detections between the frames. This approach allows good failure detection, as each block is simple enough to supervise them. In the learning approach, a neural network is trained on road data to be able to directly extract the lane markings (and thus the robot position relatively to them). The inherent nature of the neural networks allows to better take into account the context of the scene, thus reaching better results than model approaches. However, by this same nature, learning approaches suffer from their little explainability, and require considerable training sets to contain all possible road scenarios.

Finally, the last part named Lane-Level Localization (LLL) consists of finding on which way the vehicle is currently driving. Two options are possible, that are either locating the robot relatively to the overall road or apprehend the problem as a classification exercise to extract on which lane the robot is traveling. The first solution uses maps to aid it in the localization, but suffer from ambiguities in the case where several identical lanes are detected. The second solution chooses to classify each lane and selects the most likely among them. To do so, the methods take advantage of features extraction from the sensors data and from the adjacent vehicles. Furthermore, they have the benefit of not using maps that are costly to produce.
