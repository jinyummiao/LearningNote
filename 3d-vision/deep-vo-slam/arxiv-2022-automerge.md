---
description: >-
  AutoMerge: A Framework for Map Assembling and Smoothing in City-scale
  Environments
---

# \[arxiv 2022] AutoMerge

## Abstract

We present AutoMerge, a LiDAR data processing framework for assembling a large number of map segments into a complete map. AutoMerge utilizes multi-perspective fusion and adaptive loop closure detection for accurate data associations, and it uses incremental merging to assemble large maps from individual trajectory segments given in random order and with no initial estimations. Furthermore, after assembling the segments, AutoMerge performs fine matching and pose-graph optimization to globally smooth the merged map. We demonstrate AutoMerge on both city-scale merging (120km) and campus-scale repeated merging (4.5km×8). The experiments show that AutoMerge (i) surpasses the second- and third- best methods by 14% and 24% recall in segment retrieval, (ii) achieves comparable 3D mapping accuracy for 120 km large-scale map assembly, (iii) and it is robust to temporally-spaced revisits. To the best of our knowledge, AutoMerge is the first mapping approach that can merge hundreds of kilometers of individual segments without the aid of GPS.

## System Overview

<figure><img src="../../.gitbook/assets/image (921).png" alt=""><figcaption></figcaption></figure>

AutoMerge包含三个模块：1）fusion-enhanced place descriptor extraction, 2) an adaptive data-association mechanism to provide high accuracy and recall for segment-wise place retrievals, and 3) a partially decentralized system to provide centralized map merging and single agent self-localization in the world frame.

**Fusion-enhance Descriptor：**每个agent独立运行LOAM，构建子地图，并且提取自适应的描述子，该描述子有如下优势：1）it is translation-invariant due to the local translation-equivalent property of 3D point-clouds \[Adafusion: Visual-lidar fusion with adaptive weights for place recognition], 2) it is orientation-invariant due to the rotation-equivalent property of spherical harmonics \[Fast sequence-matching enhanced viewpoint-invariant 3-d place recognition], and 3) it is light-weight compared to the original raw sub-maps. 因此，每个agent向服务器提供具有视角不变性的场景描述子和ego-motion。

**Adaptive Loop Closure Detection：**混合回环检测利用sequence matching在长段的overlaps中获得连续的、正确的回环，利用基于RANSAC的单帧检测来获得局部的overlaps。By analysing the feature correlation between segments, we can balance the place retrievals from sequence-/single- frame matching to provide accurate retrievals for offline/online LCD.

**Incremental Merging：**AutoMerge用每个agent的描述子和ego-motion来提升回环检测的表现。在rough merge阶段，AutoMerge选择一个segment作为根节点，根据keyframe的关联和基于ICP的局部地图配准来将其他segments变换到已存的节点上。在refine merge阶段，通过对每对overlaps进行点云配准来获得准确的变换关系，并用因子图去优化最后合并的结果。在合并后，每个agent可以根据GO和自身的里程计估计状态来优化自己的定位。

In offline map merging, AutoMerge can detect all of the potential matches between all of the segments, and directly merge sub-maps into a global map along high confidence overlaps. In online map merging, the AutoMerge server can incrementally accumulate the key-frames and corresponding poses, and merges segments with strong connections.

## Fusion-enhanced Descriptor Extraction

<figure><img src="../../.gitbook/assets/image (979).png" alt=""><figcaption></figcaption></figure>

### Multi-perspectives Feature Extraction

#### Point-based Feature Extraction

用PointNetVLAD来提取全局描述子，给定一个局部dense map，在80x80m的bounding box中检索点集$$P=\{p_1,...,p_N|p_n \in R^3\}$$，并进行预处理。然后将P输入PointNet来提取局部特征$$F_p=\{f_1,...,f_N\}$$，用NetVLAD层来聚合局部特征，用全连接层得到降维后的全局描述子$$V_{point}$$。

#### Projection-based Feature Extraction

利用局部稠密地图，在50m范围内检索点，将它们投影到全景图中，得到的球形投影SP输入四层spherical convolution，得到局部特征$$F_s$$，再经过NetVLAD层，得到全局描述子$$V_{sphere}$$​。

### Attention Fusion

Our attention fusion module consists of two self-attention modules providing contextual information for $$V_{point}$$ and $$V_{sphere}$$ individually, and a cross attention module which aims to reweigh the importance of channels within the concatenation of $$V_{point}$$ and $$V_{sphere}$$.

<figure><img src="../../.gitbook/assets/image (926).png" alt=""><figcaption></figcaption></figure>

#### Self-attention Feature Enhancement

<figure><img src="../../.gitbook/assets/image (907).png" alt=""><figcaption></figcaption></figure>

#### Cross-attention Feature Reweighing

将两个全局描述子拼接起来，得到$$V_{cat}=[V_{point}, V_{sphere}]$$​

<figure><img src="../../.gitbook/assets/image (916).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (976).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (902).png" alt=""><figcaption></figcaption></figure>

### Learning Metrics

用lazy quadruplet loss metric进行训练，每个训练的tuple包含$$[S_a,\{S_{pos}\},\{S_{neg}\},\{S_{neg*}\}]$$，其中$$S_{neg*}$$​是与$${S_{neg}}$$距离严格大于50m的样本。Lazy quadruplet loss为：

<figure><img src="../../.gitbook/assets/image (935).png" alt=""><figcaption></figcaption></figure>

## Adaptive Loop Closure Detection

<figure><img src="../../.gitbook/assets/image (909).png" alt=""><figcaption></figcaption></figure>

### Adaptive Candidates Association

为了在多个segments中寻找可能的回环，每两个segments被构成一对。每个segment $$T_i$$​包含对应的点云子地图$$T_i=\{T^k_i\}$$，子地图是根据固定距离和对应的位姿划分的，这些都是由里程计获得的。这些子地图用fusion-enhanced descriptor进行描述，得到$$f_i=\{f^k_i\}$$。不同segments的子地图之间的相似度用$$D=d(f_i,f_j)\in R^{N_i\times N_j}$$表示，其中d()是余弦距离，$$N_i$$和$$N_j$$分别是$$T_i$$​和$$T_j$$​中的submap数量。

LCD方法的工作对象是loop candidates $$C=\{(k_i,k_j)\}$$​，其中$$k_i,k_j$$​是$$T_i,T_j$$​中submap的索引。在矩阵D上进行sequence matching，获得$$C_{seq}$$。但是如图4所示，这里还存着很多外点。因此，作者先根据特征距离用K-means将可能的匹配聚类为区域$$C_{seq_i}$$。对每个区域，用RANSAC的方法，从$$C_{seq_i}$$中挑选correspondences。在每次迭代中，对n个样本$$(k_i,k_j)$$​校验下式：

<figure><img src="../../.gitbook/assets/image (942).png" alt=""><figcaption></figcaption></figure>

其中边是在任意两个样本$$(k^1_i,k^1_j)$$​和$$(k^2_i,k^2_j)$$之间形成的。​

## Incremental Merging

### Multi-agent Clustering

首先将incremental merging问题视为经典的spectral clustering问题。

假设有实时运行的多个agents $$V=\{v_1,...,v_n\}$$，将$$v_i$$和$$v_j$$​之间的内部关联记为$$w_{ij}$$。定义一个加权图G=(V, E)，E表示边，满足$$w_{ij}=w_{ji}$$。定义V的子集$$A_i$$​，其满足：

<figure><img src="../../.gitbook/assets/image (958).png" alt=""><figcaption></figcaption></figure>

$$\overline{A_i}$$是$$A_i$$​的补集，$$W(A_i,\overline{A_i})$$是一个weighted adjacency matrix：

<figure><img src="../../.gitbook/assets/image (977).png" alt=""><figcaption></figcaption></figure>

从回环检测的角度，内部关联$$w_{ij}$$​是基于overlap长度和场景识别质量的，因此，我们定义内部关联为：

<figure><img src="../../.gitbook/assets/image (971).png" alt=""><figcaption></figcaption></figure>

其中$$F_i$$​为$$v_i$$提取的overlap场景特征，$$L_{ij}$$​为overlap区域的长度，$$C_w$$​是超参，$$\epsilon=1e-4$$​.

作者还定义了degree matrix D，其中$$d_{ii}=\sum^n_{j=1}w_{ij}$$度量了agent $$v_i$$与其他agent之间的关联。

根据spectral clustering，增量合并任务可以定义为一个mincut问题：

<figure><img src="../../.gitbook/assets/image (940).png" alt=""><figcaption></figcaption></figure>

上述mincut算法的主要局限在于它会简单地将单个agent和其他agents分离开，这是我们不期望的。为了获得更大的集合，作者使用了Ncut的目标函数：

<figure><img src="../../.gitbook/assets/image (970).png" alt=""><figcaption></figcaption></figure>

其中$$vol(A_i)$$​是集合$$A_i$$​中内部关联的度量。From the power consumption perspective, incremental clustering is trying to find the best segment option with minimum penalty to divide the original agents into different consistent sub-groups.

<figure><img src="../../.gitbook/assets/image (978).png" alt=""><figcaption></figcaption></figure>

给定agent列表$$V=\{v_1,...,v_n\}$$，根据公式9计算相似度矩阵、degree matrix和相应的Laplace matrix L。​特征值$$\lambda_i$$​可以指示出聚类状态。理论上将，如果存在k个不同集合$$\{A_i,...,A_k\}$$​，它们之间不存在关联$$W(A_i,A_j)=0,i\ne j$$​，等于0的特征值数量也为k。

In the map merging problem, partial overlaps between different sub-groups may exist, thus we set a control threshold $$\lambda_{max} \le \theta$$ to estimate the best subgroups size. Based on the first k-dimension of eigenvectors U, we can construct a key matrix $$K^{n \times k}$$, and cluster to k classes though k-means. Through the above operation, AutoMerge can cluster agents into k sub-groups.

### Incremental Merging

<figure><img src="../../.gitbook/assets/image (897).png" alt=""><figcaption></figcaption></figure>

In the AutoMerge server system, each agent has its own identity (Agent-ID). As shown in Fig. 5, based on multi-agent clustering, the received agent lists$$V=\{v_1,...,v_{10}\}$$ can be clustered into 2 sub-groups $$\{A_1,A_2\}$$based on their existing connections. When new observation for agent $$v_i$$ received, AutoMerge will automatically estimate corresponding weightings $$w_{ij}$$ between $$v_i$$ and $$v_j$$; when new overlaps are observed for existing agents, previous weak connection $$w_{2,8}$$ is further enhanced. And based on updated global graphs, AutoMerge will merge original sub-groups into a joint graph. During the merging procedure, all of the different sub-graph are optimized through self-contained individual graph optimization.

## Dataset and Criteria

<figure><img src="../../.gitbook/assets/image (969).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (919).png" alt=""><figcaption></figcaption></figure>

AutoMerge generates a dense map with traditional LiDAR odometry.  For each place, sub-maps are constructed by accumulating LiDAR scans into dense observations and keeping a distance (40m) to the vehicle’s latest position.

We extract sub-maps every 5m with a fixed 50m radius.

<figure><img src="../../.gitbook/assets/image (900).png" alt=""><figcaption></figcaption></figure>

These maps can only be extracted when the relative distance between the vehicle’s central point and the keyframe’s is 100m away. In this manner, the geometric structures for the same areas will be very similar in both under forward and reverse traversal directions.

In all of the above datasets, we count the retrieval as successful if the detected candidates are 10m apart from the ground-truth positions.

## Results

### Place Recognition Results

<figure><img src="../../.gitbook/assets/image (906).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (963).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (990).png" alt=""><figcaption><p>Localization results for different viewpoints on different datasets. For each dataset, we pick one segment from the same domain and generate test/reference queries with different yaw angles [15, 30]◦ and translational displacement [1, 2, 3, 4]m, and then analyze the average recall for top-20 retrievals.</p></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (941).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (920).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (955).png" alt=""><figcaption></figcaption></figure>

### Map Merging

<figure><img src="../../.gitbook/assets/image (947).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (903).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (985).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (899).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (923).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (914).png" alt=""><figcaption></figcaption></figure>
