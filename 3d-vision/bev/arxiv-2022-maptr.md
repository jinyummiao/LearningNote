---
description: >-
  MapTR: Structured Modeling and Learning for Online Vectorized HD Map
  Construction
---

# \[arxiv 2022] MapTR

{% embed url="https://github.com/hustvl/MapTR/" %}

## Abstract

We present MapTR, a structured end-to-end framework for efficient online vectorized HD map construction. We propose a unified permutation-based modeling approach, i.e., modeling map element as a point set with a group of equivalent permutations, which avoids the definition ambiguity of map element and eases learning. We adopt a hierarchical query embedding scheme to flexibly encode structured map information and perform hierarchical bipartite matching for map element learning. MapTR achieves the best performance and efficiency among existing vectorized map construction approaches on nuScenes dataset. In particular, MapTR-nano runs at real-time inference speed (25.1 FPS) on RTX 3090, 8× faster than the existing state-of-the-art camera-based method while achieving 3.3 higher mAP. MapTR-tiny significantly outperforms the existing stateof-the-art multi-modality method by 13.5 mAP while being faster.&#x20;

## Introduction

In this work, we present Map TRansformer (MapTR), for efficient online vectorized HD map construction. We propose a unified permutation-based modeling approach for various kinds of map elements (both closed and open shapes), i.e., modeling map element as a point set with a group of equivalent permutations. The point set determines the position of the map element. And the permutation group includes all the possible organization sequences of the point set corresponding to the same geometrical shape, avoiding the definition ambiguity of map element.

Based on permutation-based modeling, MapTR builds a structured framework for map learning. MapTR treats online vectorized HD map construction as a parallel regression problem. Hierarchical query embeddings are adopted to flexibly encode instance-level and point-level information. All instances and all points of instance are simultaneously predicted with a unified Transformer structure. And MapTR formulates the training pipeline as a hierarchical set prediction task. We perform hierarchical bipartite matching to assign instances and points in turn.

Our contributions can be summarized as follows:&#x20;

* We propose a unified permutation-based modeling approach for map elements, i.e., modeling map element as a point set with a group of equivalent permutations, which avoids the definition ambiguity of map element and eases learning.
* Based on the novel modeling, we present MapTR, a structured end-to-end framework for efficient online vectorized HD map construction. We introduce a hierarchical query embedding scheme to flexibly encode instance-level and point-level information and perform hierarchical bipartite matching for map element learning.
* MapTR is the first real-time and SOTA vectorized HD map construction approach with stable and robust performance in complex and various driving scenes. It is of great application value in autonomous driving.

## &#x20;MapTR

### Permutation-based Modeling

For structured modeling, MapTR geometrically abstracts map elements as closed shape (like pedestrian crossing) and open shape (like lane dividers). Through sampling points sequentially along the shape boundary, closed-shape element is discretized into polygon while open-shape element is discretized into polyline.

<figure><img src="../../.gitbook/assets/image (775).png" alt=""><figcaption></figcaption></figure>

Preliminarily, both polygon and polyline can be represented as an ordered point set $$V^F=\{v_0,v_1,...,v_{N_v-1}\}$$(see Fig. 3 (Vanilla)). $$N_v$$ denotes the number of points. However, the permutation of the point set is not explicitly defined and not unique. MapTR models each map element with $$\mathcal{V}=(V,\Gamma)$$, $$V=\{v_i\}^{N_v-1}_{i=0}$$ denotes the point set of the map element ($$N_v$$ is the number of points). $$\Gamma=\{\gamma_k\}$$ denotes a group of equivalent permutations of the point set V , covering all the possible organization sequences.

Specifically, for polyline element (see Fig. 3 (left)), $$\Gamma$$ includes 2 kinds of equivalent permutations, i.e.,

<figure><img src="../../.gitbook/assets/image (787).png" alt=""><figcaption></figcaption></figure>

For polygon element (see Fig. 3 (right)), $$\Gamma$$ includes $$2\times N_v$$ kinds of equivalent permutations, i.e.,

<figure><img src="../../.gitbook/assets/image (735).png" alt=""><figcaption></figcaption></figure>

By introducing the conception of equivalent permutations, MapTR models map elements in a unified manner and avoids the ambiguity problem. Based on such modeling, MapTR further introduces hierarchical bipartite matching for map element learning, and adopts a structured encoder-decoder Transformer architecture to efficiently predict map elements.

### Hierarchical Matching

MapTR parallelly infers a fixed-size set of N map elements in a single pass. N is set to be larger than the typical number of map elements in a scene. Let’s denote the set of N predicted map elements by $$\hat{Y}=\{\hat{y}_i\}^{N-1}_{i=0}$$. The set of ground-truth (GT) map elements is padded with $$\varnothing$$ (no object) to form a set with size N , denoted by $${Y}=\{{y}_i\}^{N-1}_{i=0},y_i=(c_i,V_i,\Gamma_i)$$, where $$c_i,V_i,\Gamma_i$$ are respectively the target class label, point set and permutation group of GT map element $$y_i$$, $$\hat{y}_i=(\hat{p}_i,\hat{V}_i)$$where $$\hat{y}_i$$ and $$\hat{V}_i$$ are respectively the predicted classification score and predicted point set. To achieve structured map element modeling and learning, MapTR introduces hierarchical bipartite matching, i.e., performing instance-level matching and point-level matching in order.

**Instance-level Matching.** First, we need to find an optimal instance-level label assignment $$\hat{\pi}$$ between predicted map elements $$\{\hat{y}_i\}$$ and GT map elements $$\{y_i\}$$. $$\hat{\pi}$$ is a permutation of N elements $$(\hat{\pi} \in \Pi_N)$$ with the lowest instance-level matching cost:

<figure><img src="../../.gitbook/assets/image (757).png" alt=""><figcaption></figcaption></figure>

$$L_{ins\_match}(\hat{y}_{\pi(i)},y_i)$$ is a pair-wise matching cost between prediction $$\hat{y}_{\pi(i)}$$ and GT $$y_i$$, which considers both the class label of map element and the position of point set:

<figure><img src="../../.gitbook/assets/image (722).png" alt=""><figcaption></figcaption></figure>

$$L_{Focal}(\hat{p}_{\pi(i)}, c_i)$$ is the class matching cost term, defined as the Focal Loss between predicted classification score $$\hat{p}_{\pi(i)}$$ and target class label $$c_i$$. $$L_{position}(\hat{V}_{\pi(i)},V_i)$$ is the position matching cost term, which reflects the position correlation between the predicted point set $$\hat{V}_{\pi(i)}$$ and the GT point set $$V_i$$. Hungarian algorithm is utilized to find the optimal instance-level assignment $$\hat{\pi}$$ following DETR.

**Point-level Matching.** After instance-level matching, each predicted map element $$\hat{y}_{\hat{\pi}(i)}$$ is assigned with a GT map element $$y_i$$. Then for each predicted instance assigned with positive labels $$(c_i \neq \varnothing)$$, we perform point-level matching to find an optimal point2point assignment $$\hat{\gamma} \in \Gamma$$ between predicted point set $$\hat{V}_{\hat{\pi}(i)}$$ and GT point set $$V_i$$. $$\hat{\gamma}$$ is selected among the predefined permutation group $$\Gamma$$ and with the lowest point-level matching cost:

<figure><img src="../../.gitbook/assets/image (743).png" alt=""><figcaption></figcaption></figure>

$$D_{Manhattan}(\hat{v}_j,v_{\gamma(j)})$$ is the Manhattan distance between the j-th point of the predicted point set $$\hat{V}$$ and the $$\gamma(j)$$-th point of the GT point set $$V$$.

### End-to-end Training

MapTR is trained based on the optimal instance-level and point-level assignment ($$\hat{\pi}$$ and $$\{\hat{\gamma}_i\}$$). The loss function is mainly composed of three parts, classification loss, point2point loss and direction loss, i.e.,

<figure><img src="../../.gitbook/assets/image (750).png" alt=""><figcaption></figcaption></figure>

where $$\lambda,\alpha,\beta$$ are the weights for balancing different loss terms.

**Classification Loss.** According to the instance-level optimal matching result $$\hat{\pi}$$, each predicted map element is assigned with a class label (or ’no object’ $$\varnothing$$). The classification loss is a Focal Loss term formulated as:

<figure><img src="../../.gitbook/assets/image (769).png" alt=""><figcaption></figcaption></figure>

**Point2point Loss.** Point2point loss aims at restricting the position of each predicted point. For each GT instance with index i, according to the point-level optimal matching result $$\hat{\gamma}_i$$, each predicted point $$\hat{v}_{\hat{\pi}(i),j}$$ is assigned with a GT point $$v_{i,\hat{\gamma}_i(j)}$$. The point2point loss is defined as the Manhattan distance computed between each assigned point pair:

<figure><img src="../../.gitbook/assets/image (786).png" alt=""><figcaption></figcaption></figure>

**Edge Direction Loss.** Point2point loss only restricts the node point of polyline and polygon, not considering the edge (the connecting line between adjacent points). For accurately representing map elements, the direction of the edge is important. Thus, we further design edge direction loss to restrict the geometrical shape at the higher edge level. Specifically, we consider the cosine similarity of the paired predicted edge $$\hat{e}_{\hat{\pi}(i),j}$$ and GT edge $$e_{i,\hat{\gamma}_i(j)}$$:

<figure><img src="../../.gitbook/assets/image (720).png" alt=""><figcaption></figcaption></figure>

### Architecture

<figure><img src="../../.gitbook/assets/image (718).png" alt=""><figcaption></figcaption></figure>

**Map Encoder.** The encoder of MapTR extracts features from original sensor data and transforms sensor features into a unified feature representation, i.e., BEV representation. For camera-based MapTR, given multi-view images $$\mathcal{I}=\{I_1,...,I_K\}$$, we leverage a conventional backbone to generate multi-view feature maps $$\mathcal{F}=\{F_1,...,F_K\}$$. Then 2D image features $$\mathcal{F}$$ are transformed to BEV features $$\mathcal{B}\in R^{H \times W \times C}$$ . By default, we adopt GKT as the basic 2D-toBEV transformation module, considering its easy-to-deploy property and high efficiency.

**Map Decoder.** We adopt a hierarchical query embedding scheme to explicitly encode each map element. Specifically, we define a set of instance-level queries $$\{q^{(ins)}_i\}^{N-1}_{i=0}$$ and a set of point-level queries $$\{q^{(pt)}_j\}^{N_v-1}_{j=0}$$ shared by all instances. Each map element (with index $$i$$) corresponds to a set of hierarchical queries $$\{q^{(pt)}_{ij}\}^{N_v-1}_{j=0}$$ . The hierarchical query of j-th point of i-th map element is formulated as:

<figure><img src="../../.gitbook/assets/image (754).png" alt=""><figcaption></figcaption></figure>

The map decoder contains several cascaded decoder layers which update the hierarchical queries iteratively. In each decoder layer, we adopt MHSA to make hierarchical queries exchange information with each other (both inter-instance and intra-instance). We then adopt Deformable Attention to make hierarchical queries interact with BEV features, inspired by BEVFormer. Each query $$q^{(hie)}_{ij}$$ predicts the 2-dimension normalized BEV coordinate $$(x_{ij}, y_{ij})$$ of the reference point $$p_{ij}$$. We then sample BEV features around the reference points and update queries.

Map elements are usually with irregular shapes and require long-range context. Each map element corresponds to a set of reference points $$\{p_{ij}\}^{N_v-1}_{j=0}$$ with flexible and dynamic distribution. The reference points $$\{p_{ij}\}^{N_v-1}_{j=0}$$ can adapt to the arbitrary shape of map element and capture informative context for map element learning.

The prediction head of MapTR is simple, consisting of a classification branch and a point regression branch. The classification branch predicts instance class score. The point regression branch predicts the positions of the point sets $$\hat{V}$$ . For each map element, it outputs a $$2N_v$$-dimension vector, which represents normalized BEV coordinates of the $$N_v$$ points.

## Experiments

**Implementation Details.** MapTR is trained with 8 NVIDIA GeForce RTX 3090 GPUs. We adopt AdamW optimizer and cosine annealing schedule. The initial learning rate is set to 6e−4. For MapTR-tiny, we adopt ResNet50 as the backbone. We set the size of each BEV grid to 0.3m and stack 6 transformer decoder layers. We train MapTR-tiny with a total batch size of 32 (containig 6 view images). All ablation studies are based on MapTR-tiny trained with 24 epochs. MapTR-nano is designed for real-time application. We adopts ResNet18 as the backbone. We set the size of each BEV grid to 0.75m and stack 2 transformer decoder layers. And we train MapTR-nano with a total batch size of 192. As for hyper-parameters of loss weight, $$\lambda=2,\alpha=5,\beta=5e-3$$.&#x20;

<figure><img src="../../.gitbook/assets/image (726).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (804).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (773).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (742).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (713).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (762).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (777).png" alt=""><figcaption></figcaption></figure>
