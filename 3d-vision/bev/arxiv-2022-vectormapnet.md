---
description: 'VectorMapNet: End-to-end Vectorized HD Map Learning'
---

# \[arxiv 2022] VectorMapNet

{% embed url="https://tsinghua-mars-lab.github.io/vectormapnet/" %}

## Abstract

Autonomous driving systems require a good understanding of surrounding environments, including moving obstacles and static High-Definition (HD) semantic map elements. Existing methods approach the semantic map problem by offline manual annotation, which suffers from serious scalability issues. Recent learning-based methods produce dense rasterized segmentation predictions to construct maps. However, these predictions do not include instance information of individual map elements and require heuristic post-processing, that involves many hand-designed components, to obtain vectorized maps. To that end, we introduce an end-to-end vectorized HD map learning pipeline, termed VectorMapNet. VectorMapNet takes onboard sensor observations and predicts a sparse set of polylines primitives in the bird’s-eye view to model the geometry of HD maps. This pipeline can explicitly model the spatial relation between map elements and generate vectorized maps that are friendly to downstream autonomous driving tasks without the need for post-processing.

## Introduction

