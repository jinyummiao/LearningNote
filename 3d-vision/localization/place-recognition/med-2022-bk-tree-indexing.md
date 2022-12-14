---
description: >-
  BK tree indexing for active vision-based loop-closure detection in autonomous
  navigation
---

# \[MED 2022] BK-tree indexing

## Abstract

Aiming to recognize familiar places through the camera measurements during a robot’s autonomous mission, visual loop-closure pipelines are developed for navigation frameworks. This is because the main objective for any simultaneous localization and mapping (SLAM) system is its consistent map generation. However, methods based on active vision tend to attract the researchers’ attention mainly due to their offered possibilities. This paper proposes a BK-tree structure for a visual loop-closure pipeline’s generated database when active vision is adopted. This way, we address the drawback of scalability in terms of timing occurring when querying the map for similar locations while high performances and the online nature of the system are maintained. The proposed method is built upon our previous work for visual place recognition, that is, the incremental bag-of-tracked-words. The proposed technique is evaluated on two publicly-available image-sequences. The one is recorded via an unmanned aerial vehicle (UAV) and selected due to its active vision characteristics, while the second is registered via a car; still, it is chosen as it is among the most extended datasets in visual loop-closure detection. Our experiments on an entry-level system show high recall scores for each evaluated environment and response time that satisfies real-time constraints.

## BK Tree for Image Indexing
