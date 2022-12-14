---
description: >-
  retrieval-based localization, i.e., place recognition or loop closure
  detection
---

# place recognition

### \[IROS 2009] **Online Visual Vocabulary for Robot Navigation and Mapping** <a href="#iros-2009-online-visual-vocabulary-for-robot-navigation-and-mapping" id="iros-2009-online-visual-vocabulary-for-robot-navigation-and-mapping"></a>

这篇论文提出了一个很完整的增量式构建词典树的方法，利用特征跟踪获得基本单元，自下而上构建词典树，词典树的根节点为视觉单词，叶节点为基本单元，词典树（视觉单词）的数量由一个目标函数优化得到，无需人工干预，很新颖。词典的更新构成采用了增量式地更新，利用一些方法避免了重复计算。使用LDA对特征进行降维。应用场景为水下场景的SfM算法。

### \[TRO 2012] **Automatic Visual Bag-of-Words for Online Robot Navigation and Mapping** <a href="#tro-2012-automatic-visual-bag-of-words-for-online-robot-navigation-and-mapping" id="tro-2012-automatic-visual-bag-of-words-for-online-robot-navigation-and-mapping"></a>

这篇论文是在Online Visual Vocabulary \[IROS 2009]的基础上拓展的期刊论文，在这篇论文中，作者对于词典的更新间隔进行了改进，不再是每隔m张图像更新一次词典，而是通过判断特征与单词的关联率来判断词典是否需要更新。并且，对于具有较少信息的分支，词典进行了剪枝，使得结构更加紧凑。

### \[TRO 2012] **Bags of Binary Words for Fast Place Recognition in Image Sequences** <a href="#tro-2012-bags-of-binary-words-for-fast-place-recognition-in-image-sequences" id="tro-2012-bags-of-binary-words-for-fast-place-recognition-in-image-sequences"></a>

个人感觉是“离线训练词典，在线回环检测”这种检测策略的里程碑了，被广泛用于SLAM系统中，效果和实时性都很好。主要贡献是提出了一种更为鲁棒的二进制特征（改进自FAST+BRIEF），用一个树型词典去离散化特征空间，并在树的结构中加入了inverse index table和direct index table的结构，inverse index table储存了图像中单词的权重和图像出现的图像索引值，direct index table储存了图像中特征及其关联的叶节点（视觉单词）。

### \[IROS 2014] **Fast and Effective Visual Place Recognition using Binary Codes and Disparity Information** <a href="#iros-2014-fast-and-effective-visual-place-recognition-using-binary-codes-and-disparity-information" id="iros-2014-fast-and-effective-visual-place-recognition-using-binary-codes-and-disparity-information"></a>

这篇论文作者提出了一种全局描述子，在LDB的基础上加入了disparity信息，提升了回环检测的效果。作者标注了KITTI数据集的回环真值，用以评测回环检测。

### \[RSS 2018] **Lightweight Unsupervised Deep Loop Closure** <a href="#rss-2018-lightweight-unsupervised-deep-loop-closure" id="rss-2018-lightweight-unsupervised-deep-loop-closure"></a>

这篇论文，作者设计了一个CNN模型，用来模拟HoG特征，采用随机射影变换来进行数据增强，采用真值场景的图像，让网络学习到具有良好视觉不变性的全局描述子，该特征CALC在很多数据集上达到了良好的表现。

### \[RAL 2018] **iBoW-LCD: An Appearance-Based Loop-Closure Detection Approach Using Incremental Bags of Binary Words** <a href="#ral-2018-ibow-lcd-an-appearance-based-loop-closure-detection-approach-using-incremental-bags-of-bina" id="ral-2018-ibow-lcd-an-appearance-based-loop-closure-detection-approach-using-incremental-bags-of-bina"></a>

这篇论文，作者设计了一个增量式构建词典树的方法，算法通过inverse index table来检索具有共同视觉单词的历史图像，然后利用一种新提出的dynamic island的自适应的图像聚类方法，计算当前图像与islands之间的相似度，检索到最相似的island，然后以island中相似度最高的图像作为回环候选，利用极线一致性来完成后验。dynamic island是根据当前图像计算的，不是等长划分的，更能适应不同场景、不同拍摄条件的数据集。在检索过程中，新的特征加入，词典树会有更新视觉单词、增加视觉单词和删除视觉单词的过程。最后实验也证明了，删除无用单词，不会降低算法的表现，还能提升检索效率。

### \[ICRA 2015] **IBuILD: Incremental Bag of Binary Words for Appearance Based Loop Closure Detection** <a href="#icra-2015-ibuild-incremental-bag-of-binary-words-for-appearance-based-loop-closure-detection" id="icra-2015-ibuild-incremental-bag-of-binary-words-for-appearance-based-loop-closure-detection"></a>

这篇论文，作者提出了一个增量式构建词典的方法，词典中的视觉单词有前后帧图像中的特征匹配获得，匹配的图像经过合并（删除重复特征）后开始回环检测。整个流程为：先将视觉单词与词典中的单词相互匹配，匹配到的单词为旧单词，未匹配到的为新单词；基于旧单词，利用inverted index找到回环候选；基于新旧单词的数量和出现频率，利用似然估计得到候选的分数，分数最高的视为回环；此外，增加了temporal consistency constraint；检测完t时刻的回环，再利用当前图像中提取的新旧单词对词典树进行更新，新单词加入，旧单词更新出现频率和inverted index。思路简单明了，效果还不错，就是词典规模没有限制，会随着场景增大一直增加。

### \[ICRA 2018] **Assigning Visual Words to Places for Loop Closure Detection** <a href="#icra-2018-assigning-visual-words-to-places-for-loop-closure-detection" id="icra-2018-assigning-visual-words-to-places-for-loop-closure-detection"></a>

这篇论文中，作者提出了一个增量式的回环检测方法，通过特征跟踪的情况来将图像序列动态地划分为places。在各places（即一段图像序列中）累积特征，利用GNG聚类方法来生成视觉单词。在检测时，对于搜索区域内的place进行投票，根据当前图像特征到已有视觉单词的关联对places进行投票。可以预想的，如果出现回环place，那么该place的票数应当更多，否则票数呈现随机分布的状态。在判断回环时，采用两个判断条件：1.二项分布概率小于阈值；2.累积视觉单词数大于某一阈值。找到回环候选的place后，对于place中的历史图像，找到具有相同描述子最多的图像，作为回环候选，经过几何验证和时序一致性验证后，得到回环。

### \[arxiv 2020] **Fast and Incremental Loop Closure Detection with Deep Features and Proximity Graphs** <a href="#arxiv-2020-fast-and-incremental-loop-closure-detection-with-deep-features-and-proximity-graphs" id="arxiv-2020-fast-and-incremental-loop-closure-detection-with-deep-features-and-proximity-graphs"></a>

这篇论文中，作者应用同一图像，缩放成两个不同分辨率输入网络，得到全局描述子和局部特征。应用全局描述子和HNSW方法构建一个增量式、层次化的拓扑图，图中包含了图像的邻近关系，由顶自下搜索，可以逐步直到与当前图像相似的回环候选。找到候选后，利用局部特征进行特征匹配，具有足够多的候选进行下一步验证。经过空间和时间一致性检测，输出最终的回环结果。

### \[CVPR 2016] **NetVLAD: CNN architecture for weakly supervised place recognition** <a href="#cvpr-2016-netvlad-cnn-architecture-for-weakly-supervised-place-recognition" id="cvpr-2016-netvlad-cnn-architecture-for-weakly-supervised-place-recognition"></a>

这篇经典的利用DL的PR论文中，作者受VLAD全局描述子的启发，将其转化为一个可微的CNN层，设计了一个可以end-to-end训练的全局描述子，并用谷歌街景获取triplet训练数据，用triplet ranking loss进行训练。构思非常巧妙，效果很好，算是里程碑式的一个工作吧。

### \[CVPR 2021] **Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition** <a href="#cvpr-2021-patch-netvlad-multi-scale-fusion-of-locally-global-descriptors-for-place-recognition" id="cvpr-2021-patch-netvlad-multi-scale-fusion-of-locally-global-descriptors-for-place-recognition"></a>

这篇论文中，作者用original NetVLAD检索出top-K个候选相似图像后，利用patch-level NetVLAD descriptor进行了spatial score的计算，对候选图像进行了挑选和重排。作者利用patch作为局部区域，提取NetVLAD描述子，进行patch之间的匹配，用匹配分数作为spatial score。作者还提出融合多尺度patch的匹配分数，提升算法表现，利用integral VLAD特征图的技术避免了重复计算不同尺度的VLAD描述子。在实时性和检索精度上都获得了很好的表现，获得了ECCV2020 Facebook Mapillary Visual Place Recognition Challenge的冠军。

### \[ICRA 2007] **A visual bag of words method for interactive qualitative localization and mapping** <a href="#icra-2007-a-visual-bag-of-words-method-for-interactive-qualitative-localization-and-mapping" id="icra-2007-a-visual-bag-of-words-method-for-interactive-qualitative-localization-and-mapping"></a>

很早期的incremental BoW研究，利用人机交互为机器人localization提供对错判断，为在线学习提供监督。采用两层投票机制来识别房间，使用三种特征进行描述，只有在第一层投票中投票质量够高和图像数量够多，才会进行第二层投票。incremental BoW是比较简单的将新特征与之前的单词进行关联，如果关联不上，则新建一个单词。

### \[IROS 2008] **Interactive learning of visual topological navigation** <a href="#iros-2008-interactive-learning-of-visual-topological-navigation" id="iros-2008-interactive-learning-of-visual-topological-navigation"></a>

[ICRA 2007](https://jinyummiao.github.io/post/place-recognition/#12)的改进版，加入了增量式词典构建的过程，和visual homing模块。增量式词典构建是从零开始，如果节点中关联的特征过多，则将其利用k-means继续划分，得到词典树。localization中估计位置和homing中估计目标角度都是用voting的方法，避免了学习过程重复学习之前的数据。采用SIFT和local color histogram两种特征。

### \[TRO 2008] **Fast and Incremental Method for Loop-Closure Detection Using Bags of Visual Words** <a href="#tro-2008-fast-and-incremental-method-for-loop-closure-detection-using-bags-of-visual-words" id="tro-2008-fast-and-incremental-method-for-loop-closure-detection-using-bags-of-visual-words"></a>

在[ICRA](https://jinyummiao.github.io/post/place-recognition/#12)和[IROS 2008](https://jinyummiao.github.io/post/place-recognition/#13)的基础上，用贝叶斯方法检索回环，同样增量式构建词典，利用SIFT和local color histogram两种特征分别构建词典并检测，词典带有inverted index结构，保存具有某单词的历史图像，通过该结构，在检索过程中，对历史图像进行投票，基于vote计算可能出现回环的概率。出现回环的后验概率大于某一阈值并且图像通过几何一致性检验的假设被认为是回环。

### \[IROS 2016] **Encoding the Description of Image Sequences: A Two-Layered Pipeline for Loop Closure Detection** <a href="#iros-2016-encoding-the-description-of-image-sequences-a-two-layered-pipeline-for-loop-closure-detect" id="iros-2016-encoding-the-description-of-image-sequences-a-two-layered-pipeline-for-loop-closure-detect"></a>

提出一种在图像序列上的BoW模型，离线构建词典树，将输入的图像流按照里程计反馈，划分为等物理长度的片段，对每个片段和片段中的图像提取特征，量化为BoW向量，利用inverse indexing table找到具有共视关系的序列，利用序列BoW向量进行匹配，根据时间一致性对相似度矩阵进行滤波，得到匹配序列，在匹配序列中，根据inverse indexing table找到共视图像，利用图像BoW向量进行检索，加入时空一致性检验。

### \[IJCAI 2021] **Where is your place, Visual Place Recognition?** <a href="#ijcai-2021-where-is-your-place-visual-place-recognition" id="ijcai-2021-where-is-your-place-visual-place-recognition"></a>

一篇综述论文，从agent、environment和tasks三个方向讨论了VPR。2016年Lowry的TRO综述中定义VPR问题为"given an image of a place, can a human, animal, or robot decide whether or not this image is of place it has already seen?"这篇论文中，作者根据visual overlapping对VPR做出了新的定义："the ability to recognitize one's localtion based on two observations preceived from overlapping field-f-views."\
作者指出VPR与image retrieval的区别在于，image retrieval旨在搜索类别相同的相似图像，而VPR旨在搜索相同地点的图像，而非相同类别的图像，相同地点的图像可能视觉相似度并不高。\
作者提出要根据场景和任务需求来平衡viewpoint-和appearance-invariance。比如室内的UAV，需要更强的viewpoint-invariance，而非appearance-invariance。\
在SLAM中使用VPR（比如回环检测），错误的匹配会产生灾难性的建图失败，所以需要很高精度的VPR。\
对于全局描述子，如果不增加训练数据，提升viewpoint-invariance必定会损失appearance-invariance。

### \[RAL 2020] **CoHOG: A Light-Weight, Compute-Efficient, and Training-Free Visual Place Recognition Technique for Changing Environments** <a href="#ral-2020-cohog-a-light-weight-compute-efficient-and-training-free-visual-place-recognition-technique" id="ral-2020-cohog-a-light-weight-compute-efficient-and-training-free-visual-place-recognition-technique"></a>

计算图像像素的熵，然后进一步计算图像区域的熵，选择大于阈值的区域作为信息丰富的区域，计算区域内的HOG描述子，匹配query与reference图像中region HOG descriptor，得到匹配分数，求平均作为图像匹配分数。

### \[IROS 2017] **Improving Condition- and Enviroment- Invariant Place Recognition with Semantic Place Categorization** <a href="#iros-2017-improving-condition-and-enviroment-invariant-place-recognition-with-semantic-place-categor" id="iros-2017-improving-condition-and-enviroment-invariant-place-recognition-with-semantic-place-categor"></a>

利用语义信息将图像序列划分，代替了SeqSLAM中固定尺寸的划窗策略，提升在一些条件多变场景中的鲁棒性。在获得图像的语义标签时，先用CNN获得分类概率，再输入HMM，根据图像的时间连续性，来获得图像的语义标签。

### \[RSS 2018] **LoST? Appearance-Invariant Place Recognition for Opposite Viewpoints using Visual Semantics** <a href="#rss-2018-lost-appearance-invariant-place-recognition-for-opposite-viewpoints-using-visual-semantics" id="rss-2018-lost-appearance-invariant-place-recognition-for-opposite-viewpoints-using-visual-semantics"></a>

结合基于语义和外观的全局描述子，和空间一致的局部关键点对应关系来进行场景识别。由语义标签和卷积特征图获得一种全局特征描述子LoST，对于匹配图像，提取特征图中响应值最大的位置作为关键点，进行特征匹配，进行空间一致性验证。可以应对相对视角的问题。

### \[ICRA 2018] **Don't Look Back: Robustifying Place Categorization for Viewpoint- and Condition-Invariant Place Recognition** <a href="#icra-2018-dont-look-back-robustifying-place-categorization-for-viewpoint-and-condition-invariant-pla" id="icra-2018-dont-look-back-robustifying-place-categorization-for-viewpoint-and-condition-invariant-pla"></a>

利用Place365网络提取特征，得到具备视角不变性的全局描述子，对全局描述子进行标准化，得到外观不变性。利用SeqSLAM框架搜索匹配的图像。为了解决相对视角的问题，从图像左右分别提取区域，提取两个全局描述子，分别计算距离，取较小值作为相似度。

### \[IROS 2019] **CALC2.0：Combining Appearance, Semantic and Geometric Information for Robust and Efficient Visual Loop Closure** <a href="#iros-2019-calc20combining-appearance-semantic-and-geometric-information-for-robust-and-efficient-vis" id="iros-2019-calc20combining-appearance-semantic-and-geometric-information-for-robust-and-efficient-vis"></a>

设计了一个单输入、多输出的VAE网络，让网络同时预测语义信息和重建RGB信息，结合了几何信息、外观信息和语义信息。利用triplet loss进行训练。中间层的隐含变量作为全局描述子。从conv5层的特征图中提取每个划窗中的最大响应区域作为特征关键点，去掉重复特征。参考BRIEF描述子，得到关键点的描述子。在回环检测时，先用全局描述子检索可能的回环，再用关键点匹配验证。

### \[RAL 2021] **STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition** <a href="#ral-2021-sta-vpr-spatio-temporal-alignment-for-visual-place-recognition" id="ral-2021-sta-vpr-spatio-temporal-alignment-for-visual-place-recognition"></a>

提出一种时空上对齐序列的匹配方法，来增强中层CNN特征的视角不变性。时间上，利用局部匹配DTW来减少匹配图像序列的时间复杂度；空间上，将图像特征图沿水平方向纵向切片，得到图像局部特征，利用自适应DTW来对齐局部特征，得到图像相似度。

### \[RAL 2019] **Probabilistic Appearance-Based Place Recognition Through Bag of Tracked Words** <a href="#ral-2019-probabilistic-appearance-based-place-recognition-through-bag-of-tracked-words" id="ral-2019-probabilistic-appearance-based-place-recognition-through-bag-of-tracked-words"></a>

利用特征跟踪来产生tracked words，在线构成词典，无需预训练过程。在检索过程中，当前图像的特征被关联到database中的tracked words，进行投票，利用票数计算二项概率密度函数的值，作为候选回环提取的可信度，对于候选回环进行RANSAC后验。

### \[RAL 2021] **SeqNet: Learning Descriptors for Sequence-Based Hierarchical Place Recognition** <a href="#ral-2021-seqnet-learning-descriptors-for-sequence-based-hierarchical-place-recognition" id="ral-2021-seqnet-learning-descriptors-for-sequence-based-hierarchical-place-recognition"></a>

提出了一种hierarchical VPR算法，用NetVLAD提取图像描述子，在filtering阶段，提出了一种SeqNet，其实就是一个中时间维度的1D卷积，将时间维度上的图像描述子聚合成一个序列描述子，与referenced database进行匹配，获得top-K候选。在re-ranking阶段，用简化的SeqSLAM进行序列匹配，用SeqNet(输入长度为1)对图像描述子进行一次线性变换。

### \[CVPRW 2021] **SeqNetVLAD vs PointNetVLAD: Image Sequence vs 3D Point Clouds for Day-Night Place Recognition** <a href="#cvprw-2021-seqnetvlad-vs-pointnetvlad-image-sequence-vs-3d-point-clouds-for-day-night-place-recognit" id="cvprw-2021-seqnetvlad-vs-pointnetvlad-image-sequence-vs-3d-point-clouds-for-day-night-place-recognit"></a>

这篇报告指出在极端昼夜情况下，在描述子中加入时序信息有可能获得与基于3D数据的描述子相比相近或更好的定位表现。

### \[RAL 2020] **Delta Descriptors: Change-Based Place Representation for Robust Visual Localization** <a href="#ral-2020-delta-descriptors-change-based-place-representation-for-robust-visual-localization" id="ral-2020-delta-descriptors-change-based-place-representation-for-robust-visual-localization"></a>

提出一种序列化图像的描述子，对图像描述子进行时间维度上的smoothing后，计算difference，得到基于difference的delta descriptors。我感觉，SeqNet就是将Delta Descriptors中1D卷积核变成了可训练的，即滑窗内各图像描述子不再是简单的权重为1，后半段减前半段求得difference，而是通过卷积加权求和。

### \[BMVC 2020] LiPo-LCD: Combining Lines and Points for Appearance-based Loop Closure Detection

在回环检测算法中同时使用点特征和线特征，并行使用OBIndex2来增量式维护点特征和线特征的视觉词典，并完成回环检测。使用dynamic islands来加入时间一致性约束。在RANSAC后验中加入线特征匹配，以线的端点作为额外的点匹配，帮助计算F矩阵。

### \[TIP 2021] DASGIL: Domain Adaptation for Semantic and Geometric-Aware Image-Based Localization

设计了一个单输入双输出的网络，采用多任务策略训练（深度估计、语义分割），对中间层多尺度特征进行真实数据-虚拟数据的domain adaptation监督和度量学习。从虚拟数据中学习到融合语义和几何信息的CNN特征，用于图像检索。

### \[RAL 2021] BVMatch: Lidar-Based Place Recognition Using Bird’s-Eye View Images

在BEV视角下，对稀疏3D点云进行栅格化，基于栅格内点云的数量（density）构建BV image。基于Log-Gabor filter构建MIM，以此设计具有旋转不变性的局部特征描述子BVFT，用FAST检测关键点。用K-means构建词典，实现场景识别。同时可以对两个BV image进行相对位姿计算（三个自由度），实现定位。
