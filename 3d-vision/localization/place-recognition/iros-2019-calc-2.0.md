# \[IROS 2019] CALC 2.0

设计了一个单输入、多输出的VAE网络，让网络同时预测语义信息和重建RGB信息，结合了几何信息、外观信息和语义信息。利用triplet loss进行训练。中间层的隐含变量作为全局描述子。从conv5层的特征图中提取每个划窗中的最大响应区域作为特征关键点，去掉重复特征。参考BRIEF描述子，得到关键点的描述子。在回环检测时，先用全局描述子检索可能的回环，再用关键点匹配验证。
