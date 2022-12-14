---
description: Deep Metric Learning to Rank, CVPR 2019
---

# \[CVPR 2019] AP-Loss

{% embed url="http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf" %}

{% embed url="https://github.com/kunhe/FastAP-metric-learning" %}

### Abstract

作者基于learning to rank的方法提出了一种新的深度学习方法，FastAP，通过一种源自距离量化的近似方法来优化**rank-based Average Precision**。FastAP具有较低的复杂度，适应于stochastic gradient descent (SGD)。为了全面探索该方法的优势，作者还提出了一种新的minibatch sampling策略，一种允许large-batch training的新启发式方法。

### Introduction

metric learning中最重要的应用领域就是nearest neighbor retrieval。对于该问题，几乎所有metric learning都基于相同的指导原则：_the true "neighbors" of a reference object should be closer than its "non-neighbors" in the learned metric space._&#x20;

作者将metric learning视为一种learning to rank问题，其目标是优化受learned metric影响的整体目标排序。直接优化排序相比其他算法有两个主要的优势：1.可以避免训练集的高阶爆炸，并且可以关注于对距离畸变不敏感的排序；2.值得特别注意的是，可以避免使用高度敏感的超参，如距离阈值或margin。 这篇论文的主要贡献为提出一种优化AP的方法，AP被广泛用于information retrieval任务的评估，为了实现这个rank-based and non-decomposable优化目标，作者使用了一种高效的基于量化的近似方法，并让算法适应于SGD。这个方法被称为FastAP。

### Learning to Rank with Average Precision

假设设置一个标准的信息检索任务，给定特征空间$$\chi$$，有一个query $$q \in \chi$$和一个检索数据集$$\mathcal{R} \subset \chi$$。我们的目标是训练一个神经网络$$\Psi: \chi \rightarrow \mathbb{R}^{m}$$，将输入嵌入到一个m维的欧拉空间中，并且在欧氏空间中优化AP。&#x20;

为了实现最近邻检索，我们首先要根据与q的距离对$$\mathcal{R}$$中的目标进行排序，得到一个有序的列表$${x_1, x_2,...,x_N}$$，其中$$N=|\mathcal{R}|$$。然后，我们可以得到Precision-Recall曲线：&#x20;

![](<../../.gitbook/assets/image (316).png>)

其中，Prec(i)和Rec(i)为有序列表中第i个位置上的准确率和召回率。由此，可以计算AP：&#x20;

![](<../../.gitbook/assets/image (1028).png>)

为了方便，我们假设Prec(0)=Rec(0)=0.&#x20;

上述得到AP的方法有一个问题，就是为了获得p-r曲线，需要先获得一个有序的列表，而这一步中包含了离散的排序操作。对于基于梯度的优化来说，排序是主要的障碍：虽然排序几乎处处可微，它的倒数是0或者未定义的。相反，作者的主要观点为：AP会存在另一种解释，它是基于把准确率和召回率看作距离的函数这一观点的，而非基于有序的元素。

#### FastAP

在信息检索领域，AP也可以解释为the area under precision-recall curve (AUPR)。当公式3的基数趋于无穷，这一关系是存在的：&#x20;

![](<../../.gitbook/assets/image (1012).png>)

其中$$\mathcal{R}^+, (\mathcal{R}^-) \subset \mathcal{R}$$代表了q的匹配（非匹配）集合。AP的AUPR解释允许将准确率和召回率看作距离，而非有序元素的有参数函数。这样可以帮助我们避免不可微的排序操作，进而提出一种AP的近似方法。 一个连续的p-r曲线（不是如公式1中那种有限的集合）可以定义为：&#x20;

![](<../../.gitbook/assets/image (891).png>)

其中z表示query与$$\mathcal{R}$$中元素的距离，z在区域$$\Omega$$中。AP随之变为：&#x20;

![](<../../.gitbook/assets/image (362).png>)

接着，我们定义一些概率量化来计算公式7。令$$\mathcal{Z}$$为对应距离z的随机变量，那么$$\mathcal{R}^+, \mathcal{R}^-$$的距离分布可以定义为$$p(z|\mathcal{R}^+), p(z|\mathcal{R}^-)$$。令$$P(\mathcal{R}^+)$$和$$P(\mathcal{R}^-)=1-P(\mathcal{R}^+)$$为先验概率，表示了检索集合$$\mathcal{R}$$相对于query的偏度。最后，令$$F(z)=P(\mathcal{Z}< z)$$来表示$$\mathcal{Z}$$的累积分布。 基于以上定义，准确率和召回率可以定义为：&#x20;

![](<../../.gitbook/assets/image (335).png>)

带入公式7，得到：&#x20;

![](<../../.gitbook/assets/image (869).png>)

显然地，公式12可以用有限集合来近似估计。我们首先假设嵌入函数$$\Psi$$的输出是L2-normalized，因此，$$\Omega$$或者说公式12的z是属于\[0,2]的。然后，我们将\[0,2]用有限集合量化为$$Z=\{z_1,z_2,...,z_L\}$$，令产生的离线概率分布函数PDF为P，最后我们定义这种新的近似为FastAP：&#x20;

![](<../../.gitbook/assets/image (1065).png>)

接着，作者用直方图符号来重新说明FastAP。明确的来说，作者构建了一个距离直方图，每个bin的中心点（中值）为Z的每个元素。令$$h_j$$为第j个bin中元素的数量，$$H_j=\sum_{k\le j}h_k$$为直方图的累积和。并且，令$$h^+_j$$为第j个bin内query的正确匹配数量，$$H^+_j$$为其累积和。根据这些定义，我们可以重写公式13的概率量化，得到一个简单的表达式：&#x20;

![](<../../.gitbook/assets/image (698).png>)

进行histogram bining和计算FastAP的时间复杂度为O(NL)。

### Stochastic Optimization

AP被定义为关于query和retrieval set间的检索问题。在minibatches中，一个自然的选择是定义in-batch检索问题，其中检索集$$\mathcal{R}$$被限制在minibatch中。特别地，我们将每个样本都视为q，来从这个batch内其他样本中检索匹配。每个样本的检索都可以得到一个AP，一个minibatch内的整体目标即为它们的平均值mAP。&#x20;

为了使用梯度下降法优化目标，公式14内的直方图必须使用允许梯度下降的方法来构建。为此，我们使用了简单的线性插值技术来用一种可微的soft bining技术来代替一般的bining处理。这种插值使得整数型的bin计数变为连续的，定义为$$\hat{h}$$，累积和为$$\hat{H}$$。基于这一可微的bining处理，我们现在可以获得FastAP的偏微分。&#x20;

并且，与doap中的bining不同，这篇论文中的FastAP可以直接用于训练浮点型描述子，而doap中对应部分其实是将浮点型描述子量化为与二进制描述子一样的直方图，然后用二进制描述子的优化方法去训练，会带来额外的损失。

### Large-Batch Training

作者首先说了，data parallelism对于FastAP是不可取的，因为FastAP是不可分解的：即每个样本目标函数的值是由这个batch内其他样本来决定的。&#x20;

作者提出一种启发式的方法来让FastAP可以进行large-batch training。The main insight is that the loss layer takes the embedding matrix of the minibatch as input (see supplementary material). Thus, a large batch can be first broken into smaller chunks to incrementally compute the embedding matrix. Then, we compute the gradients with respect to the embedding matrix, which is a relatively lightweight operation involving only the loss layer. Finally, gradients are back-propagated through the network, again in chunks. This solution works even with a single GPU.（没看懂.....）

### Minibatch Sampling

![](<../../.gitbook/assets/image (559).png>)

大体上来讲，就是作者提出一种采样方法来让一个batch内的negatives更hard，作者利用categories这一概念，一个category包含一些class label对应于此的类，所以在采样一个batch的数据时，先挑选少量几个categories，再从每个category中挑选单独的类，这样一个category中不同类就构成了hard negatives。

### code with comments

代码是作者开源的，加了一些自己的注释方便理解。

```python
import torch
from torch.autograd import Variable, Function

def softBinning(D, mid, Delta):
    """
    Args:
        D:      torch.Tensor(N x N), distance matrix
        mid:    torch.Tensor(1), middle value of an interval in histogram
        Delta:  torch.Tensor(1), step of histogram
    """
    y = 1 - torch.abs(D-mid)/Delta
    return torch.max(torch.Tensor([0]).cuda(), y)

def dSoftBinning(D, mid, Delta):
    side1 = (D > (mid - Delta)).type(torch.float)
    side2 = (D <= mid).type(torch.float)
    ind1 = (side1 * side2) #.type(torch.uint8)

    side1 = (D > mid).type(torch.float)
    side2 = (D <= (mid + Delta)).type(torch.float)
    ind2 = (side1 * side2) #.type(torch.uint8)

    return (ind1 - ind2)/Delta
    

class FastAP(torch.autograd.Function):
    """
    FastAP - autograd function definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank", 
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019

    NOTE:
        Given a input batch, FastAP does not sample triplets from it as it's not 
        a triplet-based method. Therefore, FastAP does not take a Sampler as input. 
        Rather, we specify how the input batch is selected.
    """

    @staticmethod
    def forward(ctx, input, target, num_bins):
        """
        Args:
            input:     torch.Tensor(N x embed_dim), embedding matrix
            target:    torch.Tensor(N x 1), class labels
            num_bins:  int, number of bins in distance histogram
        """
        N = target.size()[0]
        assert input.size()[0] == N, "Batch size donesn't match!"
        
        # 1. get affinity matrix
        Y   = target.unsqueeze(1) # shape(N)
        Aff = 2 * (Y == Y.t()).type(torch.float) - 1 # shape(N, N), value{-1, 1}, 1:matched, -1:unmatched
        Aff.masked_fill_(torch.eye(N, N).byte(), 0)  # set diagonal to 0

        I_pos = (Aff > 0).type(torch.float).cuda() # bool, positive matches
        I_neg = (Aff < 0).type(torch.float).cuda() # bool, negatives
        N_pos = torch.sum(I_pos, 1) # the number of positives for each query

        # 2. compute distances from embeddings
        # squared Euclidean distance with range [0,4]
        dist2 = 2 - 2 * torch.mm(input, input.t()) # shape(N, N), value[0, 4], less -> more similar

        # 3. estimate discrete histograms
        Delta = torch.tensor(4. / num_bins).cuda() # step
        Z     = torch.linspace(0., 4., steps=num_bins+1).cuda() # histograms
        L     = Z.size()[0] # length of histograms
        h_pos = torch.zeros((N, L)).cuda() # shape(N, L)
        h_neg = torch.zeros((N, L)).cuda() # shape(N, L)
        for l in range(L): # for each interval of histogram
            pulse    = softBinning(dist2, Z[l], Delta) # shape(N, N), the distance ratio related to corresponding interval
            h_pos[:,l] = torch.sum(pulse * I_pos, 1) # number of positives locating in corresponding interval
            h_neg[:,l] = torch.sum(pulse * I_neg, 1) # number of negatives locating in corresponding interval

        H_pos = torch.cumsum(h_pos, 1) # shape(N, L), number of positive matches for each query under threshold (precision)
        h     = h_pos + h_neg # shape(N, L)
        H     = torch.cumsum(h, 1) # shape(N, L), number of total matches for each query under threshold (base)
        
        # 4. compate FastAP, as in paper "Deep Metric Learning to Rank"
        FastAP = h_pos * H_pos / H
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP,1) / N_pos
        FastAP = FastAP[ ~torch.isnan(FastAP) ]
        loss   = 1 - torch.mean(FastAP)
        if torch.rand(1) > 0.99:
            print("loss value (1-mean(FastAP)): ", loss.item())

        # 6. save for backward
        ctx.save_for_backward(input, target)
        ctx.Z     = Z
        ctx.Delta = Delta
        ctx.dist2 = dist2
        ctx.I_pos = I_pos
        ctx.I_neg = I_neg
        ctx.h_pos = h_pos
        ctx.h_neg = h_neg
        ctx.H_pos = H_pos
        ctx.N_pos = N_pos
        ctx.h     = h
        ctx.H     = H
        ctx.L     = torch.tensor(L)
        
        return loss

    
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors

        Z     = Variable(ctx.Z     , requires_grad = False)
        Delta = Variable(ctx.Delta , requires_grad = False)
        dist2 = Variable(ctx.dist2 , requires_grad = False)
        I_pos = Variable(ctx.I_pos , requires_grad = False)
        I_neg = Variable(ctx.I_neg , requires_grad = False)
        h     = Variable(ctx.h     , requires_grad = False)
        H     = Variable(ctx.H     , requires_grad = False)
        h_pos = Variable(ctx.h_pos , requires_grad = False)
        h_neg = Variable(ctx.h_neg , requires_grad = False)
        H_pos = Variable(ctx.H_pos , requires_grad = False)
        N_pos = Variable(ctx.N_pos , requires_grad = False)

        L     = Z.size()[0]
        H2    = torch.pow(H,2)
        H_neg = H - H_pos

        # 1. d(FastAP)/d(h+)
        LTM1 = torch.tril(torch.ones(L,L), -1)  # lower traingular matrix
        tmp1 = h_pos * H_neg / H2
        tmp1[torch.isnan(tmp1)] = 0

        d_AP_h_pos = (H_pos * H + h_pos * H_neg) / H2 
        d_AP_h_pos = d_AP_h_pos + torch.mm(tmp1, LTM1.cuda())
        d_AP_h_pos = d_AP_h_pos / N_pos.repeat(L,1).t()
        d_AP_h_pos[torch.isnan(d_AP_h_pos) | torch.isinf(d_AP_h_pos)] = 0


        # 2. d(FastAP)/d(h-)
        LTM0 = torch.tril(torch.ones(L,L), 0)  # lower triangular matrix
        tmp2 = -h_pos * H_pos / H2
        tmp2[torch.isnan(tmp2)] = 0

        d_AP_h_neg = torch.mm(tmp2, LTM0.cuda())
        d_AP_h_neg = d_AP_h_neg / N_pos.repeat(L,1).t()
        d_AP_h_neg[torch.isnan(d_AP_h_neg) | torch.isinf(d_AP_h_neg)] = 0


        # 3. d(FastAP)/d(embedding)
        d_AP_x = 0
        for l in range(L):
            dpulse = dSoftBinning(dist2, Z[l], Delta)
            dpulse[torch.isnan(dpulse) | torch.isinf(dpulse)] = 0
            ddp = dpulse * I_pos
            ddn = dpulse * I_neg

            alpha_p = torch.diag(d_AP_h_pos[:,l]) # N*N
            alpha_n = torch.diag(d_AP_h_neg[:,l])
            Ap = torch.mm(ddp, alpha_p) + torch.mm(alpha_p, ddp)
            An = torch.mm(ddn, alpha_n) + torch.mm(alpha_n, ddn)

            # accumulate gradient 
            d_AP_x = d_AP_x - torch.mm(input.t(), (Ap+An))

        grad_input = -d_AP_x
        return grad_input.t(), None, None    


class FastAPLoss(torch.nn.Module):
    """
    FastAP - loss layer definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank", 
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """
    def __init__(self, num_bins=10):
        super(FastAPLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, batch, labels):
        return FastAP.apply(batch, labels, self.num_bins)
```
