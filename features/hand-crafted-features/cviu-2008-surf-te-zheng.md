# \[CVIU 2008] SURF特征

{% embed url="https://www.sciencedirect.com/science/article/abs/pii/S1077314207001555" %}

### 基本介绍

> SURF (Speeded Up Robust Features) is a robust local feature detector, first presented by Herbert Bay et al. in 2006, that can be used in computer vision tasks like object recognition or 3D reconstruction. It is partly inspired by the SIFT descriptor.The standard version of SURF is several times faster than SIFT and claimed by its authors to be more robust against different image transformations than SIFT. SURF is based on sums of 2D Haar wavelet responses and makes an efficient use of integral images. It uses an integer approximation to the determinant of Hessian blob detector, which can be computed extremely quickly with an integral image (3 integer operations). For features, it uses the sum of the Haar wavelet response around the point of interest. Again, these can be computed with the aid of the integral image.

SURF特征先利用Hessian矩阵确定候选点，然后进行非极大抑制，计算复杂度大大降低。

### Hessian矩阵构建

SIFT特征建立图像金字塔，在每一层进行高斯滤波并求取图像差DoG进行特征点的提取，而SURF特征用的是Hessian矩阵进行特征点提取，因此Hessian矩阵是SURF特征的核心。假设函数$$f(x,y)$$，Hessian矩阵是由函数偏导数组成的：&#x20;

![](../../.gitbook/assets/surf\_0.png)

从而每个像素点都可以求一个Hessian矩阵，Hessian矩阵判别式为：&#x20;

![](../../.gitbook/assets/surf\_1.png)

判别式的值是Hessian矩阵的特征值，可以根据判别式取值的正负，来判别该点是否是绩点的值。在SURF特征中，函数$$f(x,y)$$为图像像素$$I(x,y)$$，选用二阶标准高斯函数作为滤波器，得到滤波后的图像L，进而得到偏微分$$L_{xx},L_{xy},L_{yy}$$：

![](../../.gitbook/assets/surf\_2.png)

由于特征需要尺度不变性，所以在进行Hessian矩阵构造前，需要对图像进行高斯滤波：

$$L(x,y,\sigma)=G(\sigma)\cdot I(x,y)$$，$$L(x,y,\sigma)$$

是一幅图像在不同解析度下的表示，可以利用高斯核$$G(\sigma)$$与图像函数$$I(x,y)$$在点(x,y)的卷积来实现。在离散数学中，一阶导数是相邻像素的灰度差：$$L_x=L(x+1,y)-L(x,y)$$；二阶导数是对一阶导数的再次求导：

$$L_{xx}=[L(x+1,y)-L(x,y)]-[L(x,y)-L(x-1,y)]=L(x+1,y)+L(x-1,y)-2L(x,y)$$因此：&#x20;

![](../../.gitbook/assets/surf\_3.png)

由于高斯核符合正太分布，从中心点往外，系数越来越小，为了提高运算速度，SURF算法使用了盒式滤波器来代替高斯滤波器，在$$L_{xy}$$上乘了一个加权系数0.9，目的是平衡因使用盒式滤波器近似所带来的误差，则Hessian矩阵的判别式可以表示为：&#x20;

![](../../.gitbook/assets/surf\_4.png)

$$9\times 9$$的高斯滤波器模板和近似的盒式滤波器模板示意图如下，上边两幅图是$$9\times 9$$高斯滤波器模板分别在图像上二阶导数$$D_{yy}$$和$$D_{xy}$$对应的值，下边两幅图是使用盒式滤波器对其近似，灰色部分的像素值为0，黑色为-2，白色为1。

![](../../.gitbook/assets/surf\_5.png)

盒式滤波器可以将图像的滤波转换为计算图像上不同区域间像素的加减运算，即积分图的使用，因此可以提高运算速度。

### 尺度空间生成

图像的尺度空间是这幅图像在不同解析度下的表示。同SIFT特征一样，SURF特征的尺度空间由O组（每一组是一个octave）S层组成，不同的是，SIFT算法下一组图像的长宽均是上一组的一半，同组不同层图像之间大小一样，但是所使用的尺度空间因子（高斯模糊系数$$\sigma$$）逐渐增大；而在SURF特征中，不同组间图像的大小都是一致的，不同的是不同组间使用的盒式滤波器的模板尺寸逐渐增大，同一组不同层图像使用相同尺寸的滤波器，但是滤波器的尺度空间因子（高斯模糊系数$$\sigma$$逐渐增大。如下图：&#x20;

![](../../.gitbook/assets/surf\_6.png)

### 特征点过滤并进行准确定位

SURF特征的定位过程与SIFT特征一致，将每个像素点的Hessian矩阵的判别式值与其图像域和尺度域的所有相邻点进行比较，当其大于（或小于）所有相邻点时，该点就是极值点。初步定位出特征点后，再滤除能量较弱的关键点以及错误定位的关键点，筛选出最终的稳定的特征点。

&#x20;

![](<../../.gitbook/assets/image (219).png>)

### 计算特征点主方向

SURF特征统计特征点圆形邻域内的Haar小波特征，即在特征点的圆形邻域内，统计60度扇形内所有点的水平、垂直方向Haar小波响应的总和，并给这些响应值赋予高斯权重系数，靠近特征点的响应贡献大，远离特征点的响应贡献小，然后扇形以0.2rad为间隔进行旋转，再次统计，将响应总和最大的那个扇形的方向作为该特征点的主方向。&#x20;

![](../../.gitbook/assets/surf\_7.png)

### 生成特征描述

SURF特征提取特征点周围$$4\times 4$$个矩形区域块，所取得的矩形区域是沿着特征点的主方向的。每个子区域统计25个像素点水平方向和垂直方向的Haar小波特征，这里水平、垂直方向都是相对主方向而言的。该Haar小波特征为水平方向值之和，水平方向绝对值之和，垂直方向值之和，垂直方向绝对值之和，一般为4x4x4=64维。&#x20;

![](../../.gitbook/assets/surf\_8.png)
