# hand-crafted features



<table><thead><tr><th width="150">Feature</th><th width="150">from</th><th width="453.8210180623974">innovation</th></tr></thead><tbody><tr><td>Harris角点</td><td>ACC 1998</td><td>采用二维高斯函数形式的窗口函数，对窗口内像素点变化函数E泰勒展开，计算参数矩阵M的特征值，定义角点响应值<span class="math">R=det(M)-k\cdot {tr}^2(M)</span>，根据R的值判断边缘、角点或平滑区域</td></tr><tr><td>GFTT角点</td><td>CVPR 1998</td><td>Good Feature to Track的角点响应值被定义为<span class="math">R=min(\lambda_1,\lambda_2)</span>，相比Harris角点要求更严苛</td></tr><tr><td>SIFT特征</td><td>IJCV 2004</td><td>在尺度空间DoG中检测极值，作为关键点，统计方向直方图，作为关键点的方向，利用主要方向旋转图像获得旋转不变性，由方向直方图获得256维局部描述子</td></tr><tr><td>SURF特征</td><td>CVIU 2008</td><td>构建尺度空间，求Hessian矩阵，根据Hessian矩阵的判别式检测极值作为关键点，统计扇形区域内的Haar小波响应，得到主方向，在关键点周围4x4个子区域内求5x5像素邻域的水平方向和垂直方向的小波特征，得到64维描述子</td></tr><tr><td>ORB特征</td><td>ICCV 2011</td><td>利用FAST算法检测关键点，根据目标像素与周围像素的灰度值来判别关键点，利用BRIEF算法描述关键点，提取二进制描述子，非常高效</td></tr></tbody></table>
