# Deep Hashing Network

对于大多数现有的图像检索的监督哈希方法，首先将图像表示为手工制作或机器学习特征的向量，然后是另一个单独的量化步骤，生成二进制代码。

在本文中，提出了一种新的深度哈希网络(DHN)结构，在该结构中，我们共同学习适合哈希编码的良好图像表示，并正式控制量化误差。以构建**数据依赖**的哈希编码以实现有效的图像检索。





哈希学习可以分为无监督方法和有监督方法。虽然无监督方法更普遍，可以在没有语义标签或相关性的情况下进行训练，但它们受到语义差距困境的限制(Smeuldersetal.2000)，即对象的高级语义描述通常不同于低级特征描述符。监督方法可以结合语义标签或相关性来减轻语义差距，提高哈希质量，即用更少的代码位实现准确的搜索



使用深度神经网络可以更有效地学习特征表示和哈希编码，而且可以自然地编码任何非线性哈希函数。

****

 Hashing for  similarity search: A survey

CNNH  DNNH

____

利用两个关键问题进一步改进了DNNH：（1）有原则地控制量化误差，（2）设计了一个更有原则的成对交叉熵损失来连接成对汉明距离与两两相似标签

****

## 模型架构

![image-20220309195207934](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220309195207934.png)

DHN模型由四个关键部分组成：（1）一个具有多个卷积池层的子网络来捕获图像表示；

​													   （2）是全连接哈希层，生成紧凑的二进制哈希码；

​													   （3）用于相似性保持学习的两两交叉熵损失层；

​													   （4）是用于控制哈希质量的两两量化损失。在标准图像检索数据集上的大量实验表明，所提出的DHN                  																模型比最新的最先进的哈希方法有显著的提高。

该架构接受成对形式的输入图像(x~i~，x~j~，s~ij~)

为了鼓励fch层表示$z_i^l$是二进制码，我们首先利用双曲切线(tanh)激活$a^l(x)=tanh(x)$将其输出压缩在[−1,1]内。为了保证fch表示$z_i^l$是良好的哈希编码，我们必须保持S中给定对之间的相似性，并控制将隐藏表示二值化为二进制码的量化误差。

保持成对相似性和在贝叶斯框架中控制控制量化误差。

## 模型公式

1.AlexNet:深度卷积神经网络(CNN)由5个卷积层(conv1-conv5)和3个完全连接层(fc6-fc8)组成

​	每个fc层使用$z^l_i=a^l(W^lz^{l-1}_i+b^l)\ \ a^l=RELU$		

2.fch层：$h_i=z^l_i$ 使用激活函数$a^l(x)=tanh(x)$将其输出压缩在[−1,1]内。为了保证fch表示$z_i^l$是良好的哈希编码，我们必须保持S中给定对之间的相似性，并控制将隐藏表示二值化为二进制码的量化误差。

公式如下:

$ dist_H(h_i,h_j) = \frac 12(K− <h_i,h_j>)$

给定成对相似性标签S={s~ij~}，哈希码H=[h~1~，...，h~N~]的对数最大后验(MAP)估计如下：
$$
log p (H|S) ∝ log p (S|H) p (H)=\sum_{s_{ij} \in S}logp(s_{ij}|h_i,h_j)p(h_i)p(h_j)\ \ \ \ \ \ \ \ \ (1)
$$
p(H)为先验分布
$$
p(s_{ij}|h_i,h_j) =
\begin{cases}
\sigma(<Z^l_i,Z^l_j>), & s_{ij}=1\\
1-\sigma(<Z^l_i,Z^l_j>), & s_{ij}=0
\end{cases}\\
=\sigma(<Z^l_i,Z^l_j>)^{s_{ij}}(1-\sigma(<Z^l_i,Z^l_j>))^{1-s_{ij}}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (2)
$$
$\sigma(x)=\frac{1} {1+e^{-x}},$ $h_i=z^l_i$

由于具有二值约束的h~i~∈{−1,1}^K^的方程（1）的离散优化非常具有挑战性，为了便于优化，连续松弛应用于二元约束，现有的哈希方法广泛采用。然而，连续松弛将导致两个重要问题：（1）不可控量化误差通过二值化连续嵌入二进制代码，和（2）通过采用连续嵌入之间的内积作为代理汉明二进制代码之间的距离产生了非常大的近似误差。

为了控制量化误差，缩小汉明距离与代理距离在学习高质量哈希码之间的差距，本文提出了一种新的双峰拉普拉斯先验方法

![image-20220309204237476](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220309204237476.png)

![image-20220309204259386](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220309204259386.png)

DNH的优化问题

![image-20220309205257965](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220309205257965.png)

 $\lambda=1/\epsilon$是成对交叉熵损失L和成对量化损失Q之间的权衡参数

$Θ=\{W^l, b^l\} $表示网络参数的集合

![image-20220309204845244](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220309204845244.png)

![image-20220310150441024](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220310150441024.png)

![image-20220309204903112](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220309204903112.png)

1∈R^K^的向量

![image-20220310150512471](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220310150512471.png)

由于Q是一个非光滑函数，其导数难以计算，我们采用绝对函数|x|≈logcosh,它将（6）简化为

![image-20220309205216059](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220309205216059.png)

通过优化方程（4）中的MAP估计，我们可以实现对哈希码的统计最优学习

最后，我们可以通过简单的量化$h←sgn(z_l)$得到Kbit的二进制码。由于最小化了（4）的量化误差，这最后的二值化步骤将导致非常小的检索质量损失

****

## DHN实验



![image-20220310150024527](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220310150024527.png)

## DHN变体

DHN-B是没有二值化的DHN变体(h←sgn(zl)没有执行)，这可能是性能的上限。

DHN-Q是没有量化损失的DHN变体（λ=0）

DHN-E是DHN变体，采用广泛采用的成对平方损失$L=\sum_{s_{ij}\in S(S_{ij}-\frac1K<Z_i^l,Z_j^l>)}$，而不是成对交叉熵损失（5）

![image-20220310150338762](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220310150338762.png)