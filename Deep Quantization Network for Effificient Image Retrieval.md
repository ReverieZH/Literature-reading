# **Deep Quantization Network for Effificient Image Retrieval**

## 模型架构

DQN模型由四个关键部分组成：（1）一个具有多个卷积池层的子网络来捕获深度图像表示；

​													   （2）一个全连接层来生成哈希编码最优降维表示

​													   （3）用于保持相似性的学习的成对余弦损失层

​														（4）用于控制哈希质量和瓶颈表示量化性的乘积量化损失。

![image-20220310155146158](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220310155146158.png)

## 模型公式

学习到B-bit hashc ode , fcb层有R个单元，使用tanh(x)

$dist_H(h_i,h_j)=\frac 12(B-<h_i,h_j>)$

![image-20220311113438139](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220311113438139.png)

$\Large \frac { <z_i^l,z_j^l> } {||z^l_i||||z^l_j||} \in \{ -1,1\}$ 余弦距离被广泛用于减轻向量长度的多样性和提高检索质量，而在监督哈希学习中还没有得到很好的探索

![image-20220311114102268](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220311114102268.png)

将$fcb的输出表示划分在M个子空间中 z^l_i=\{\ z^l_{i1},z^l_{i2}....z^l_{iM}\}\ \ \ \ z^l_{iM}\in \Bbb R^{R/M}$ 

通过k-Mean将每个子空间m的所有子向量$\{z^l_{im}\}^N_{i=1}$量化为K个簇   R->M->K

$\Large C_m=[c_{m1},c_{m2}...c_{mK}]\in \Bbb R^{\frac R M \times K} $表示第m个子空间中的K个码字（集群中心）的码本

$h_{im}$是一个k位的编码 表示 在第m个codebook$C_m$中的一个k位codeword  是用来近似第i点$z_i^l$

$h_i=[h_{i1},h_{i2}....h_{im}] \in \Bbb R ^{MK}$是$z_i^l$的编码,用来连接M个子空间的k位编码$\{h_{im}\}$



使用（1）控制了将fcb的特征表示zli二值化为二进制码hi的量化误差，而（2）提高了fcb表示zli的量化性，从而可以有效地量化

![image-20220311144338621](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220311144338621.png)

![image-20220311144417558](C:\Users\13449\AppData\Roaming\Typora\typora-user-images\image-20220311144417558.png)

给定固定的${z^l_i}$表示，码本$C=diag(C_1，...，C_M)$和二进制码$H=[h_1，...，h_N]$可以通过M个独立的k-mean来学习

·