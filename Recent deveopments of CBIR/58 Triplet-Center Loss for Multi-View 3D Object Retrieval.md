# Triplet-Center Loss for Multi-View 3D Object Retrieval





现有的三维对象识别算法大多集中于利用softmax损失的深度学习模型的强鉴别能力进行三维数据分类，而利用深度度量学习学习鉴别特征或多或少被忽略。

在本文中，我们研究了三维对象检索的深度度量学习损失的变量，它没有得到足够的关注。

首先，引入了两种代表性损失，三重损失和中心损失，可以比传统的分类损失学习更多的鉴别特征

在此基础上，提出了一种新的三重中心损失，可以进一步增强特征的鉴别能力。

所提出的三重中心损失为每个类学习一个中心，并要求来自同一类的样本和中心之间的距离比来自不同类的距离更近。

在两个流行的三维对象检索基准和两个广泛采用的基于草图的三维形状检索基准上的广泛实验结果一致证明了我们提出的损失的有效性，与最先进的技术相比，已经取得了显著的改进。

# **1. Introduction**

在过去的几年里，三维形状分析受到了计算机视觉和图形社区的广泛关注。

特别是，由于强大的深度学习方法和大规模的3D模型基准测试（例如ShapeNet），在这一领域已经进行了许多新的尝试。

三维对象检索是形状分析中的一个基本问题，是处理和分析三维数据的最关键问题。

然而，大多数基于深度学习的方法专注于利用深度学习模型的强大鉴别能力来对3D数据进行分类，例如，[33,14,25]，只有少数新的基于深度学习的方法专门为三维对象提出了大规模的检索方法。

由于社区历史悠久，三维对象检索可以大致分为两类：基于视图和基于模型的方法。

- 基于视图的方法[1,33]从一组二维视图投影中提取或学习形状特征，其中通常采用二维卷积神经网络(CNN)来处理此类投影图像。

- 基于模型的方法[11,38]直接从原始的三维表示中获得三维形状特征，因此首选三维CNN。

到目前为止，基于视图的方法在检索精度方面通常优于基于模型的方法，正如最近在大规模3D SHape检索竞赛(SHREC)[4,27,28]的竞赛中所报道的那样。

一个著名的三维对象检索的例子是多视图卷积神经网络(MVCNN)[33]，这是CNN以端到端可训练的方式学习到的多个二维投影特征的组合。与MVCNN类似，人们已经努力建立了一个统一的深度学习模型，可以同时执行三维对象分类和检索的任务。包括MVCNN在内的这些方法认为，一个经过深度学习训练的强分类模型，通常可以同时为三维对象检索提供忠实的相似性。	

事实上，深度学习方法三维对象检索非常类似于图像或其他对象检索，几个损失函数如对比损失[8]和三重损失[29]引入训练CNN，为了学习一个度量或嵌入空间，使实例从同一类别更接近比来自不同类别的。特别是，用端到端度量学习训练一个普通的CNN模型，显示了其在人脸识别[29]和人再识别(re-ID)方面的优势。虽然利用这种损失函数在二维图像集的重新识别任务中取得了显著的进展，但它们或多或少被三维对象检索区域所忽视。事实上，大多数现有的3D形状检索的深度学习方法都专注于设计复杂的深度神经网络架构或利用3D对象的不同表示。

与大多数现有的算法相比，在本文中，我们认为训练CNN的三重损失[29]或中心损失[39]特定的距离测量，也可以带来性能效益3D对象检索，明显优于最先进的方法在最受欢迎的基准数据集，如ModelNet40和 ShapeNet  Core55。

综上所述，我们首先做了以下贡献：

1)介绍了用于三维对象检索的两种典型损失函数，并充分研究了它们对检索性能的影响；

2)我们提出了一种新的损失函数三中心损失(TCL)，并证明了基于同一CNN模型的TCL得到了最先进的结果，优于其他方法。





所提出的TCL，受中心损失[39]和三重损失[29]的启发，学习中心为每个类，要求样本和中心之间的距离从相同的类小于来自不同的类，这样从同一类的样本的特性更接近相应的中心，同时远离其他不同类的中心。

与只关注减少类内变化的中心损失不同，TCL也考虑了类间的可分性。与三联体损失相比，TCL避免了三联体的复杂构造和硬样本挖掘机制。使用TCL，我们的CNN三维对象检索CNN模型建立在MVCNN[33]的框架上，如图1所示。

![image-20220812115918269](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812115918269.png)

因此，我们的方法可以看作是一种基于视图的方法，它将三维形状特征的提取和距离度量学习统一到一个端到端学习过程中。

除了ModelNet40和ShapeNetCore55之外，我们还分别在两个广泛采用的基准测试，SHREC‘13和SHREC’14草图轨迹基准数据集上展示了它在基于草图的3D形状检索任务中的优势。

# **2. Related work**

随着像ShapeNet[7]这样的大规模标记的3D形状集合的出现，最近出现了越来越多的关于3D形状分析，特别是深度学习的文献。我们建议读者参考[34,13]来进行三维形状检索的全面调查。在本节中，我们将主要介绍基于深度学习机制的代表性三维形状检索方法。

一般来说，三维形状检索方法大致可以分为两类：基于三维模型的方法和基于视图的方法。基于三维模型的方法直接从三维数据格式中学习形状特征，如多边形网格或曲面[5,6,42,43]、体素网格[21,40,18,25,30]和点云[24,26]。例如，Furuya等人[11]提出了DLAN网络来直接处理三维形状的局部区域，并聚合局部三维旋转不变特征来执行检索任务。Klokov等人[15]提出了kd网络来处理非结构化点云，并利用学习到的特征来执行检索任务。这些方法的主要局限性在于形状表示的限制（例如，光滑流形），或高计算复杂度，特别是对于基于体素的方法。最近，Wang等人[38]提出了一种基于oc树表示的三维CNN，与传统的基于全身素的表示相比，它可以大大提高计算效率。

基于视图的方法通常呈现一个视图或多个视图的三维形状首先，这样复杂的图像特征提取器例如CNN可以利用提取特征从二维渲染视图，然后这些提取视图特征组装成一个紧凑的形状描述符最后用于检索或分类任务。例如，MVCNN[33]利用最大池化层来聚合由一个共享的CNN提取的不同视图的特性。Bai等人的[1,2]提出了一个基于深度学习的三维形状搜索引擎，名为GIFT，它特别关注形状检索的实时性和可伸缩性。为此，利用GPU和 inverted file加速了视图特征的提取和索引，并在各种形状基准上取得了良好的检索性能。通常，从基于视图的方法中学习到的特征对三维形状更具鉴别性，从而在大多数情况下具有更好的检索性能。



然而，上述方法大多不是专门为三维形状检索任务而设计的，而在图像检索社区，为了学习更鲁棒和鉴别性的特征，深度度量学习已被广泛采用。由温伯格和索尔提出的三重态损失[29]，鼓励具有相同身份的数据点的特征比具有不同身份的数据点的特征更接近。一些三重态损失的变体也被提出，如[19,23,37]。然而，三联体损失可能会遭受耗时挖掘和巨大的数据扩展的问题。为了解决这一问题，最近Hermans等人[12]提出了一种基于批硬的三重损失(BHL)，该方法从在线训练批次中挖掘硬负和硬正样本，并在几个人的re-ID基准上实现了最先进的结果。另一方面，提出了中心损失[39]，它作为 softmax最大损失的辅助损失，以学习更多的鉴别特征。中心损失的主要目标是学习每个类的特征的中心，并将同一类的特征更紧密地拉到相应的中心。中心损失的主要目标是学习每个类的特征的中心，并将同一类的特征更紧密地拉到相应的中心。

受深度度量学习方法在二维图像检索/re-ID任务中的成功应用的启发，我们在三维对象检索中引入了两种具有代表性的深度度量学习损失，即三联体损失和中心损失。

此外，还提出了一种新的三重中心损失计算方法。最近，Wang等人的[36]提出了一个类似的人脸验证损失问题。然而，我们的三重态-中心的损失来自于一个非常不同的直觉1。此外，我们的损失消除了对特征和权值的规范化的需要，并且不重用来自完全连接层的权值。在两个3D形状检索基准和两个基于3D素描的检索基准上，比最先进的显著改进证明了它的有效性。



# **3. Proposed method**

在形状检索任务中，获得一个形状的鲁棒性和判别性表示是获得良好性能的关键。

通常，这可以通过利用softmax损失在标记的训练集上训练CNN来部分实现。然而，**<font color='red'>在softmax损失的监督下优化的学习特征在本质上没有足够的区别性，因为它们只专注于寻找分离不同类形状的决策边界，而没有考虑特征的类内紧凑性</font>**。

如图2(a)所示，虽然两类的样本被决策边界精心分隔，但存在显著的类内差异。

![image-20220812125147921](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812125147921.png)

为了解决这个问题，人们提出了许多深度度量学习算法。本文首先介绍了两种具有代表性的损失，即三联体损失[29]和中心损失[39]。然后，基于这两个损失，我们推导出我们提出的TCL。

## **3.1. Review on triplet loss**

三联体损失，顾名思义，是根据训练样本$(x^i_a，x^i_+，x^i_−)$计算的，其中$(x^i_a，x^i_+)$有相同的类标签，$(x^i_a，x^i_-)$有不同的类标签。

$x^i_a$通常被认为是三重联体的锚。直观地说，三重态损失鼓励找到一个嵌入空间，来自相同类别(即$x^i_+$和$x^i_a$)的样本之间的距离比来自不同类别(即$x^i_−$和$x^i_a$)的样本至少小一个边缘m。

具体来说，三重态损失可以计算如下：

![image-20220812130014831](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812130014831.png)

其中，f（·）表示神经网络的特征嵌入输出，D（·）表示两个输入向量之间的距离，N表示训练集中的三联体数，i表示第i个三联体。

然而，当训练数据集变大时，三联体的数量呈三次增长，这通常会导致一个较长的不切实际的训练周期。

此外，三联体损失的性能在很大程度上依赖于硬三联体的挖掘，这也是很耗时的。

与此同时，如何定义“好的”硬三胞胎仍然是一个有待解决的问题。以上所有因素都使三重态损失难以训练。为了克服这些三重态损失的限制，我们将整合它与中心损失(见sec 3.2)并提出了一种新的TCL损失

## **3.2. Review on center loss**

中心损失[39]被建议用于补偿人脸验证中的softmax损失。它为每个类的特征学习一个中心，同时尝试将同一类的深度特征拉近相应的中心，如图2(b).所示

基本上，中心损失可以表述为：

![image-20220812130234158](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812130234158.png)

其中$c_{y_i}∈R^d$为yi类的中心，d为特征的维度。函数D（·）代表欧几里德距离的平方。在训练期间，中心损失鼓励相同类别的实例更接近一个可学习的类别中心。但是，由于参数化中心在每次迭代时都是基于一个小批处理而不是整个数据集来更新的，这是非常不稳定的，因此它在训练过程中必须在softmax损失的联合监督下进行。

## **3.3. The proposed triplet-center loss**

**Motivation.**

虽然联合监督中心损失和softmax损失的目的是为了减少类内的变化，并在人脸识别方面取得了非常有前途的性能，但是，如图2(b)所示，即使类内的变化非常小，类间的集群很可能是重叠的。

这是因为它没有显式地考虑类间的可分性。而对于三联体损失，它直接优化网络的最终任务，但受到三联体构建的复杂性。

受这两个代表性损失的激励，我们提出了三重中心损失，以便有效地学习更多的鉴别特征。

**Defifinitions.**

TCL的目标是利用三重损失和中心损失的优势，即有效地最小化深度学习特征的类内距离，同时最大化深度特征的类间距离。

让给定的训练数据集$\{(x_i，y_i)\}^N_{i=1}$由N个样本$x_i∈X$组成，以及相关的标签$y_i∈\{1,2，...，|Y|\}$。

这些样本被嵌入到d维向量中，神经网络用$f_θ(.)$表示。

在TCL中，我们假设来自同一类的三维形状的特征共享一个相应的中心。

因此，我们可以得到$C=\{c_1，c_2，...，c_{|Y|}\}$，其中$c_y∈R^d$为带有标签为y的样本的中心向量，|Y|为中心数。

为简单起见，我们在本文中采用$f_i$来表示$f(x_i)$。与中心损失类似，我们在每次迭代中基于一个小批处理来更新参数化中心。

给定一批带有M个样本的训练数据，我们将TCL定义为

![image-20220812131300610](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812131300610.png)

![image-20220812131411919](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812131411919.png)

如图2(c)所示，TCL是将样本与其相应中心$c_y^i$之间的距离推近于样本与其最近的负中心(即其他类$C\{c_y^i\}$)的中心之间的距离通过一个margin m。

计算输入特征嵌入的反向传播梯度和相应的中心，我们假设使用以下符号来进行演示：

$\Bbb 1[condition]$是一个指标函数如果条件满足输出1否则输出0

$q_i= argmin_{j \ne y^j} D(f_i,c_j)$是一个整数指数表示第i个样本的最近的负中心，

$\widetilde L_i$表示第i个样本的三中心损失

![image-20220812132742181](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812132742181.png)

然后，我们的TCL损失的导数(等式3)关于第i个样本$\Large \frac {∂L_{tc}}{∂f_i}$和第j个中心$\Large \frac {∂L_{tc}}{∂c_j}$的特征嵌入，可以计算如下：

![image-20220812133007583](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812133007583.png)

![image-20220812133103951](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812133103951.png)

**Joint supervision with softmax loss.**

Softmax损失侧重于将样本映射到离散的标签上，而TCL的目标是将度量学习直接应用到学习到的嵌入中。与中心损耗不同，TCL可以独立于softmax损耗而使用。

然而，根据我们在Sec 4中的实验，这两种损失也可以结合在一起，以实现更有区别和鲁棒的嵌入。它可以被写成

![image-20220812134136620](D:\文献阅读\Recent deveopments of CBIR\image\image-20220812134136620.png)

λ是一个超参数，它控制着TCL和softmax损失之间的权衡。我们将softmax损失带来的好处归因于TCL的参数中心是随机初始化和更新的，基于小批而不是整个数据集，这可能是棘手的，而softmax损失可以作为寻找更好的类中心的更好的指导。

## **3.4. Discussion**

**Compared with triplet loss.**

与由三重样本(xia、xi+、xi−)不同，TCL的三联体由第i个样本xi、其对应的中心cyi和最近的负中心组成。对于有N个样本的训练数据集，TCL只形成N个三联体，而三联体损失的三联体数为O(N3)，远远多于TCL。

因此，与三联体损失相比，TCL避免了构建三联体的复杂性和挖掘硬样本的必要性。我们提供了两个损失的实证分析，4.1并使用t-SNE可视化学习的嵌入。

**Compared with center loss.**

TCL可以作为三重损失的变体，可以利用监督训练神经网络独立于softmax损失，而中心损失必须与softmax结合使学习可行 否则深入学习特性和中心将根据[39]降为零。

此外，TCL同时明确地最大化了类内紧凑性和类间可分性，而中心损失忽略了后者，这可能导致类间重叠。

此外，中心损失的目的是减少样本与其相应中心之间的绝对距离，而TCL则以铰链式损失惩罚相对距离，更放松，更容易训练。

关于TCL损失和中心损失的实证分析，见Sec4.1的细节。

# **4. Experiments**

在本节中，我们评估了所提出的TCL在两种三维形状检索任务上的性能：通用的三维形状检索任务和基于草图的三维形状检索任务。前者是一个域内检索任务，其中给定的查询和数据库中的示例都是三维模型。虽然后者是一个跨域检索任务，但给定的查询则是2D草图。



## **4.1. Generic 3D shape retrieval task**

**Datasets.**

我们在两个著名的3D形状基准上评估了我们的TCL损失：ModelNet40[40]和形状网核心55[27]。ModelNet40数据集包含了来自40个常见类别的12,311个CAD模型。对于这个数据集，我们遵循之前[33,1]关于训练和测试分割设置的工作，即每个类别随机选择100个模型，其中80个模型作为训练数据，其余用于测试。ShapeNet Core 55数据集[27]共由55个类别中的51,190个三维形状组成，并进一步分为204个子类别。由于其类别的多样性和类内的巨大变化，ShapeNet55数据集是相当具有挑战性的。整个数据集分为训练集、验证集和测试集，分别包含35,765、5、159、10、266个模型。此外，该数据集有两个变体(ShapeNet 55扰动数据集和ShapeNet 55正常数据集)。对于普通数据集，形状是对齐的。对于数据集的扰动版本，每个模型都是随机旋转一个角度的。我们在更具挑战性的扰动数据集上进行了实验。所有的检索性能都被报告在测试集上。

**Implementation Details.**

实验代码用PyTorch http://pytorch.org/实现，并在一个带有四台NvidiaTitan-xgpu、一个Intel i7 CPU和64GB内存的服务器上执行。我们选择具有批归一化[32]2的VGG-A作为我们的基础网络。VGG-A包含8个卷积层（conv1-8），内核为3x3和3个全连接层（fc9-11）。视图池层被放置在conv6之后。视图池化层前后的层分别记为CNN1和CNN2。我们用高斯分布初始化中心，其均值和标准差分别为（0,0.01）。为了优化，我们对所有实验采用16的小批量随机梯度下降算法。优化CNN1和CNN2的学习速率分别设置为1e-4和1e-3。

实验代码用PyTorch http://pytorch.org/实现，并在一个带有四台NvidiaTitan-xgpu、一个Intel i7 CPU和64GB内存的服务器上执行。我们选择具有批归一化[32]2的VGG-A作为我们的基础网络。VGG-A包含8个卷积层（conv1-8），内核为3x3和3个全连接层（fc9-11）。视图池层被放置在conv6之后。视图池化层前后的层分别记为CNN1和CNN2。我们用高斯分布初始化中心，其均值和标准差分别为（0,0.01）。为了优化，我们对所有实验采用16的小批量随机梯度下降算法。优化CNN1和CNN2的学习速率分别设置为1e-4和1e-3。

对于CNN1和CNN2，权重衰减为1e-4，动量为0.9。在我们的实验中，中心学习率被设置为0.1

我们把中心的梯度剪成0.01。在测试过程中，我们提取了4096维的fc-10的输出，作为我们所有检索任务的特征。