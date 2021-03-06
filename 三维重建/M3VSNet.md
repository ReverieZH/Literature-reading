与传统的MVS方法相比，目前采用的基于监督学习的网络的多视点立体声(MVS)方法具有显著的性能。然而，用于训练的地面-真实深度图很难获得，而且是在有限的情况下。\

在本文中，我们提出了一种新的无监督多度量MVS网络，名为M3VSNet，用于没有任何监督的密集点云重建。

为了提高点云重构的鲁棒性和完整性，我们提出了一种新的多度量损失函数，该函数结合了像素级和特征级损失函数，从匹配对应的不同角度学习固有的约束条件。此外，我们还在三维点云格式中加入了正常深度的一致性，以提高估计深度图的准确性和连续性。

实验结果表明，M3VSNet建立了最先进的无监督方法，在DTU数据集上取得了与之前的监督MVSNet相当的性能，并在坦克和模板基准上证明了强大的泛化能力和有效的改进。

一、引言

基于监督学习的MVS方法取得了显著的进展，特别是提高了密集点云重建的效率和完整性

这些基于学习的方法通过学习和推断信息来处理立体对应难以获得的匹配模糊度。

然而，这些基于监督学习的方法强烈依赖于具有地面真实深度图的训练数据集，这些数据集的场景种类有限，也不容易获得。

在本文中，我们提出了一种新的无监督多度量MVS网络，名为M3VSNet，如图1所示，即使在非理想环境下，它也可以推断出密集点云重建的深度图。最重要的是，我们提出了一种新的多度量损失函数，即像素级和特征级损失函数。关键的见解是，人类的视觉系统通过物体特征[4]来感知周围的世界。在该损失函数方面，可以很好地保证光度和几何匹配的一致性，与MVSNet中唯一使用的光度约束相比，其更准确和鲁棒性

具体来说，我们引入了来自预先训练过的VGG16网络的多尺度特征图，作为特征级丢失的重要线索。低级特征表示学习更多的纹理细节，而高级特征学习具有较大的接受域的语义信息。不同层次的特征是对不同的感受域的表现。通过聚合多尺度特征，我们提出的M3VSNet可以同时考虑低级图像纹理和高级语义信息。因此，该网络可以很好地提高匹配对应的鲁棒性和准确性。与在无纹理、反射、反射和纹理重复区[24,30,31]等具有挑战性的情况下只使用失配误差的网络相比，M3VSNet可以通过考虑多尺度语义特征之间的相似性来提高鲁棒性。





金字塔特征聚合从具有更多上下文信息的从低级到高级表示中提取特征。然后使用与MVSNet相同的基于方差的成本体积生成和三维U-Net正则化来生成初始深度图。M3VSNet的先进体系结构由正常深度一致性和多度量损失两部分组成。在生成初始深度图后，我们结合新的正线深度一致性，以考虑正线和局部曲面切线之间的正交性。更重要的是，我们构造了多度量损失，它包括像素损失和特征损失。我们将在下面的几个部分中简要地描述每个模块。