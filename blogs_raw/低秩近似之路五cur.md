---
title: 低秩近似之路（五）：CUR
slug: 低秩近似之路五cur
date: 2025-01-12
tags: 近似, 最优, 矩阵, 低秩, 生成模型
status: completed
---

# 低秩近似之路（五）：CUR

**原文链接**: [https://spaces.ac.cn/archives/10662](https://spaces.ac.cn/archives/10662)

**发布日期**: 

---

再次回到低秩近似之路上。在[《低秩近似之路（四）：ID》](/archives/10501)中，我们介绍了“插值分解（Interpolative Decomposition，ID）”，这是为矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$寻找$\boldsymbol{C}\boldsymbol{Z}$形式的近似的过程，其中$\boldsymbol{C}\in\mathbb{R}^{n\times r}$是矩阵$\boldsymbol{M}$的若干列，而$\boldsymbol{Z}\in\mathbb{R}^{r\times m}$是任意矩阵。

这篇文章我们将介绍CUR分解，它跟插值分解的思想一脉相承，都是以原始矩阵的行、列为“骨架”来构建原始矩阵的近似，跟ID只用行或列之一不同，CUR分解同时用到了行和列。

## 基本定义 #

其实这不是本站第一次出现CUR分解了。早在[《Nyströmformer：基于矩阵分解的线性化Attention方案》](/archives/8180)我们就介绍过矩阵的Nyström近似，它实际上就是CUR分解，后来在[《利用CUR分解加速交互式相似度模型的检索》](/archives/9336)还介绍了CUR分解在降低交互式相似度模型的检索复杂度的应用。

CUR分解能有这些应用，关键在于它名称中的“C”和“R”。具体来说，CUR分解试图为矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$寻找如下形式的近似：  
\begin{equation}\mathop{\text{argmin}}_{S_1,S_2,\boldsymbol{\mathcal{U}}}\Vert \underbrace{\boldsymbol{M}_{[:,S_1]}}_{\boldsymbol{\mathcal{C}}}\boldsymbol{\mathcal{U}}\underbrace{\boldsymbol{M}_{[S_2,:]}}_{\boldsymbol{\mathcal{R}}} - \boldsymbol{M}\Vert_F^2\quad\text{s.t.}\quad \left\\{\begin{aligned}&S_1\subset\\{0,1,\cdots,m-1\\},|S_1|=r\\\  
&S_2\subset\\{0,1,\cdots,n-1\\},|S_2|=r \\\  
&\boldsymbol{\mathcal{U}}\in\mathbb{R}^{r\times r}  
\end{aligned}\right.\end{equation}  
为了区分SVD的$\boldsymbol{U}$，这里用了花体的$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{U}},\boldsymbol{\mathcal{R}}$。作为对比，上一篇介绍的ID是  
\begin{equation}\mathop{\text{argmin}}_{S,\boldsymbol{Z}}\Vert \underbrace{\boldsymbol{M}_{[:,S]}}_{\boldsymbol{C}}\boldsymbol{Z} - \boldsymbol{M}\Vert_F^2\quad\text{s.t.}\quad \left\\{\begin{aligned}  
&S\subset\\{0,1,\cdots,m-1\\},|S|=r\\\  
&\boldsymbol{Z}\in\mathbb{R}^{r\times m}  
\end{aligned}\right.\end{equation}  
而SVD找的低秩近似是  
\begin{equation}\mathop{\text{argmin}}_{\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}}\Vert \boldsymbol{U}_{[:,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:,:r]}^{\top} - \boldsymbol{M}\Vert_F^2\quad\text{s.t.}\quad \left\\{\begin{aligned}  
&\boldsymbol{U}\in\mathbb{R}^{n\times n}, \boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}_n \\\  
&\boldsymbol{V}\in\mathbb{R}^{m\times n}, \boldsymbol{V}^{\top}\boldsymbol{V} = \boldsymbol{I}_m \\\  
&\boldsymbol{\Sigma}=\text{diag}(\sigma_1,\cdots,\sigma_{\min(n,m)})\in\mathbb{R}_{\geq 0}^{n\times m}  
\end{aligned}\right.\end{equation}  
在[SVD篇](/archives/10407)我们证明过，SVD可以找到$r$秩近似的最优解，但它本身的计算复杂度高，并且$\boldsymbol{U},\boldsymbol{V}$的物理意义并不直观。相比之下，CUR分解用原本矩阵的列$\boldsymbol{\mathcal{C}}$和行$\boldsymbol{\mathcal{R}}$替代了$\boldsymbol{\mathcal{U}},\boldsymbol{V}$，虽然在近似程度方面不如SVD，但在可解释性、储存成本、计算成本等方面都更优。

从外观上来看，SVD近似的左右矩阵$\boldsymbol{U},\boldsymbol{V}$更复杂而中间矩阵$\boldsymbol{\Sigma}$更简单，而CUR分解则相反，它是左右矩阵左右矩阵$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{R}}$更简单而中间矩阵$\boldsymbol{\mathcal{U}}$更复杂。

## U的选择 #

很明显，CUR分解的难度在于行列的选择，因为当$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{R}}$给定后，$\boldsymbol{\mathcal{U}}$的最优解是可以利用[伪逆](/archives/10366)解析地表示出来：  
\begin{equation}\boldsymbol{\mathcal{U}}^* = \boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}\boldsymbol{\mathcal{R}}^{\dagger}\end{equation}  
求解过程可以参考伪逆篇的推导。其实这个解也很直观，假设$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{R}}$都是可逆矩阵的话，那么方程$\boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}}=\boldsymbol{M}$的解自然是$\boldsymbol{\mathcal{U}}=\boldsymbol{\mathcal{C}}^{-1}\boldsymbol{M}\boldsymbol{\mathcal{R}}^{-1}$，而不可逆的时候就把逆${}^{-1}$换成伪逆${}^{\dagger}$。

除了这个理论最优解外，CUR分解还有一个经常用的、某种意义上更为直观的选择：  
\begin{equation}\boldsymbol{\mathcal{U}} = \boldsymbol{M}_{[S_2,S_1]}^{\dagger}\end{equation}  
注意切片运算的优先级高于转置和伪逆，所以这个$\boldsymbol{\mathcal{U}}$实际上就是$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{R}}$的公共部分组成的子矩阵的伪逆。

怎么理解这个选择呢？通过交换行列，我们可以让被选中的行列都排在前面，并假设$\boldsymbol{M}_{[S_2,S_1]}$可逆，那么该结果可以用分块矩阵写成  
\begin{equation}\underbrace{\begin{pmatrix}\boldsymbol{A} & \boldsymbol{B} \\\ \boldsymbol{C} & \boldsymbol{D}\end{pmatrix}}_{\boldsymbol{M}} \approx \underbrace{\begin{pmatrix}\boldsymbol{A} \\\ \boldsymbol{C}\end{pmatrix}}_{\boldsymbol{\mathcal{C}}}\,\,\underbrace{\boldsymbol{A}^{-1}}_{\boldsymbol{\mathcal{U}}}\,\,\underbrace{\begin{pmatrix}\boldsymbol{A} & \boldsymbol{B}\end{pmatrix}}_{\boldsymbol{\mathcal{R}}} = \begin{pmatrix}\boldsymbol{A} & \boldsymbol{B} \\\ \boldsymbol{C} & \boldsymbol{C}\boldsymbol{A}^{-1}\boldsymbol{B}\end{pmatrix}\label{eq:id-abcd}\end{equation}  
可以看到，此时的CUR分解精确地重建出了选出来的$\boldsymbol{A},\boldsymbol{B},\boldsymbol{C}$（或者说$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{R}}$），并用$\boldsymbol{C}\boldsymbol{A}^{-1}\boldsymbol{B}$来近似$\boldsymbol{D}$，此时的CUR分解相当于一种“矩阵补全（Matrix Completion）”方法。

值得指出的是，由于两个$\boldsymbol{\mathcal{U}}$都用到了伪逆，且伪逆的定义并不要求方阵，所以最一般的CUR分解其实不要求$\boldsymbol{\mathcal{C}}$/$\boldsymbol{\mathcal{R}}$具有相同的列数/行数，如果有必要，我们可以为$\boldsymbol{\mathcal{C}}$/$\boldsymbol{\mathcal{R}}$选择不同数量的列/行。

## 行列选择 #

解决完$\boldsymbol{\mathcal{U}}$之后，下面主要就是$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{R}}$的选择了。由于行、列的选择本质上是等价的，所以下面我们以列的选择为例。

也就是说，下面我们的任务就是从矩阵$\boldsymbol{M}$中选出$r$个关键列，作为它的“骨架”，也可以叫做“轮廓”、“草图”等，这个问题我们其实在上两篇文章（即[CR篇](/archives/10427)和[ID篇](/archives/10501)）已经探究过，里边的方案也可以用来构建CUR分解的$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{R}}$，包括

> 1、选择模长最大的$r$列；
> 
> 2、以模长为权随机采样$r$列；
> 
> 3、均匀随机采样$r$列；
> 
> 4、按列驱QR分解选择前$r$列。

这些方案各有优劣，都有它们的适用场景和隐含假设。除此之外，我们也可以考虑一些更直观的做法，比如考虑到关键列的含义似乎跟“聚类中心”相似，所以我们可以将$n$个列向量聚成$k$类，然后选择距离聚类中心最近的$k$个向量。当$n$实在太大时，又可以先随机抽取一部分，然后再在这些向量中执行上述选择算法。

总的来说，列选择是矩阵近似中的一个经典问题，英文关键词是Randomized Linear Algebra、Column Subset Selection等，大家一搜就可以找到很多资料。

## 杠杆分数 #

当然，作为一篇新文章，最好还是要介绍一些新方法，所以接下来我们再介绍另外两种列选择的思路。第一种我们称为“杠杆分数（Leverage Scores）”，它是通过线性回归的思想来进行列选择。

首先，我们将矩阵$\boldsymbol{M}$视为$m$个$n$维样本，然后相应地有$m$个$d$维向量构成目标矩阵$\boldsymbol{Y}$，我们的任务是用$\boldsymbol{M}$预测$\boldsymbol{Y}$，模型采用最简单的线性模型，优化目标是最小二乘  
\begin{equation}\boldsymbol{W}^* = \mathop{\text{argmin}}_{\boldsymbol{W}} \Vert\boldsymbol{Y} - \boldsymbol{W}\boldsymbol{M}\Vert_F^2\label{eq:linear-loss}\end{equation}  
这个目标我们在伪逆篇已经解决过，答案是$\boldsymbol{W}^* = \boldsymbol{Y}\boldsymbol{M}^{\dagger}$，假设$n < m$且$\boldsymbol{M}$的秩为$n$，那么可以进一步写出$\boldsymbol{W}^* = \boldsymbol{Y}\boldsymbol{M}^{\top}(\boldsymbol{M}\boldsymbol{M}^{\top})^{-1}$，于是我们有  
\begin{equation}\hat{\boldsymbol{Y}} = \boldsymbol{W}^*\boldsymbol{M} = \boldsymbol{Y}\boldsymbol{M}^{\top}(\boldsymbol{M}\boldsymbol{M}^{\top})^{-1}\boldsymbol{M} = \boldsymbol{Y}\boldsymbol{H}\end{equation}  
这里$\boldsymbol{H}=\boldsymbol{M}^{\top}(\boldsymbol{M}\boldsymbol{M}^{\top})^{-1}\boldsymbol{M}$称为“帽子矩阵（Hat Matrix）”，据说是因为它将$\boldsymbol{Y}$变成$\hat{\boldsymbol{Y}}$，就像是给$\boldsymbol{Y}$带上了一定帽子（即$\hat{}$）。设$\boldsymbol{m}_i$是$\boldsymbol{M}$的第$i$个列向量，在这里也就是第$i$个样本，那么我们认为  
\begin{equation}\boldsymbol{H}_{i,i} = \boldsymbol{m}_i^{\top}(\boldsymbol{M}\boldsymbol{M}^{\top})^{-1}\boldsymbol{m}_i\end{equation}  
衡量了该样本在预测$\hat{\boldsymbol{Y}}$时的作用，这就是“杠杆分数（Leverage Scores）”。我们认为选出$r$个关键列，就相当于要选出$r$个最重要的样本，于是可以选择$\boldsymbol{H}_{i,i}$最大的$r$列。

当$\boldsymbol{M}\boldsymbol{M}^{\top}$不可逆时，论文[《Input Sparsity Time Low-Rank Approximation via Ridge Leverage Score Sampling》](https://papers.cool/arxiv/1511.07263)将Leverage Scores推广为“Ridge Leverage Score”，实际就是在目标$\eqref{eq:linear-loss}$基础上加了个正则项使其可逆。但实际上我们知道，伪逆的概念是不要求满秩的，所以可以直接通过SVD来计算伪逆，这就不需要额外引入正则项了。

设$\boldsymbol{M}$的SVD为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，那么  
\begin{equation}\boldsymbol{H} = \boldsymbol{M}^{\dagger}\boldsymbol{M} = (\boldsymbol{V}\boldsymbol{\Sigma}^{\dagger}\boldsymbol{U}^{\top})(\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}) = \boldsymbol{V}(\boldsymbol{\Sigma}^{\dagger}\boldsymbol{\Sigma})\boldsymbol{V}^{\top}\end{equation}  
假设$\boldsymbol{M}$的秩为$\gamma$（$\gamma$不是$r$），那么按照伪逆的计算规则，$\boldsymbol{\Sigma}^{\dagger}\boldsymbol{\Sigma}$是一个$m\times m$的对角矩阵，对角线上前$\gamma$个元素全是1，其余是0，所以  
\begin{equation}\boldsymbol{H} = \boldsymbol{V}_{[:,:\gamma]}\boldsymbol{V}_{[:,:\gamma]}^{\top}\quad\Rightarrow\quad \boldsymbol{H}_{i,i} = \Vert\boldsymbol{V}_{[i-1,:\gamma]}\Vert^2\end{equation}  
注意，$\boldsymbol{H}_{i,i}$表示$\boldsymbol{H}$的第$i$行、$i$列元素，计数从1开始，但切片的规则按照Python来，计数从0开始，所以最后的切片是${}_{[i-1,:\gamma]}$。现在我们看到，要计算$\boldsymbol{M}$的列的杠杆分数，只需要将它SVD为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，然后计算$\boldsymbol{V}_{[:,:\gamma]}$各行的模长平方；同理，要计算行的杠杆分数，则只需要计算$\boldsymbol{U}_{[:,:\gamma]}$各行的模长平方。

由于$\boldsymbol{V}$本身是正交矩阵，因此恒成立  
\begin{equation}\sum_{i=1}^m \boldsymbol{H}_{i,i} = \sum_{i=1}^m\Vert\boldsymbol{V}_{[i-1,:\gamma]}\Vert^2 = \gamma\end{equation}  
因此除了选杠杆分数最大的$r$列之外，还可以构造分布$p_i = \boldsymbol{H}_{i,i} / \gamma$来随机采样。

杠杆分数跟$\boldsymbol{M}$的秩$\gamma$相关，而$\boldsymbol{M}$的秩等于$\boldsymbol{M}$的非零奇异值个数，所以它会被接近零的奇异值影响，但这些偏小的奇异值实际作用不大，所以实践中$\gamma$一般取$\boldsymbol{M}$较显著的奇异值（主奇异值）个数。杠杆分数的另一个问题是需要先SVD，实践中多数会采用一些近似SVD算法。至于近似SVD算法的内容，我们后面有机会再谈了。

## DEIM法 #

另外一种要介绍的列选择方法叫做[DEIM](https://papers.cool/arxiv/1407.5516)，全称是Discrete Empirical Interpolation Method，这里也不去考究这个名称的来源了，但大致上可以确定的是，杠杆分数和DEIM都是CUR分解常用的列选择方法，而且DEIM跟CUR分解的联系更为密切，所以近年来愈加流行。

DEIM的出发点是恒等式$\eqref{eq:id-abcd}$，在该式的$\boldsymbol{\mathcal{C}},\boldsymbol{\mathcal{U}},\boldsymbol{\mathcal{R}}$之下，CUR分解的误差取决于$\Vert \boldsymbol{D} - \boldsymbol{C}\boldsymbol{A}^{-1}\boldsymbol{B}\Vert_F$。什么时候这个式子会小呢？直观的想法是$\boldsymbol{A}$比较大，$\boldsymbol{B},\boldsymbol{C},\boldsymbol{D}$都比较小时，整个式子肯定也很小。但$\boldsymbol{A}$是一个矩阵，怎么衡量大小呢？行列式的绝对值可以作为一个参考指标。所以，一个可行方案是选择让$\boldsymbol{M}_{[S_2,S_1]}$行列式绝对值最大的对应行列。

当然，这个方案只有理论价值，因为精确找到行列式绝对值最大的子矩阵也是NP-Hard的，但它提供了一个目标，我们可以尝试找一个贪心解，当$r=1$时，找绝对值最大的行列式也就是找绝对值最大的元素，这是可以接受的，然后可以递归下去。DEIM沿用了这个思路，但它不是从$\boldsymbol{M}$出发，而是借鉴了杠杆分数的做法，从SVD之后的$\boldsymbol{V}$出发。

杠杆分数将 _从$\boldsymbol{M}$找关键列_ 转化为 _从$\boldsymbol{V}_{[:,:\gamma]}$找关键行_ ，排序指标是行模长平方，DEIM则试图通过为$\boldsymbol{V}$找CUR近似来找关键行。可$\boldsymbol{M}$的CUR还没解决，现在又来一个$\boldsymbol{V}$的CUR，不是越搞越复杂？不会，这里还是简化一点的，因为$\boldsymbol{V}$是$\boldsymbol{M}$的SVD结果，它按照奇异值大小排序过了，所以我们可以认为$\boldsymbol{V}$的列已经按重要性排好序了，因此最重要的$r$列必然是$\boldsymbol{V}_{[:,:r]}$，我们只需要选择行。

如前面所述，求解思路是贪心算法，最重要的列自然是第一列$\boldsymbol{V}_{[:,0]}$，那最重要的行呢？我们要选择它跟第一列相交的行列时绝对值最大的那一行，说白了就是第一列绝对值对大的元素所在的那一行，这样我们就有了初始行列。假设我们已经选出了$k$个关键行，下标集为$S_k$，那怎么选出第$k+1$关键行呢？首先，我们知道由 _已选出的$k$行_ 和 _前$k$列_ 构建的CUR近似是$\boldsymbol{V}_{[:,:k]}\boldsymbol{V}_{[S_k,:k]}^{-1}\boldsymbol{V}_{[S_k,:]}$，第$k+1$列的误差是  
\begin{equation}\boldsymbol{V}_{[:,k]} - \boldsymbol{V}_{[:,:k]}\boldsymbol{V}_{[S_k,:k]}^{-1}\boldsymbol{V}_{[S_k,k]}\end{equation}  
由式$\eqref{eq:id-abcd}$我们知道此种CUR近似能恢复所选的行和列，所以上式中已被选出的$k$行对应的分量必然为零，因此剩余的绝对值最大的非零元素必然不在已选出的$k$行之中，我们选择它的所在行作为第$k+1$个关键行。

简而言之，DEIM利用奇异值分解已经为$\boldsymbol{V}$的列向量排好序的特点，将CUR分解转化为一个单纯的行搜索问题，减少了搜索方向，然后通过贪心算法求解，每一步的选择依据是误差最大元素所在行。更详细的介绍和证明可以参考[《A DEIM Induced CUR Factorization》](https://papers.cool/arxiv/1407.5516)和[《CUR Matrix Factorizations: Algorithms, Analysis, Applications》](https://personal.math.vt.edu/embree/cur_talk.pdf)。

## 文章小结 #

本文介绍了CUR分解，它可以视为上一篇文章介绍的插值分解（ID）的进一步延伸，特点是同时以原始矩阵的若干行与列作为骨架来构建低秩近似。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10662>_

_**更详细的转载事宜请参考：**_[《科学空间FAQ》](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《科学空间FAQ》")

**如果您还有什么疑惑或建议，欢迎在下方评论区继续讨论。**

**如果您觉得本文还不错，欢迎分享/打赏本文。打赏并非要从中获得收益，而是希望知道科学空间获得了多少读者的真心关注。当然，如果你无视它，也不会影响你的阅读。再次表示欢迎和感谢！**

打赏

![科学空间](https://spaces.ac.cn/usr/themes/geekg/payment/wx.png)

微信打赏

![科学空间](https://spaces.ac.cn/usr/themes/geekg/payment/zfb.png)

支付宝打赏

因为网站后台对打赏并无记录，因此欢迎在打赏时候备注留言。你还可以[**点击这里**](http://mail.qq.com/cgi-bin/qm_share?t=qm_mailme&email=tN7d1drY3drrx8H0xcWa19vZ)或在下方评论区留言来告知你的建议或需求。

**如果您需要引用本文，请参考：**

苏剑林. (Jan. 12, 2025). 《低秩近似之路（五）：CUR 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10662>

@online{kexuefm-10662,  
title={低秩近似之路（五）：CUR},  
author={苏剑林},  
year={2025},  
month={Jan},  
url={\url{https://spaces.ac.cn/archives/10662}},  
} 


---

## 公式推导与注释

本节将详细推导CUR分解的数学理论，包括定义、算法、误差分析和应用。所有重要公式都使用编号标记。

### 1. CUR分解的形式化定义

**定义1.1 (CUR分解)**: 给定矩阵 $\boldsymbol{M}\in\mathbb{R}^{n\times m}$，其CUR分解定义为：

\begin{equation}
\boldsymbol{M} \approx \boldsymbol{\mathcal{C}} \boldsymbol{\mathcal{U}} \boldsymbol{\mathcal{R}}
\tag{1}
\end{equation}

其中：
- $\boldsymbol{\mathcal{C}} \in \mathbb{R}^{n \times r}$ 是从 $\boldsymbol{M}$ 中选择的 $r$ 列组成的矩阵
- $\boldsymbol{\mathcal{R}} \in \mathbb{R}^{r \times m}$ 是从 $\boldsymbol{M}$ 中选择的 $r$ 行组成的矩阵
- $\boldsymbol{\mathcal{U}} \in \mathbb{R}^{r \times r}$ 是中间系数矩阵

**优化目标**: CUR分解的优化问题可形式化为：

\begin{equation}
\min_{S_1, S_2, \boldsymbol{\mathcal{U}}} \|\boldsymbol{M}_{[:,S_1]} \boldsymbol{\mathcal{U}} \boldsymbol{M}_{[S_2,:]} - \boldsymbol{M}\|_F^2
\tag{2}
\end{equation}

约束条件：
\begin{equation}
\begin{cases}
S_1 \subset \{1,2,\ldots,m\}, \quad |S_1| = r \\
S_2 \subset \{1,2,\ldots,n\}, \quad |S_2| = r \\
\boldsymbol{\mathcal{U}} \in \mathbb{R}^{r \times r}
\end{cases}
\tag{3}
\end{equation}

### 2. $\boldsymbol{\mathcal{U}}$ 的最优解推导

**命题2.1**: 当 $\boldsymbol{\mathcal{C}}$ 和 $\boldsymbol{\mathcal{R}}$ 固定时，$\boldsymbol{\mathcal{U}}$ 的最优解为：

\begin{equation}
\boldsymbol{\mathcal{U}}^* = \boldsymbol{\mathcal{C}}^{\dagger} \boldsymbol{M} \boldsymbol{\mathcal{R}}^{\dagger}
\tag{4}
\end{equation}

其中 $\dagger$ 表示Moore-Penrose伪逆。

**证明**: 优化目标可以写为：

\begin{equation}
f(\boldsymbol{\mathcal{U}}) = \|\boldsymbol{\mathcal{C}} \boldsymbol{\mathcal{U}} \boldsymbol{\mathcal{R}} - \boldsymbol{M}\|_F^2
\tag{5}
\end{equation}

展开Frobenius范数：

\begin{equation}
f(\boldsymbol{\mathcal{U}}) = \text{tr}\left[(\boldsymbol{\mathcal{C}} \boldsymbol{\mathcal{U}} \boldsymbol{\mathcal{R}} - \boldsymbol{M})^T (\boldsymbol{\mathcal{C}} \boldsymbol{\mathcal{U}} \boldsymbol{\mathcal{R}} - \boldsymbol{M})\right]
\tag{6}
\end{equation}

展开并利用迹的循环性质：

\begin{equation}
\begin{aligned}
f(\boldsymbol{\mathcal{U}}) &= \text{tr}(\boldsymbol{\mathcal{R}}^T \boldsymbol{\mathcal{U}}^T \boldsymbol{\mathcal{C}}^T \boldsymbol{\mathcal{C}} \boldsymbol{\mathcal{U}} \boldsymbol{\mathcal{R}}) - 2\text{tr}(\boldsymbol{\mathcal{R}}^T \boldsymbol{\mathcal{U}}^T \boldsymbol{\mathcal{C}}^T \boldsymbol{M}) + \text{tr}(\boldsymbol{M}^T \boldsymbol{M})
\end{aligned}
\tag{7}
\end{equation}

对 $\boldsymbol{\mathcal{U}}$ 求导并令其为零（使用矩阵求导法则 $\frac{\partial}{\partial \boldsymbol{X}} \text{tr}(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B}) = \boldsymbol{A}^T \boldsymbol{B}^T$）：

\begin{equation}
\frac{\partial f}{\partial \boldsymbol{\mathcal{U}}} = 2\boldsymbol{\mathcal{C}}^T \boldsymbol{\mathcal{C}} \boldsymbol{\mathcal{U}} \boldsymbol{\mathcal{R}} \boldsymbol{\mathcal{R}}^T - 2\boldsymbol{\mathcal{C}}^T \boldsymbol{M} \boldsymbol{\mathcal{R}}^T = 0
\tag{8}
\end{equation}

整理得到正规方程：

\begin{equation}
\boldsymbol{\mathcal{C}}^T \boldsymbol{\mathcal{C}} \boldsymbol{\mathcal{U}} \boldsymbol{\mathcal{R}} \boldsymbol{\mathcal{R}}^T = \boldsymbol{\mathcal{C}}^T \boldsymbol{M} \boldsymbol{\mathcal{R}}^T
\tag{9}
\end{equation}

利用伪逆的性质 $\boldsymbol{A}^{\dagger} = (\boldsymbol{A}^T \boldsymbol{A})^{-1} \boldsymbol{A}^T$（当 $\boldsymbol{A}$ 列满秩时），我们得到：

\begin{equation}
\boldsymbol{\mathcal{U}}^* = (\boldsymbol{\mathcal{C}}^T \boldsymbol{\mathcal{C}})^{-1} \boldsymbol{\mathcal{C}}^T \boldsymbol{M} \boldsymbol{\mathcal{R}}^T (\boldsymbol{\mathcal{R}} \boldsymbol{\mathcal{R}}^T)^{-1} = \boldsymbol{\mathcal{C}}^{\dagger} \boldsymbol{M} \boldsymbol{\mathcal{R}}^{\dagger}
\tag{10}
\end{equation}

**几何解释**: 公式(4)的几何意义是：$\boldsymbol{\mathcal{U}}$ 通过最小二乘方法将 $\boldsymbol{M}$ 从列空间 $\text{col}(\boldsymbol{\mathcal{C}})$ 和行空间 $\text{row}(\boldsymbol{\mathcal{R}})$ 的角度进行最佳逼近。

### 3. 交叉子矩阵方法

**命题3.1**: CUR分解的另一个常用选择是：

\begin{equation}
\boldsymbol{\mathcal{U}} = \boldsymbol{M}_{[S_2, S_1]}^{\dagger}
\tag{11}
\end{equation}

其中 $\boldsymbol{M}_{[S_2, S_1]}$ 是 $\boldsymbol{\mathcal{C}}$ 和 $\boldsymbol{\mathcal{R}}$ 的交叉子矩阵。

**分块矩阵表示**: 不失一般性，通过行列重排，设选中的行列排在前面：

\begin{equation}
\boldsymbol{M} = \begin{pmatrix}
\boldsymbol{A} & \boldsymbol{B} \\
\boldsymbol{C} & \boldsymbol{D}
\end{pmatrix}
\tag{12}
\end{equation}

其中 $\boldsymbol{A} \in \mathbb{R}^{r \times r}$ 是交叉子矩阵，则CUR近似为：

\begin{equation}
\boldsymbol{M} \approx \begin{pmatrix}
\boldsymbol{A} \\
\boldsymbol{C}
\end{pmatrix} \boldsymbol{A}^{\dagger} \begin{pmatrix}
\boldsymbol{A} & \boldsymbol{B}
\end{pmatrix} = \begin{pmatrix}
\boldsymbol{A}\boldsymbol{A}^{\dagger}\boldsymbol{A} & \boldsymbol{A}\boldsymbol{A}^{\dagger}\boldsymbol{B} \\
\boldsymbol{C}\boldsymbol{A}^{\dagger}\boldsymbol{A} & \boldsymbol{C}\boldsymbol{A}^{\dagger}\boldsymbol{B}
\end{pmatrix}
\tag{13}
\end{equation}

当 $\boldsymbol{A}$ 满秩时，$\boldsymbol{A}\boldsymbol{A}^{\dagger} = \boldsymbol{I}_r$，因此：

\begin{equation}
\boldsymbol{M} \approx \begin{pmatrix}
\boldsymbol{A} & \boldsymbol{B} \\
\boldsymbol{C} & \boldsymbol{C}\boldsymbol{A}^{-1}\boldsymbol{B}
\end{pmatrix}
\tag{14}
\end{equation}

**解释**:
- 选中的行列 $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}$ 被**精确重构**
- 未选中部分 $\boldsymbol{D}$ 通过 $\boldsymbol{C}\boldsymbol{A}^{-1}\boldsymbol{B}$ 进行**插值近似**
- 这种方法相当于一种**矩阵补全**策略

### 4. 与SVD的关系

**定理4.1 (SVD的最优性)**: 设 $\boldsymbol{M}$ 的SVD分解为：

\begin{equation}
\boldsymbol{M} = \sum_{i=1}^{\min(n,m)} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^T = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T
\tag{15}
\end{equation}

则秩-$r$ 的最优逼近为：

\begin{equation}
\boldsymbol{M}_r^* = \sum_{i=1}^{r} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^T = \boldsymbol{U}_{:,1:r} \boldsymbol{\Sigma}_{1:r,1:r} \boldsymbol{V}_{:,1:r}^T
\tag{16}
\end{equation}

满足：

\begin{equation}
\|\boldsymbol{M} - \boldsymbol{M}_r^*\|_F^2 = \sum_{i=r+1}^{\min(n,m)} \sigma_i^2
\tag{17}
\end{equation}

**CUR与SVD的误差比较**: 定义近似比率：

\begin{equation}
\rho = \frac{\|\boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}}\|_F^2}{\|\boldsymbol{M} - \boldsymbol{M}_r^*\|_F^2}
\tag{18}
\end{equation}

理论上 $\rho \geq 1$，好的CUR算法能使 $\rho$ 接近1。

**命题4.2**: 对于良好选择的 $\boldsymbol{\mathcal{C}}, \boldsymbol{\mathcal{R}}$，存在常数 $C$ 使得：

\begin{equation}
\|\boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}\boldsymbol{\mathcal{R}}^{\dagger}\boldsymbol{\mathcal{R}}\|_F \leq C \cdot \|\boldsymbol{M} - \boldsymbol{M}_r^*\|_F
\tag{19}
\end{equation}

### 5. 杠杆分数采样理论

**定义5.1 (帽子矩阵)**: 对于 $\boldsymbol{M} \in \mathbb{R}^{n \times m}$ ($n < m$)，列帽子矩阵定义为：

\begin{equation}
\boldsymbol{H}_{\text{col}} = \boldsymbol{M}^T (\boldsymbol{M}\boldsymbol{M}^T)^{\dagger} \boldsymbol{M}
\tag{20}
\end{equation}

行帽子矩阵定义为：

\begin{equation}
\boldsymbol{H}_{\text{row}} = \boldsymbol{M}^T (\boldsymbol{M}^T\boldsymbol{M})^{\dagger} \boldsymbol{M}
\tag{21}
\end{equation}

**几何意义**: $\boldsymbol{H}_{\text{col}}$ 将向量投影到 $\boldsymbol{M}$ 的列空间。设线性回归问题为：

\begin{equation}
\min_{\boldsymbol{W}} \|\boldsymbol{Y} - \boldsymbol{W}\boldsymbol{M}\|_F^2
\tag{22}
\end{equation}

最优解为 $\boldsymbol{W}^* = \boldsymbol{Y}\boldsymbol{M}^{\dagger}$，预测值为：

\begin{equation}
\hat{\boldsymbol{Y}} = \boldsymbol{W}^* \boldsymbol{M} = \boldsymbol{Y} \boldsymbol{M}^{\dagger} \boldsymbol{M} = \boldsymbol{Y} \boldsymbol{H}_{\text{col}}
\tag{23}
\end{equation}

**定义5.2 (杠杆分数)**: 第 $j$ 列的杠杆分数定义为：

\begin{equation}
\ell_j = [\boldsymbol{H}_{\text{col}}]_{jj} = \boldsymbol{m}_j^T (\boldsymbol{M}\boldsymbol{M}^T)^{\dagger} \boldsymbol{m}_j
\tag{24}
\end{equation}

其中 $\boldsymbol{m}_j$ 是 $\boldsymbol{M}$ 的第 $j$ 列。

**命题5.3 (杠杆分数的SVD表示)**: 设 $\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T$，秩为 $\gamma$，则：

\begin{equation}
\boldsymbol{H}_{\text{col}} = \boldsymbol{M}^{\dagger} \boldsymbol{M} = \boldsymbol{V}\boldsymbol{\Sigma}^{\dagger}\boldsymbol{\Sigma}\boldsymbol{V}^T = \boldsymbol{V}_{:,1:\gamma} \boldsymbol{V}_{:,1:\gamma}^T
\tag{25}
\end{equation}

因此：

\begin{equation}
\ell_j = \|\boldsymbol{V}_{j,1:\gamma}\|_2^2 = \sum_{i=1}^{\gamma} V_{ji}^2
\tag{26}
\end{equation}

**性质**: 杠杆分数满足：

\begin{equation}
\sum_{j=1}^{m} \ell_j = \text{tr}(\boldsymbol{H}_{\text{col}}) = \gamma
\tag{27}
\end{equation}

**采样策略**:
1. **确定性**: 选择杠杆分数最大的 $r$ 列
2. **随机性**: 按概率分布 $p_j = \ell_j / \gamma$ 采样 $r$ 列

**定理5.4 (杠杆分数采样误差界)**: 若按杠杆分数采样 $r = O(k \log k / \epsilon^2)$ 列（$k$ 是目标秩），则以高概率：

\begin{equation}
\|\boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}\|_F \leq (1+\epsilon) \|\boldsymbol{M} - \boldsymbol{M}_k\|_F
\tag{28}
\end{equation}

### 6. DEIM算法详解

**算法思想**: DEIM (Discrete Empirical Interpolation Method) 基于贪心策略，从SVD的右奇异向量中选择行。

**输入**: 矩阵 $\boldsymbol{M} \in \mathbb{R}^{n \times m}$，目标秩 $r$

**步骤**:

**Step 1**: 计算SVD（或截断SVD）：

\begin{equation}
\boldsymbol{M} \approx \boldsymbol{U}_{:,1:r} \boldsymbol{\Sigma}_{1:r,1:r} \boldsymbol{V}_{:,1:r}^T
\tag{29}
\end{equation}

**Step 2**: 初始化，选择 $\boldsymbol{V}_{:,1}$ 绝对值最大元素的行：

\begin{equation}
i_1 = \arg\max_{i \in \{1,\ldots,m\}} |V_{i,1}|
\tag{30}
\end{equation}

设 $S_1 = \{i_1\}$。

**Step 3**: 对 $k = 2, 3, \ldots, r$，递归选择：

计算残差：

\begin{equation}
\boldsymbol{r}_k = \boldsymbol{V}_{:,k} - \boldsymbol{V}_{:,1:k-1} \boldsymbol{V}_{S_{k-1},1:k-1}^{-1} \boldsymbol{V}_{S_{k-1},k}
\tag{31}
\end{equation}

选择残差最大的行：

\begin{equation}
i_k = \arg\max_{i \notin S_{k-1}} |r_{k,i}|
\tag{32}
\end{equation}

更新 $S_k = S_{k-1} \cup \{i_k\}$。

**Step 4**: 输出列索引 $S_r$，构造：

\begin{equation}
\boldsymbol{\mathcal{C}} = \boldsymbol{M}_{:,S_r}, \quad \boldsymbol{\mathcal{U}} = \boldsymbol{M}_{S_r,S_r}^{\dagger}
\tag{33}
\end{equation}

**定理6.1 (DEIM误差界)**: DEIM选择的列满足：

\begin{equation}
\|\boldsymbol{V}_{:,1:r} - \boldsymbol{V}_{:,1:r}\boldsymbol{V}_{S_r,1:r}^{-1}\boldsymbol{V}_{S_r,:}\|_F \leq \sqrt{r(r+1)} \|\boldsymbol{V}_{:,r+1:}\|_F
\tag{34}
\end{equation}

**贪心准则的直觉**: 在每一步，我们选择当前CUR近似误差最大的位置，确保：
- 选中的列能捕获主要的方差
- 交叉子矩阵 $\boldsymbol{V}_{S_r,1:r}$ 尽可能良好条件（行列式绝对值大）

### 7. 误差分析

**定理7.1 (一般误差界)**: 对于CUR分解 $\boldsymbol{M} \approx \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}}$，近似误差可分解为：

\begin{equation}
\boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}} = (\boldsymbol{I} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger})\boldsymbol{M} + \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}(\boldsymbol{I} - \boldsymbol{\mathcal{R}}^{\dagger}\boldsymbol{\mathcal{R}})
\tag{35}
\end{equation}

当使用 $\boldsymbol{\mathcal{U}} = \boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}\boldsymbol{\mathcal{R}}^{\dagger}$ 时。

**证明**: 展开右边：

\begin{equation}
\begin{aligned}
&(\boldsymbol{I} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger})\boldsymbol{M} + \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}(\boldsymbol{I} - \boldsymbol{\mathcal{R}}^{\dagger}\boldsymbol{\mathcal{R}}) \\
&= \boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M} + \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}\boldsymbol{\mathcal{R}}^{\dagger}\boldsymbol{\mathcal{R}} \\
&= \boldsymbol{M} - \boldsymbol{\mathcal{C}}(\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}\boldsymbol{\mathcal{R}}^{\dagger})\boldsymbol{\mathcal{R}} \\
&= \boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}}
\end{aligned}
\tag{36}
\end{equation}

**推论7.2**: 利用三角不等式：

\begin{equation}
\|\boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}}\|_F \leq \|(\boldsymbol{I} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger})\boldsymbol{M}\|_F + \|\boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{C}}^{\dagger}\boldsymbol{M}(\boldsymbol{I} - \boldsymbol{\mathcal{R}}^{\dagger}\boldsymbol{\mathcal{R}})\|_F
\tag{37}
\end{equation}

第一项是列选择误差，第二项是行选择误差。

**定理7.3 (随机CUR误差界)**: 若按杠杆分数采样 $r = \Omega(k/\epsilon^2)$ 列和行，则以概率至少 $1-\delta$：

\begin{equation}
\|\boldsymbol{M} - \boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}}\|_F \leq (1+\epsilon)\|\boldsymbol{M} - \boldsymbol{M}_k\|_F + \delta \|\boldsymbol{M}\|_F
\tag{38}
\end{equation}

### 8. 算法复杂度分析

**SVD方法**:
- 完整SVD: $O(\min(nm^2, n^2m))$
- 截断SVD (如randomized SVD): $O(nmr)$

**CUR分解复杂度**:

**杠杆分数方法**:
1. 计算截断SVD: $O(nmr)$
2. 计算杠杆分数: $O(mr)$
3. 选择/采样列: $O(m + nr)$
4. 计算 $\boldsymbol{\mathcal{U}}$: $O(nr^2 + r^2m + r^3)$

总计: $O(nmr + r^3)$

**DEIM方法**:
1. 计算截断SVD: $O(nmr)$
2. 贪心选择（$r$ 次迭代，每次 $O(mr)$）: $O(mr^2)$
3. 计算 $\boldsymbol{\mathcal{U}}$: $O(r^3)$

总计: $O(nmr + mr^2)$

**存储复杂度对比**:
- SVD: 需存储 $\boldsymbol{U}_{:,1:r}, \boldsymbol{\Sigma}_{1:r,1:r}, \boldsymbol{V}_{:,1:r}^T$，共 $O(nr + mr + r)$
- CUR: 需存储列索引、行索引和 $\boldsymbol{\mathcal{U}}$，共 $O(nr + mr + r^2)$

### 9. 数值示例

**示例9.1**: 考虑一个低秩矩阵：

\begin{equation}
\boldsymbol{M} = \begin{pmatrix}
1 & 2 & 3 & 4 \\
2 & 4 & 6 & 8 \\
3 & 6 & 9 & 12 \\
4 & 8 & 12 & 16
\end{pmatrix} + \boldsymbol{E}
\tag{39}
\end{equation}

其中 $\boldsymbol{E}$ 是小扰动。精确地，$\boldsymbol{M} = \boldsymbol{v}\boldsymbol{v}^T$，$\boldsymbol{v} = (1,2,3,4)^T$，秩为1。

**SVD**: 主奇异值 $\sigma_1 \approx \sqrt{30}$，对应：

\begin{equation}
\boldsymbol{u}_1 = \frac{1}{\sqrt{30}}(1,2,3,4)^T, \quad \boldsymbol{v}_1 = \frac{1}{\sqrt{30}}(1,2,3,4)^T
\tag{40}
\end{equation}

秩-1近似：$\boldsymbol{M}_1 = \sigma_1 \boldsymbol{u}_1 \boldsymbol{v}_1^T = \boldsymbol{M}$（无扰动时）。

**CUR** (选择第1列和第1行):

\begin{equation}
\boldsymbol{\mathcal{C}} = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}, \quad
\boldsymbol{\mathcal{R}} = \begin{pmatrix} 1 & 2 & 3 & 4 \end{pmatrix}
\tag{41}
\end{equation}

\begin{equation}
\boldsymbol{\mathcal{U}} = (\boldsymbol{\mathcal{C}}^T \boldsymbol{\mathcal{C}})^{-1} \boldsymbol{\mathcal{C}}^T \boldsymbol{M} \boldsymbol{\mathcal{R}}^T (\boldsymbol{\mathcal{R}} \boldsymbol{\mathcal{R}}^T)^{-1} = \frac{1}{30} \cdot 30 \cdot \frac{1}{30} = 1
\tag{42}
\end{equation}

CUR近似：

\begin{equation}
\boldsymbol{\mathcal{C}}\boldsymbol{\mathcal{U}}\boldsymbol{\mathcal{R}} = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix} \cdot 1 \cdot \begin{pmatrix} 1 & 2 & 3 & 4 \end{pmatrix} = \boldsymbol{M}
\tag{43}
\end{equation}

完美重构！

**示例9.2 (杠杆分数计算)**: 对于

\begin{equation}
\boldsymbol{M} = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{pmatrix}
\tag{44}
\end{equation}

SVD给出 $\boldsymbol{V} = \boldsymbol{I}_3$（前3个奇异值为1），因此：

\begin{equation}
\ell_1 = \|\boldsymbol{V}_{1,:}\|^2 = 1, \quad \ell_2 = 1, \quad \ell_3 = 1
\tag{45}
\end{equation}

所有列的杠杆分数相等，随机采样时每列被选中的概率相同。

### 10. 应用场景

**应用10.1 (推荐系统)**: 用户-物品评分矩阵 $\boldsymbol{R} \in \mathbb{R}^{n_u \times n_i}$，CUR分解：
- $\boldsymbol{\mathcal{C}}$: 代表性物品
- $\boldsymbol{\mathcal{R}}$: 代表性用户
- $\boldsymbol{\mathcal{U}}$: 代表用户对代表物品的偏好

优势：可解释性强，能识别"原型用户"和"原型物品"。

**应用10.2 (文档聚类)**: 文档-词矩阵 $\boldsymbol{X} \in \mathbb{R}^{n_d \times n_w}$，CUR选择：
- 代表性词汇（列）
- 代表性文档（行）

用于降维和特征选择。

**应用10.3 (图像压缩)**: 图像矩阵 $\boldsymbol{I} \in \mathbb{R}^{h \times w}$：
- $\boldsymbol{\mathcal{C}}$: 代表性列（垂直条纹）
- $\boldsymbol{\mathcal{R}}$: 代表性行（水平条纹）

存储成本：$r(h+w+r)$ vs. 原始 $hw$，当 $r \ll \min(h,w)$ 时显著压缩。

### 11. CUR的可解释性优势

相比SVD，CUR的关键优势在于**可解释性**：

**对比表**:

| 方面 | SVD | CUR |
|------|-----|-----|
| 基向量 | 抽象的正交向量 | 原始数据的行/列 |
| 物理意义 | 难以解释 | 直观可解释 |
| 稀疏性 | 密集矩阵 | 保持原始稀疏性 |
| 计算 | 需全矩阵 | 可增量/流式 |

**定理11.1 (稀疏性保持)**: 若 $\boldsymbol{M}$ 是稀疏矩阵（每行/列非零元素 $\leq s$），则：
- $\boldsymbol{\mathcal{C}}, \boldsymbol{\mathcal{R}}$ 同样稀疏（每列/行非零元素 $\leq s$）
- SVD的 $\boldsymbol{U}, \boldsymbol{V}$ 通常是稠密的

### 12. 小结与展望

CUR分解通过以下方式实现低秩近似：

\begin{equation}
\underbrace{\text{原始矩阵}}_{\boldsymbol{M}} \approx \underbrace{\text{代表列}}_{\boldsymbol{\mathcal{C}}} \times \underbrace{\text{系数矩阵}}_{\boldsymbol{\mathcal{U}}} \times \underbrace{\text{代表行}}_{\boldsymbol{\mathcal{R}}}
\tag{46}
\end{equation}

**核心要点**:
1. $\boldsymbol{\mathcal{U}}$ 的最优解通过伪逆给出（公式4）
2. 列/行选择是关键：杠杆分数和DEIM是主流方法
3. 误差界与SVD相当（在对数因子内）
4. 可解释性和稀疏性是主要优势

**未来方向**:
- 自适应采样策略
- 流式/在线CUR算法
- 张量CUR分解
- 深度学习中的应用（注意力机制的CUR近似）

\begin{equation}
\boxed{\text{CUR分解：保持数据原始结构的智能低秩近似}}
\tag{47}
\end{equation}

