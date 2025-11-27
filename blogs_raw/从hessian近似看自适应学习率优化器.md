---
title: 从Hessian近似看自适应学习率优化器
slug: 从hessian近似看自适应学习率优化器
date: 2024-11-29
tags: 详细推导, 优化, 梯度, 学习率, 优化器, 生成模型
status: completed
---
# 从Hessian近似看自适应学习率优化器

**原文链接**: [https://spaces.ac.cn/archives/10588](https://spaces.ac.cn/archives/10588)

**发布日期**: 

---

这几天在重温去年的Meta的一篇论文[《A Theory on Adam Instability in Large-Scale Machine Learning》](https://papers.cool/arxiv/2304.09871)，里边给出了看待Adam等自适应学习率优化器的新视角：它指出梯度平方的滑动平均某种程度上近似于在估计Hessian矩阵的平方，从而Adam、RMSprop等优化器实际上近似于二阶的Newton法。

这个角度颇为新颖，而且表面上跟以往的一些Hessian近似有明显的差异，因此值得我们去学习和思考一番。

## 牛顿下降 #

设损失函数为$\mathcal{L}(\boldsymbol{\theta})$，其中待优化参数为$\boldsymbol{\theta}$，我们的优化目标是  
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})\label{eq:loss}\end{equation}  
假设$\boldsymbol{\theta}$的当前值是$\boldsymbol{\theta}_t$，Newton法通过将损失函数展开到二阶来寻求$\boldsymbol{\theta}_{t+1}$：  
\begin{equation}\mathcal{L}(\boldsymbol{\theta})\approx \mathcal{L}(\boldsymbol{\theta}_t) + \boldsymbol{g}_t^{\top}(\boldsymbol{\theta} - \boldsymbol{\theta}_t) + \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}_t)^{\top}\boldsymbol{\mathcal{H}}_t(\boldsymbol{\theta} - \boldsymbol{\theta}_t)\end{equation}  
其中$\boldsymbol{g}_t = \nabla_{\boldsymbol{\theta}_t}\mathcal{L}(\boldsymbol{\theta}_t)$是梯度、 $\boldsymbol{\mathcal{H}}_t=\nabla_{\boldsymbol{\theta}_t}^2\mathcal{L}(\boldsymbol{\theta}_t)$是Hessian矩阵。假定Hessian矩阵的正定性，那么上式右端就存在唯一的最小值$\boldsymbol{\theta}_t - \boldsymbol{\mathcal{H}}_t^{-1}\boldsymbol{g}_t$，Newton法将它作为下一步的$\boldsymbol{\theta}_{t+1}$：  
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t-\boldsymbol{\mathcal{H}}_t^{-1}\boldsymbol{g}_t = \boldsymbol{\theta}_t - (\nabla_{\boldsymbol{\theta}_t}^2\mathcal{L})^{-1} \nabla_{\boldsymbol{\theta}_t}\mathcal{L}\end{equation}  
注意上式没有额外的学习率参数，因此Newton法天生就是自适应学习率算法。当然，由于Hessian矩阵的复杂度正比于参数量的平方，所以在深度学习中完整的Newton法基本上只有理论价值了，真要想应用Newton法，要对Hessian矩阵做比较大的简化假设，比如对角矩阵或者低秩矩阵。

在Newton法视角下，SGD就是假设了$\boldsymbol{\mathcal{H}}_t=\eta_t^{-1}\boldsymbol{I}$，而Adam则是假设$\boldsymbol{\mathcal{H}}_t=\eta_t^{-1}\text{diag}(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon)$，其中  
\begin{equation}\text{Adam}:=\left\\{\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\\  
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t\odot\boldsymbol{g}_t\\\  
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\\  
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.  
\end{aligned}\right.\end{equation}  
接下来我们想要证明的是，$\eta_t^{-1}\text{diag}(\sqrt{\hat{\boldsymbol{v}}_t})$是$\boldsymbol{\mathcal{H}}_t$的一个更好的近似。

## 梯度近似 #

证明的要点是利用梯度的一阶近似：  
\begin{equation}\boldsymbol{g}_{\boldsymbol{\theta}} \approx \boldsymbol{g}_{\boldsymbol{\theta}^*} + \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)\end{equation}  
其中$\boldsymbol{g}_{\boldsymbol{\theta}^*}$和$\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}$表明我们在$\boldsymbol{\theta}=\boldsymbol{\theta}^*$处展开，这里的$\boldsymbol{\theta}^*$就是我们要寻找的目标$\eqref{eq:loss}$，在此处模型的梯度为零，从而上式可以简化成  
\begin{equation}\boldsymbol{g}_{\boldsymbol{\theta}} \approx \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)\end{equation}  
于是  
\begin{equation}\boldsymbol{g}_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}}^{\top} \approx \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)(\boldsymbol{\theta} - \boldsymbol{\theta}^*)^{\top}\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}^{\top}\end{equation}  
假设训练进入“正轨”后，模型将会长期处于围绕着$\boldsymbol{\theta}^*$“打转”、缓慢且螺旋地收敛的状态，那么一定程度上我们可以将$\boldsymbol{\theta} - \boldsymbol{\theta}^*$视为正态分布$\mathcal{N}(\boldsymbol{0},\sigma^2\boldsymbol{I})$的随机变量，那么  
\begin{equation}\mathbb{E}[\boldsymbol{g}_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}}^{\top}] \approx \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}\mathbb{E}[(\boldsymbol{\theta} - \boldsymbol{\theta}^*)(\boldsymbol{\theta} - \boldsymbol{\theta}^*)^{\top}]\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}^{\top} = \sigma^2\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}^{\top}\label{eq:hessian-2}\end{equation}  
假设Hessian矩阵是对角阵，那么上式我们可以只保留对角线元素  
\begin{equation}\text{diag}(\mathbb{E}[\boldsymbol{g}_{\boldsymbol{\theta}}\odot\boldsymbol{g}_{\boldsymbol{\theta}}]) \approx \sigma^2\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}^2\quad\Rightarrow\quad \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*} = \frac{1}{\sigma}\text{diag}(\sqrt{\mathbb{E}[\boldsymbol{g}_{\boldsymbol{\theta}}\odot\boldsymbol{g}_{\boldsymbol{\theta}}]})\end{equation}  
是不是有点相似了？Adam的$\hat{\boldsymbol{v}}_t$是对梯度平方的滑动平均，它可以看作在近似$\mathbb{E}[\boldsymbol{g}_{\boldsymbol{\theta}}\odot\boldsymbol{g}_{\boldsymbol{\theta}}]$。最后我们再假设$\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}_t}$相比$\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}$变化不大，就可以得到$\eta_t^{-1}\text{diag}(\sqrt{\hat{\boldsymbol{v}}_t})$是$\boldsymbol{\mathcal{H}}_t$近似的结论。

这也可以解释为什么Adam的$\beta_2$通常都大于$\beta_1$。为了更准确地估计Hessian，$\hat{\boldsymbol{v}}_t$的滑动平均应该尽可能“长期”（接近均匀平均），所以$\beta_2$应该要很接近于1；而动量$\hat{\boldsymbol{m}}_t$是梯度的滑动平均，如果梯度的平均过于长期的话，那么结果将会接近$\boldsymbol{g}_{\boldsymbol{\theta}^*}=\boldsymbol{0}$，这反而不好，因此动量的滑动平均要更局部些。

## 相关工作 #

对于比较了解Hessian矩阵理论的读者，看到上述结论后的第一反应也许不是熟悉而是疑惑，这是因为Hessian矩阵的一个经典近似是Jacobi矩阵（类似梯度）的外积，而这里的Hessian近似则是梯度外积的平方根，两者差了个根号。

具体来说，我们以平方误差损失为例  
\begin{equation}\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2}\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[\Vert \boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})\Vert^2]\label{eq:loss-2}\end{equation}  
我们在$\boldsymbol{\theta}_t$处展开，有$\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})\approx \boldsymbol{f}_{\boldsymbol{\theta}_t}(\boldsymbol{x}) + \boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}_t}^{\top} (\boldsymbol{\theta} - \boldsymbol{\theta}_t)$，其中$\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}_t}=\nabla_{\boldsymbol{\theta}_t} \boldsymbol{f}_{\boldsymbol{\theta}_t}(\boldsymbol{x})$是Jacobi矩阵，代入上式得到  
\begin{equation}\mathcal{L}(\boldsymbol{\theta}) \approx \frac{1}{2}\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[\Vert \boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}_t}(\boldsymbol{x}) - \boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}_t}^{\top} (\boldsymbol{\theta} - \boldsymbol{\theta}_t)\Vert^2]\end{equation}  
经过简化后的上式只是关于$\boldsymbol{\theta}$的二次型，因此可以直接写出它的Hessian矩阵，结果是  
\begin{equation}\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}_t} \approx \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}_t}\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}_t}^{\top}]\end{equation}  
这就是基于Jacobi矩阵外积的Hessian近似，它是“[Gauss–Newton法](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)”的理论基础。当然，$\boldsymbol{\mathcal{J}}$还不是$\boldsymbol{g}$，我们要试图将结果跟$\mathcal{g}$联系起来。对式$\eqref{eq:loss-2}$直接求导得  
\begin{equation}\boldsymbol{g}_{\boldsymbol{\theta}} = \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))]\end{equation}  
于是  
\begin{equation}\begin{aligned}  
\boldsymbol{g}_{\boldsymbol{\theta}} \boldsymbol{g}_{\boldsymbol{\theta}}^{\top} =&\, \big(\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))]\big)\big(\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))]\big)^{\top} \\\\[5pt]  
=&\, \big(\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))]\big)\big(\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}[(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))^{\top}\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}^{\top}]\big) \\\\[5pt]  
\approx&\, \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}\big[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))^{\top}\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}^{\top}\big] \\\\[5pt]  
\approx&\, \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}\Big[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}\big[(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))(\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))^{\top}\big]\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}^{\top}\Big] \\\\[5pt]  
\end{aligned}\end{equation}  
这里两个约等号其实没有太多道理，可以勉强看成是平均场近似，而$\boldsymbol{y} - \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})$是回归预测的残差，我们通常假设它服从$\mathcal{N}(\boldsymbol{0},\sigma^2\boldsymbol{I})$，因此有  
\begin{equation}\boldsymbol{g}_{\boldsymbol{\theta}} \boldsymbol{g}_{\boldsymbol{\theta}}^{\top} \approx \sigma^2\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}}\big[\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}\boldsymbol{\mathcal{J}}_{\boldsymbol{\theta}}^{\top}\big] \approx \sigma^2 \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}_t}\label{eq:hessian-t}\end{equation}  
这就揭示了$\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}_t}$与$\boldsymbol{g}_{\boldsymbol{\theta}} \boldsymbol{g}_{\boldsymbol{\theta}}^{\top}$的联系。对比上一节的式$\eqref{eq:hessian-2}$，可以发现表面上刚好差了个平方。

看推导过程，两个结果似乎都没明显错误，那怎么理解这种不一致性呢？我们可以这样理解：式$\eqref{eq:hessian-t}$给出的是$t$时刻的Hessian近似，属于“瞬时近似”，而式$\eqref{eq:hessian-2}$则是时间步的“长期平均”结果，长期的平均作用抵销了一部分强度（但理论上也会使得估计更准确），从而需要多开一个平方根。

类似的效应也出现在[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209)介绍的SDE中，SDE的噪声项强度需要比非噪声项高半阶，同样是因为噪声项在长期平均之下会抵消，所以噪声需要更高阶才能在最终的结果中体现出噪声的作用。

## 更多联系 #

在前面的推导中，我们假设了$\boldsymbol{\theta}^*$是理论最优点，从而有$\boldsymbol{g} _{\boldsymbol{\theta}^*} = \boldsymbol{0}$。如果$\boldsymbol{\theta}^*$是任意一点呢？那么式$\eqref{eq:hessian-2}$将变成  
\begin{equation}\mathbb{E}[(\boldsymbol{g}_{\boldsymbol{\theta}}-\boldsymbol{g} _{\boldsymbol{\theta}^*})(\boldsymbol{g}_{\boldsymbol{\theta}}-\boldsymbol{g} _{\boldsymbol{\theta}^*})^{\top}] \approx \sigma^2\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}^{\top}\end{equation}  
也就是说我们只要滑动平均的是协方差而不是二阶矩，就可以得到局部范围内的Hessian近似。这正好跟[AdaBelief优化器](https://papers.cool/arxiv/2010.07468)的做法对应上了，它的$\boldsymbol{v}$滑动平均的是$\boldsymbol{g}$与$\boldsymbol{m}$的差的平方：  
\begin{equation}\text{AdaBelief}:=\left\\{\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\\  
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) (\boldsymbol{g}_t - \boldsymbol{m}_t)\odot(\boldsymbol{g}_t - \boldsymbol{m}_t)\\\  
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\\  
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.  
\end{aligned}\right.\end{equation}

## 文章小结 #

本文介绍了从Newton法和Hessian近似看待Adam等自适应学习率优化器的一个视角，并讨论了Hessian近似的相关结果。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10588>_

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

苏剑林. (Nov. 29, 2024). 《从Hessian近似看自适应学习率优化器 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10588>

@online{kexuefm-10588,  
title={从Hessian近似看自适应学习率优化器},  
author={苏剑林},  
year={2024},  
month={Nov},  
url={\url{https://spaces.ac.cn/archives/10588}},  
} 


---

## 公式推导与注释

### 1. Hessian矩阵的定义和作用

#### 1.1 基本定义

对于损失函数$\mathcal{L}(\boldsymbol{\theta}):\mathbb{R}^n\to\mathbb{R}$，Hessian矩阵定义为：

$$
\boldsymbol{\mathcal{H}}(\boldsymbol{\theta}) = \nabla^2_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}) = \begin{bmatrix}
\frac{\partial^2 \mathcal{L}}{\partial \theta_1^2} & \frac{\partial^2 \mathcal{L}}{\partial \theta_1 \partial \theta_2} & \cdots & \frac{\partial^2 \mathcal{L}}{\partial \theta_1 \partial \theta_n} \\
\frac{\partial^2 \mathcal{L}}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 \mathcal{L}}{\partial \theta_2^2} & \cdots & \frac{\partial^2 \mathcal{L}}{\partial \theta_2 \partial \theta_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 \mathcal{L}}{\partial \theta_n \partial \theta_1} & \frac{\partial^2 \mathcal{L}}{\partial \theta_n \partial \theta_2} & \cdots & \frac{\partial^2 \mathcal{L}}{\partial \theta_n^2}
\end{bmatrix}
$$

**性质**：由Schwarz定理，在连续可微的条件下，Hessian矩阵是对称矩阵，即$\boldsymbol{\mathcal{H}}^{\top} = \boldsymbol{\mathcal{H}}$。

#### 1.2 几何意义

Hessian矩阵刻画了损失函数在参数空间中的曲率信息。对于二次函数$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^{\top}\boldsymbol{A}\boldsymbol{\theta} + \boldsymbol{b}^{\top}\boldsymbol{\theta} + c$，其Hessian矩阵恒为$\boldsymbol{\mathcal{H}} = \boldsymbol{A}$。

**正定性判断**：
- 若$\boldsymbol{\mathcal{H}} \succ 0$（正定），则$\boldsymbol{\theta}$处为局部最小值
- 若$\boldsymbol{\mathcal{H}} \prec 0$（负定），则$\boldsymbol{\theta}$处为局部最大值
- 若$\boldsymbol{\mathcal{H}}$不定，则$\boldsymbol{\theta}$处为鞍点

#### 1.3 条件数与优化难度

Hessian矩阵的条件数$\kappa(\boldsymbol{\mathcal{H}}) = \frac{\lambda_{\max}}{\lambda_{\min}}$决定了优化的难度，其中$\lambda_{\max}$和$\lambda_{\min}$分别是最大和最小特征值。条件数越大，优化越困难。

对于梯度下降法，收敛速度受控于：
$$
\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\| \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|
$$

当$\kappa \gg 1$时，收敛速度极慢。这正是二阶方法的优势所在——通过$\boldsymbol{\mathcal{H}}^{-1}$预条件化可以改善条件数。

### 2. 牛顿法的完整推导

#### 2.1 泰勒展开与二阶近似

在$\boldsymbol{\theta}_t$处对$\mathcal{L}(\boldsymbol{\theta})$进行二阶泰勒展开：

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}_t) + \nabla\mathcal{L}(\boldsymbol{\theta}_t)^{\top}(\boldsymbol{\theta} - \boldsymbol{\theta}_t) + \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}_t)^{\top}\boldsymbol{\mathcal{H}}_t(\boldsymbol{\theta} - \boldsymbol{\theta}_t) + O(\|\boldsymbol{\theta} - \boldsymbol{\theta}_t\|^3)
$$

忽略高阶项，定义二阶近似：
$$
\tilde{\mathcal{L}}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}_t) + \boldsymbol{g}_t^{\top}(\boldsymbol{\theta} - \boldsymbol{\theta}_t) + \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}_t)^{\top}\boldsymbol{\mathcal{H}}_t(\boldsymbol{\theta} - \boldsymbol{\theta}_t)
$$

#### 2.2 最优步长的导出

对$\tilde{\mathcal{L}}(\boldsymbol{\theta})$关于$\boldsymbol{\theta}$求导：
$$
\frac{\partial \tilde{\mathcal{L}}}{\partial \boldsymbol{\theta}} = \boldsymbol{g}_t + \boldsymbol{\mathcal{H}}_t(\boldsymbol{\theta} - \boldsymbol{\theta}_t)
$$

令导数为零，得到：
$$
\boldsymbol{g}_t + \boldsymbol{\mathcal{H}}_t(\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t) = \boldsymbol{0}
$$

解得牛顿更新规则：
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \boldsymbol{\mathcal{H}}_t^{-1}\boldsymbol{g}_t
$$

#### 2.3 收敛性分析

**定理（牛顿法局部二次收敛）**：假设$\boldsymbol{\mathcal{H}}$在$\boldsymbol{\theta}^*$的邻域内Lipschitz连续，且$\boldsymbol{\mathcal{H}}(\boldsymbol{\theta}^*) \succ 0$，则存在$\delta > 0$，当$\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\| < \delta$时，牛顿法二次收敛：

$$
\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\| \leq C\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2
$$

**证明思路**：在最优点$\boldsymbol{\theta}^*$处，$\boldsymbol{g}^* = \boldsymbol{0}$。对梯度在$\boldsymbol{\theta}^*$处泰勒展开：
$$
\boldsymbol{g}_t = \boldsymbol{g}^* + \boldsymbol{\mathcal{H}}^*(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + O(\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2) = \boldsymbol{\mathcal{H}}^*(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + O(\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2)
$$

代入牛顿更新：
$$
\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* = \boldsymbol{\theta}_t - \boldsymbol{\theta}^* - \boldsymbol{\mathcal{H}}_t^{-1}\boldsymbol{g}_t = \boldsymbol{\mathcal{H}}_t^{-1}(\boldsymbol{\mathcal{H}}_t - \boldsymbol{\mathcal{H}}^*)(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + O(\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2)
$$

由Lipschitz连续性，$\|\boldsymbol{\mathcal{H}}_t - \boldsymbol{\mathcal{H}}^*\| \leq L\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|$，因此：
$$
\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\| \leq C\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2
$$

#### 2.4 牛顿法的计算复杂度挑战

对于$n$维参数：
- **计算Hessian矩阵**：$O(n^2)$存储，$O(n^2)$或$O(n^3)$计算（取决于实现）
- **求逆或求解线性系统**：$O(n^3)$（如Cholesky分解）
- **每步总复杂度**：$O(n^3)$

在深度学习中，$n$通常为$10^6 \sim 10^{10}$，这使得完整的牛顿法不可行。这也是为什么我们需要Hessian近似方法。

### 3. 自适应学习率的理论基础

#### 3.1 预条件化梯度下降

预条件化的思想是通过矩阵$\boldsymbol{P}_t$变换梯度：
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{P}_t^{-1}\boldsymbol{g}_t
$$

**最优预条件器**：从牛顿法的角度，最优的预条件器是$\boldsymbol{P}_t = \boldsymbol{\mathcal{H}}_t$。

#### 3.2 对角近似的合理性

假设参数之间的二阶相互作用较弱，即Hessian矩阵近似对角：
$$
\boldsymbol{\mathcal{H}} \approx \text{diag}(h_1, h_2, \ldots, h_n)
$$

此时，不同参数的最优学习率应该不同：
$$
\theta_{i,t+1} = \theta_{i,t} - \frac{\eta_t}{h_i}g_{i,t}
$$

这正是自适应学习率的核心思想——为每个参数分配不同的有效学习率$\frac{\eta_t}{h_i}$。

#### 3.3 曲率自适应的必要性

考虑一个简单的二维二次函数：
$$
\mathcal{L}(\theta_1, \theta_2) = \frac{1}{2}(a\theta_1^2 + b\theta_2^2), \quad a \gg b
$$

其Hessian矩阵为$\boldsymbol{\mathcal{H}} = \text{diag}(a, b)$。使用统一学习率$\eta$的SGD：
$$
\theta_{1,t+1} = \theta_{1,t} - \eta a\theta_{1,t}, \quad \theta_{2,t+1} = \theta_{2,t} - \eta b\theta_{2,t}
$$

为了保证$\theta_1$方向收敛，需要$\eta < \frac{2}{a}$；但这会导致$\theta_2$方向收敛极慢（因为$b \ll a$）。

若使用自适应学习率$\eta_i = \frac{\eta}{h_i}$，则：
$$
\theta_{1,t+1} = \theta_{1,t} - \frac{\eta}{a} a\theta_{1,t} = (1-\eta)\theta_{1,t}
$$
$$
\theta_{2,t+1} = \theta_{2,t} - \frac{\eta}{b} b\theta_{2,t} = (1-\eta)\theta_{2,t}
$$

两个方向以相同速度收敛，解决了条件数问题。

### 4. Adam作为对角Hessian近似

#### 4.1 从梯度方差到Hessian平方

回顾正文中的核心推导。假设在最优点$\boldsymbol{\theta}^*$附近，梯度可以线性近似：
$$
\boldsymbol{g}_{\boldsymbol{\theta}} \approx \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)
$$

当$\boldsymbol{\theta} - \boldsymbol{\theta}^* \sim \mathcal{N}(\boldsymbol{0}, \sigma^2\boldsymbol{I})$时：
$$
\mathbb{E}[\boldsymbol{g}_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}}^{\top}] = \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}\mathbb{E}[(\boldsymbol{\theta} - \boldsymbol{\theta}^*)(\boldsymbol{\theta} - \boldsymbol{\theta}^*)^{\top}]\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}^{\top} = \sigma^2\boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}^2
$$

**对角化**：假设$\boldsymbol{\mathcal{H}}$对角，则：
$$
\text{diag}(\mathbb{E}[\boldsymbol{g} \odot \boldsymbol{g}]) = \sigma^2 \text{diag}(\boldsymbol{\mathcal{H}}^2) = \sigma^2 \boldsymbol{h}^2
$$

其中$\boldsymbol{h} = (h_1, h_2, \ldots, h_n)^{\top}$是Hessian的对角元素。解得：
$$
h_i = \frac{1}{\sigma}\sqrt{\mathbb{E}[g_i^2]}
$$

#### 4.2 Adam的滑动平均估计

Adam通过指数移动平均（EMA）估计$\mathbb{E}[g_i^2]$：
$$
v_{i,t} = \beta_2 v_{i,t-1} + (1-\beta_2)g_{i,t}^2
$$

展开递归关系：
$$
v_{i,t} = (1-\beta_2)\sum_{k=1}^{t}\beta_2^{t-k}g_{i,k}^2
$$

这是一个指数加权平均，有效窗口大小约为$\frac{1}{1-\beta_2}$。当$\beta_2 = 0.999$时，有效窗口约为1000步。

**偏差修正**：初始时$v_{i,t}$被低估（因为$v_{i,0} = 0$）。偏差修正：
$$
\hat{v}_{i,t} = \frac{v_{i,t}}{1-\beta_2^t}
$$

可以证明，当$g_{i,k}$服从稳态分布时，$\mathbb{E}[\hat{v}_{i,t}] \approx \mathbb{E}[g_i^2]$。

#### 4.3 Adam更新的Hessian解释

结合以上分析，Adam的更新可以写成：
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \hat{\boldsymbol{\mathcal{H}}}_t^{-1}\hat{\boldsymbol{m}}_t
$$

其中近似Hessian为：
$$
\hat{\boldsymbol{\mathcal{H}}}_t = \frac{1}{\eta_t\sigma}\text{diag}(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon)
$$

**$\epsilon$的作用**：
1. 数值稳定性：防止除零
2. 正则化：相当于添加了$\frac{\epsilon}{\eta_t\sigma}$的先验到Hessian对角元素
3. 有效学习率下界：确保$\frac{\eta_t}{\sqrt{v_{i,t}}+\epsilon} \geq \frac{\eta_t}{\epsilon}$

#### 4.4 动量项的作用

Adam的动量$\hat{\boldsymbol{m}}_t$是梯度的指数移动平均：
$$
\hat{m}_{i,t} = \frac{\sum_{k=1}^{t}\beta_1^{t-k}(1-\beta_1)g_{i,k}}{1-\beta_1^t}
$$

从Hessian角度理解，动量近似于在滑动窗口内对梯度求平均，这相当于在$\boldsymbol{\theta}$的局部邻域内积分梯度信息。如果窗口过长（$\beta_1$接近1），则：
$$
\lim_{\beta_1 \to 1}\hat{m}_{i,t} \approx \mathbb{E}_{\boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\theta}^*, \sigma^2\boldsymbol{I})}[g_i] = 0
$$

这解释了为什么$\beta_1 < \beta_2$：动量需要局部信息，而Hessian估计需要长期统计。

### 5. AdaGrad的Hessian解释

#### 5.1 AdaGrad算法

AdaGrad累积历史梯度的平方和：
$$
G_{i,t} = \sum_{k=1}^{t}g_{i,k}^2
$$

更新规则：
$$
\theta_{i,t+1} = \theta_{i,t} - \frac{\eta}{\sqrt{G_{i,t}} + \epsilon}g_{i,t}
$$

#### 5.2 均匀平均的Hessian估计

AdaGrad相当于对梯度平方进行均匀平均（而非指数加权）：
$$
\bar{g}_{i,t}^2 = \frac{1}{t}\sum_{k=1}^{t}g_{i,k}^2 \approx \mathbb{E}[g_i^2]
$$

因此$G_{i,t} \approx t\mathbb{E}[g_i^2]$，有效学习率为：
$$
\eta_{i,t}^{\text{eff}} = \frac{\eta}{\sqrt{t\mathbb{E}[g_i^2]}} \propto \frac{1}{\sqrt{t}}
$$

这是AdaGrad学习率衰减的根源。

#### 5.3 对角Hessian近似

类比Adam的分析，AdaGrad隐式假设：
$$
h_i \approx \frac{1}{\sigma}\sqrt{\frac{G_{i,t}}{t}} = \frac{1}{\sigma}\sqrt{\mathbb{E}[g_i^2]}
$$

**与Adam的对比**：
- Adam使用指数移动平均，可以适应非平稳分布
- AdaGrad使用累积平均，适合平稳分布但学习率单调递减

#### 5.4 稀疏梯度场景

AdaGrad在稀疏梯度场景下表现优异。假设第$i$个参数的梯度只在一小部分步骤非零：
$$
g_{i,k} = \begin{cases}
\bar{g}_i, & k \in S_i \\
0, & k \notin S_i
\end{cases}
$$

其中$|S_i| = s \ll t$。此时：
$$
G_{i,t} = s\bar{g}_i^2, \quad \eta_{i,t}^{\text{eff}} = \frac{\eta}{\sqrt{s}\bar{g}_i}
$$

稀疏参数获得更大的有效学习率（因为$s$小），这正是NLP和推荐系统需要的特性。

### 6. RMSProp的改进机制

#### 6.1 RMSProp算法

RMSProp通过指数移动平均解决AdaGrad学习率单调递减的问题：
$$
v_{i,t} = \beta v_{i,t-1} + (1-\beta)g_{i,t}^2
$$
$$
\theta_{i,t+1} = \theta_{i,t} - \frac{\eta}{\sqrt{v_{i,t}} + \epsilon}g_{i,t}
$$

通常$\beta = 0.9$或$0.99$。

#### 6.2 非平稳Hessian追踪

在非平稳优化问题中，Hessian矩阵随时间变化：$\boldsymbol{\mathcal{H}}_t \neq \boldsymbol{\mathcal{H}}_{t'}$。AdaGrad的累积平均无法适应这种变化，而RMSProp的指数加权平均给予近期梯度更高权重：
$$
v_{i,t} \approx \frac{1-\beta}{1-\beta^t}\sum_{k=1}^{t}\beta^{t-k}g_{i,k}^2 \approx \mathbb{E}_{\text{recent}}[g_i^2]
$$

**追踪能力分析**：假设Hessian在$t_0$时刻发生突变，从$h_i^{\text{old}}$变为$h_i^{\text{new}}$。定义追踪误差：
$$
\varepsilon_t = |v_{i,t} - (h_i^{\text{new}})^2\sigma^2|
$$

可以证明：
$$
\varepsilon_t \leq \beta^{t-t_0}\varepsilon_{t_0}
$$

追踪时间常数$\tau = \frac{1}{1-\beta}$。对于$\beta = 0.9$，$\tau = 10$步；$\beta = 0.99$时，$\tau = 100$步。

#### 6.3 与Adam的关系

Adam可以看作RMSProp + 动量：
- RMSProp：$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{v_t}+\epsilon}g_t$
- Adam：$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{v_t}+\epsilon}m_t$

动量$m_t$降低了梯度噪声的方差。设$\text{Var}[g_t] = \sigma_g^2$，则：
$$
\text{Var}[m_t] \approx \frac{1-\beta_1}{1+\beta_1}\sigma_g^2 < \sigma_g^2
$$

这使得Adam在噪声环境下更稳定。

### 7. 自然梯度与Fisher信息矩阵

#### 7.1 自然梯度的动机

普通梯度下降在参数空间中沿着欧几里得梯度方向移动。但参数空间的欧几里得度量不一定反映模型输出空间的真实距离。

**例子**：考虑两组参数$\boldsymbol{\theta}_1$和$\boldsymbol{\theta}_2$，它们在参数空间中距离相同（$\|\boldsymbol{\theta}_1 - \boldsymbol{\theta}_0\| = \|\boldsymbol{\theta}_2 - \boldsymbol{\theta}_0\|$），但可能导致模型输出分布的差异很大。

#### 7.2 Fisher信息矩阵

对于概率模型$p(\boldsymbol{y}|\boldsymbol{x}; \boldsymbol{\theta})$，Fisher信息矩阵定义为：
$$
\boldsymbol{F}(\boldsymbol{\theta}) = \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim p_{\text{data}}}\mathbb{E}_{\boldsymbol{y}'\sim p(\cdot|\boldsymbol{x};\boldsymbol{\theta})}\left[\nabla_{\boldsymbol{\theta}}\log p(\boldsymbol{y}'|\boldsymbol{x};\boldsymbol{\theta})\nabla_{\boldsymbol{\theta}}\log p(\boldsymbol{y}'|\boldsymbol{x};\boldsymbol{\theta})^{\top}\right]
$$

Fisher信息矩阵度量了参数变化对模型输出分布的影响，是参数空间的**黎曼度量**。

#### 7.3 自然梯度定义

自然梯度是在Fisher度量下的最速下降方向：
$$
\tilde{\boldsymbol{g}}_t = \boldsymbol{F}_t^{-1}\boldsymbol{g}_t
$$

更新规则：
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{F}_t^{-1}\boldsymbol{g}_t
$$

**几何解释**：自然梯度在KL散度意义下找到最优更新方向。对于约束优化问题：
$$
\min_{\boldsymbol{\delta}} \boldsymbol{g}_t^{\top}\boldsymbol{\delta}, \quad \text{s.t.} \quad D_{\text{KL}}(p(\cdot;\boldsymbol{\theta}_t) \| p(\cdot;\boldsymbol{\theta}_t + \boldsymbol{\delta})) \leq \epsilon
$$

解为$\boldsymbol{\delta}^* \propto -\boldsymbol{F}_t^{-1}\boldsymbol{g}_t$。

#### 7.4 Fisher矩阵与Hessian的关系

对于负对数似然损失$\mathcal{L}(\boldsymbol{\theta}) = -\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})}[\log p(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})]$：

**一阶导**：
$$
\boldsymbol{g}(\boldsymbol{\theta}) = -\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})}[\nabla_{\boldsymbol{\theta}}\log p(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})]
$$

**二阶导（Hessian）**：
$$
\boldsymbol{\mathcal{H}}(\boldsymbol{\theta}) = -\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})}[\nabla_{\boldsymbol{\theta}}^2\log p(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})]
$$

**Fisher信息矩阵**：
$$
\boldsymbol{F}(\boldsymbol{\theta}) = \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim p(\cdot;\boldsymbol{\theta})}[\nabla_{\boldsymbol{\theta}}\log p(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})\nabla_{\boldsymbol{\theta}}\log p(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})^{\top}]
$$

在模型正确指定（$p_{\text{data}} = p(\cdot;\boldsymbol{\theta}^*)$）的情况下，有恒等式：
$$
\boldsymbol{F}(\boldsymbol{\theta}) = \boldsymbol{\mathcal{H}}(\boldsymbol{\theta})
$$

**证明**：利用$\mathbb{E}_{\boldsymbol{y}\sim p}[\nabla\log p] = \boldsymbol{0}$，计算：
$$
\nabla^2_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{y}}[\log p(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})] = \mathbb{E}_{\boldsymbol{y}}[\nabla^2\log p] + \mathbb{E}_{\boldsymbol{y}}[\nabla\log p \cdot \nabla\log p^{\top}]
$$

左边为0（因为$\mathbb{E}_{\boldsymbol{y}}[\log p]$不依赖于$\boldsymbol{\theta}$在真实分布下），因此：
$$
\mathbb{E}_{\boldsymbol{y}}[\nabla^2\log p] = -\mathbb{E}_{\boldsymbol{y}}[\nabla\log p \cdot \nabla\log p^{\top}]
$$

即$\boldsymbol{\mathcal{H}} = \boldsymbol{F}$。

#### 7.5 对角Fisher近似与Adam

实践中常用对角Fisher近似：
$$
\boldsymbol{F} \approx \text{diag}(\mathbb{E}[(\nabla_{\theta_i}\log p)^2])
$$

这与Adam的$\text{diag}(\mathbb{E}[g_i^2])$形式一致！因此，Adam也可以看作是对角自然梯度的近似。

### 8. K-FAC和Shampoo的块对角近似

#### 8.1 完整矩阵近似的困难

对于$n$维参数，存储完整的$\boldsymbol{F}$或$\boldsymbol{\mathcal{H}}$需要$O(n^2)$空间，求逆需要$O(n^3)$时间。在深度网络中不可行。

**块对角近似**：假设不同层之间的参数独立，将Hessian分解为块对角：
$$
\boldsymbol{\mathcal{H}} \approx \begin{bmatrix}
\boldsymbol{\mathcal{H}}_1 & & \\
& \boldsymbol{\mathcal{H}}_2 & \\
& & \ddots
\end{bmatrix}
$$

每个块对应一层的参数。

#### 8.2 K-FAC（Kronecker-Factored Approximate Curvature）

K-FAC进一步对每层的Fisher矩阵进行Kronecker分解。

**设定**：考虑全连接层$\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}$，参数为$\boldsymbol{W} \in \mathbb{R}^{m \times n}$。将$\boldsymbol{W}$向量化为$\text{vec}(\boldsymbol{W}) \in \mathbb{R}^{mn}$。

**Fisher矩阵**：对于这一层，
$$
\boldsymbol{F}_{\boldsymbol{W}} = \mathbb{E}\left[\nabla_{\text{vec}(\boldsymbol{W})}\log p \cdot \nabla_{\text{vec}(\boldsymbol{W})}\log p^{\top}\right]
$$

**Kronecker近似**：K-FAC假设
$$
\boldsymbol{F}_{\boldsymbol{W}} \approx \boldsymbol{A} \otimes \boldsymbol{S}
$$

其中：
- $\boldsymbol{A} = \mathbb{E}[\boldsymbol{a}\boldsymbol{a}^{\top}] \in \mathbb{R}^{m \times m}$，$\boldsymbol{a}$为该层的激活值
- $\boldsymbol{S} = \mathbb{E}[\boldsymbol{s}\boldsymbol{s}^{\top}] \in \mathbb{R}^{n \times n}$，$\boldsymbol{s}$为该层的梯度
- $\otimes$表示Kronecker积

**Kronecker积性质**：
$$
(\boldsymbol{A} \otimes \boldsymbol{S})^{-1} = \boldsymbol{A}^{-1} \otimes \boldsymbol{S}^{-1}
$$

因此，求逆复杂度从$O((mn)^3)$降低到$O(m^3 + n^3)$。

**更新规则**：
$$
\text{vec}(\boldsymbol{W}_{t+1}) = \text{vec}(\boldsymbol{W}_t) - \eta_t(\boldsymbol{A}_t^{-1} \otimes \boldsymbol{S}_t^{-1})\text{vec}(\boldsymbol{G}_t)
$$

等价于：
$$
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t - \eta_t \boldsymbol{A}_t^{-1}\boldsymbol{G}_t\boldsymbol{S}_t^{-1}
$$

其中$\boldsymbol{G}_t = \nabla_{\boldsymbol{W}_t}\mathcal{L}$。

#### 8.3 Shampoo算法

Shampoo是K-FAC的变体，使用不同的矩阵分解策略。

**核心思想**：对于矩阵参数$\boldsymbol{W} \in \mathbb{R}^{m \times n}$，近似Hessian为：
$$
\boldsymbol{\mathcal{H}}_{\boldsymbol{W}} \approx \boldsymbol{L} \otimes \boldsymbol{R}
$$

其中$\boldsymbol{L} \in \mathbb{R}^{m \times m}$和$\boldsymbol{R} \in \mathbb{R}^{n \times n}$通过AdaGrad式的累积估计：
$$
\boldsymbol{L}_t = \boldsymbol{L}_{t-1} + \boldsymbol{G}_t\boldsymbol{G}_t^{\top}
$$
$$
\boldsymbol{R}_t = \boldsymbol{R}_{t-1} + \boldsymbol{G}_t^{\top}\boldsymbol{G}_t
$$

**更新规则**：
$$
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t - \eta_t \boldsymbol{L}_t^{-1/4}\boldsymbol{G}_t\boldsymbol{R}_t^{-1/4}
$$

注意这里使用$-1/4$次幂（而非$-1/2$），是为了平衡两个方向的预条件化强度。

**计算优化**：
- 通过特征分解计算矩阵的分数次幂
- 每隔若干步更新$\boldsymbol{L}_t^{-1/4}$和$\boldsymbol{R}_t^{-1/4}$（而非每步）
- 对于向量参数（如bias），退化为AdaGrad式的对角预条件

#### 8.4 块对角近似的理论保证

**定理**：假设真实Hessian可以分解为$\boldsymbol{\mathcal{H}} = \boldsymbol{\mathcal{H}}_1 \oplus \boldsymbol{\mathcal{H}}_2 \oplus \cdots \oplus \boldsymbol{\mathcal{H}}_L$（块对角），且每块的Kronecker近似误差有界：
$$
\|\boldsymbol{\mathcal{H}}_l - \boldsymbol{A}_l \otimes \boldsymbol{S}_l\|_F \leq \epsilon_l
$$

则K-FAC的预条件化误差满足：
$$
\left\|\boldsymbol{\mathcal{H}}^{-1} - \bigoplus_{l=1}^{L}(\boldsymbol{A}_l^{-1} \otimes \boldsymbol{S}_l^{-1})\right\| \leq C\sum_{l=1}^{L}\frac{\epsilon_l}{\lambda_{\min}(\boldsymbol{\mathcal{H}}_l)^2}
$$

这说明，当Kronecker结构近似准确时，K-FAC可以很好地近似牛顿法。

### 9. Muon的隐式Hessian预条件

#### 9.1 Muon算法概述

Muon（Momentum with Orthogonalization）是一种新型优化器，它通过动量的正交化隐式地进行Hessian预条件化。

**核心更新**：
$$
\boldsymbol{m}_t = \beta \boldsymbol{m}_{t-1} + (1-\beta)\boldsymbol{g}_t
$$
$$
\tilde{\boldsymbol{m}}_t = \text{Orth}(\boldsymbol{m}_t, \boldsymbol{\theta}_t)
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \tilde{\boldsymbol{m}}_t
$$

其中$\text{Orth}(\boldsymbol{m}, \boldsymbol{\theta})$表示将$\boldsymbol{m}$相对于某个度量正交化。

#### 9.2 正交化与隐式预条件

**观察**：在牛顿法框架下，更新方向$\boldsymbol{d}_t = -\boldsymbol{\mathcal{H}}_t^{-1}\boldsymbol{g}_t$满足关于Hessian度量的"正交性"。

考虑二次函数$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^{\top}\boldsymbol{\mathcal{H}}\boldsymbol{\theta} - \boldsymbol{b}^{\top}\boldsymbol{\theta}$，最优解$\boldsymbol{\theta}^* = \boldsymbol{\mathcal{H}}^{-1}\boldsymbol{b}$。定义Hessian范数：
$$
\|\boldsymbol{x}\|_{\boldsymbol{\mathcal{H}}} = \sqrt{\boldsymbol{x}^{\top}\boldsymbol{\mathcal{H}}\boldsymbol{x}}
$$

牛顿方向$\boldsymbol{d}_t = -\boldsymbol{\mathcal{H}}^{-1}\boldsymbol{g}_t$使得$\|\boldsymbol{\theta}_t + \boldsymbol{d}_t - \boldsymbol{\theta}^*\|_{\boldsymbol{\mathcal{H}}}$最小。

#### 9.3 具体实现：Newton-Schulz迭代

Muon使用Newton-Schulz迭代近似矩阵的逆平方根，实现隐式的Hessian预条件。

**Newton-Schulz迭代**：给定矩阵$\boldsymbol{A}$，迭代计算$\boldsymbol{A}^{-1/2}$：
$$
\boldsymbol{X}_{k+1} = \frac{1}{2}\boldsymbol{X}_k(3\boldsymbol{I} - \boldsymbol{A}\boldsymbol{X}_k^2)
$$

初始化$\boldsymbol{X}_0 = \frac{1}{\|\boldsymbol{A}\|}\boldsymbol{I}$，该迭代三阶收敛到$\boldsymbol{A}^{-1/2}$。

**Muon中的应用**：对于动量$\boldsymbol{m}_t$，计算：
$$
\boldsymbol{G}_t = \boldsymbol{m}_t\boldsymbol{m}_t^{\top}
$$
$$
\boldsymbol{G}_t^{-1/2} \approx \text{NewtonSchulz}(\boldsymbol{G}_t, n_{\text{iter}})
$$
$$
\tilde{\boldsymbol{m}}_t = \boldsymbol{G}_t^{-1/2}\boldsymbol{m}_t
$$

这隐式地对梯度二阶矩进行了预条件化。

#### 9.4 与Adam的对比

**Adam的预条件**：
$$
\boldsymbol{P}_{\text{Adam}} = \text{diag}(\sqrt{v_1}, \sqrt{v_2}, \ldots, \sqrt{v_n})
$$

**Muon的预条件**：
$$
\boldsymbol{P}_{\text{Muon}} \approx (\boldsymbol{m}_t\boldsymbol{m}_t^{\top})^{1/2}
$$

Muon考虑了参数之间的相关性（非对角结构），理论上能捕捉更多曲率信息。但计算成本也更高。

**复杂度对比**：
- Adam：$O(n)$（对角运算）
- Muon：$O(n^2)$或更高（取决于Newton-Schulz迭代次数和矩阵维度）

实践中，Muon通常用于高维但结构化的参数（如Transformer的权重矩阵）。

### 10. 收敛速度的理论分析

#### 10.1 SGD的收敛速率

对于$L$-光滑、$\mu$-强凸的函数，SGD with constant step size $\eta \leq \frac{1}{L}$满足：
$$
\mathbb{E}[\mathcal{L}(\boldsymbol{\theta}_t)] - \mathcal{L}(\boldsymbol{\theta}^*) \leq \frac{\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2\eta t} + \frac{\eta\sigma^2}{2}
$$

其中$\sigma^2 = \mathbb{E}[\|\boldsymbol{g} - \nabla\mathcal{L}\|^2]$是梯度噪声的方差。

**最优学习率**：平衡两项，取$\eta^* \sim \frac{1}{\sqrt{t}}$，得到收敛率$O(1/\sqrt{t})$。

#### 10.2 自适应方法的改进

Adam等自适应方法通过参数特定的学习率改善条件数依赖。

**定理（Adam收敛率，简化版）**：在凸且Lipschitz梯度假设下，Adam的regret bound为：
$$
\sum_{t=1}^{T}[\mathcal{L}(\boldsymbol{\theta}_t) - \mathcal{L}(\boldsymbol{\theta}^*)] \leq O\left(\frac{d}{\sqrt{T}}\right)
$$

其中$d$是参数维度。对比SGD的$O(d\sqrt{T})$，Adam在高维稀疏情况下优势明显。

**关键改进**：Adam的有效学习率$\frac{\eta}{\sqrt{v_{i,t}}}$自适应于每个参数的梯度尺度，等效于：
$$
\eta_{i,t}^{\text{eff}} = \frac{\eta}{\sqrt{\sum_{k=1}^{t}g_{i,k}^2}}
$$

对于稀疏梯度（部分$g_{i,k} = 0$），相应参数保持较大学习率，加速收敛。

#### 10.3 二阶方法的理论优势

**牛顿法的收敛率**：在$\boldsymbol{\theta}^*$的邻域内，牛顿法二次收敛：
$$
\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\| \leq C\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2
$$

**准牛顿法**（如K-FAC）：在Hessian近似误差$\|\boldsymbol{\mathcal{H}} - \tilde{\boldsymbol{\mathcal{H}}}\| \leq \epsilon$的条件下，超线性收敛：
$$
\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\| \leq C(1+\epsilon)\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^{1+\alpha}
$$

其中$\alpha \in (0, 1]$取决于近似质量。

#### 10.4 非凸优化的挑战

深度学习中的损失函数是非凸的，以上凸优化理论不完全适用。

**鞍点问题**：Hessian矩阵可能不定（有负特征值），导致牛顿法方向错误。

**解决方案**：
1. **Hessian修正**：$\tilde{\boldsymbol{\mathcal{H}}} = \boldsymbol{\mathcal{H}} + \lambda\boldsymbol{I}$，其中$\lambda > |\lambda_{\min}(\boldsymbol{\mathcal{H}})|$
2. **信赖域方法**：限制更新步长$\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t\| \leq \Delta_t$
3. **对角近似**：只使用Hessian对角元素（通常非负），避免负曲率

**Adam的鲁棒性**：Adam通过$\sqrt{v_t}$始终保证更新方向与梯度相反（下降方向），自然避免了负曲率问题。

#### 10.5 收敛速度的实证比较

以ImageNet训练ResNet-50为例（假设数据）：

| 优化器 | 收敛步数（至90% acc） | 每步时间 | 总时间 |
|--------|----------------------|----------|--------|
| SGD | 90 epochs | 1.0x | 90x |
| Adam | 70 epochs | 1.0x | 70x |
| K-FAC | 50 epochs | 1.5x | 75x |
| Shampoo | 45 epochs | 2.0x | 90x |

**观察**：
- Adam比SGD快（步数少）
- 二阶方法（K-FAC/Shampoo）步数更少，但每步计算成本高
- 实际墙钟时间的权衡取决于具体硬件和实现

### 11. 深度学习中的实践考虑

#### 11.1 Hessian近似的有效性条件

**局部近似假设**：大多数分析假设在$\boldsymbol{\theta}^*$附近，梯度可线性近似：
$$
\boldsymbol{g}_{\boldsymbol{\theta}} \approx \boldsymbol{\mathcal{H}}_{\boldsymbol{\theta}^*}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)
$$

这在训练后期（微调阶段）较为准确，但在训练早期可能不成立。

**平稳性假设**：Adam等方法假设梯度统计量相对平稳。但在：
- Learning rate warm-up阶段
- Batch size变化时
- 跨越不同训练阶段（如预训练→微调）

这些假设可能失效，需要调整超参数（如重置动量累积）。

#### 11.2 数值稳定性

**$\epsilon$的选择**：Adam中的$\epsilon$（典型值$10^{-8}$）需要平衡：
- 太小：数值不稳定，尤其在混合精度训练中
- 太大：抵消了自适应效果，退化为接近SGD

**建议**：
- FP32训练：$\epsilon = 10^{-8}$
- FP16/BF16训练：$\epsilon = 10^{-6}$或$10^{-5}$

#### 11.3 超参数选择的Hessian视角

**学习率$\eta$**：从$\boldsymbol{\mathcal{H}} \approx \frac{1}{\eta\sigma}\text{diag}(\sqrt{v})$，有效学习率为：
$$
\eta_{i}^{\text{eff}} = \frac{\eta}{\sqrt{v_i}}
$$

**$\beta_2$的选择**：$\beta_2$控制Hessian估计的时间窗口。较大的$\beta_2$（如0.999）使得估计更稳定但反应慢；较小的$\beta_2$（如0.9）反应快但可能不准确。

**经验法则**：
- 稳定、平稳的任务（如收敛后的微调）：$\beta_2 = 0.999$
- 快速变化的任务（如强化学习、GAN）：$\beta_2 = 0.9$ 或更小

#### 11.4 不同优化器的适用场景

| 优化器 | 最适合场景 | Hessian近似类型 | 计算成本 |
|--------|-----------|----------------|----------|
| SGD+Momentum | 视觉、凸优化 | $\boldsymbol{\mathcal{H}} \approx \eta^{-1}\boldsymbol{I}$ | 最低 |
| AdaGrad | 稀疏特征、NLP | 对角累积 | 低 |
| RMSProp | 非平稳、RL | 对角指数平均 | 低 |
| Adam | 通用、默认选择 | 对角+动量 | 低 |
| AdaBelief | 改进的Adam | 对角协方差 | 低 |
| K-FAC | 大规模视觉 | Kronecker块 | 中 |
| Shampoo | Transformer | Kronecker块 | 中 |
| Muon | 结构化参数 | 隐式全矩阵 | 高 |

### 12. 总结与展望

#### 12.1 核心洞察

本推导揭示了自适应学习率优化器的统一视角：

1. **二阶本质**：Adam等自适应方法本质上是对角Hessian近似的牛顿法
2. **梯度平方的统计意义**：$\mathbb{E}[g^2] \propto \boldsymbol{\mathcal{H}}^2$（长期平均）
3. **Fisher-Hessian联系**：在概率模型中，Fisher信息矩阵等于Hessian
4. **块结构的利用**：K-FAC/Shampoo通过Kronecker分解在计算成本和近似精度间权衡

#### 12.2 理论与实践的差距

尽管理论优美，实践中仍有gap：

1. **非凸性**：深度网络的非凸性使得局部二阶近似不总是有效
2. **计算成本**：完整Hessian的$O(n^2)$复杂度在大模型中不可行
3. **超参数敏感性**：自适应方法引入更多超参数（$\beta_1, \beta_2, \epsilon$），需要仔细调节

#### 12.3 未来方向

1. **自动超参数调节**：根据Hessian估计动态调整$\eta, \beta_2$
2. **混合精度Hessian**：在低精度下高效计算和存储Hessian近似
3. **Layer-wise自适应**：不同层使用不同的Hessian近似策略
4. **理论保证**：在实际深度网络设定下建立收敛性理论

**最终思考**：Hessian近似为理解优化器提供了统一框架。从对角近似（Adam）到块近似（K-FAC）再到隐式近似（Muon），本质上是在计算成本与曲率信息精度之间寻找最优平衡。深度学习的未来优化器设计，很可能继续沿着这条主线——更高效地近似和利用Hessian信息。

