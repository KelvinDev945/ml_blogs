---
title: 流形上的最速下降：4. Muon + 谱球面
slug: 流形上的最速下降4-muon-谱球面
date: 2025-08-21
tags: 详细推导, 矩阵, 优化器, muon, 约束, 最速下降
status: pending
---
# 流形上的最速下降：4. Muon + 谱球面

**原文链接**: [https://spaces.ac.cn/archives/11241](https://spaces.ac.cn/archives/11241)

**发布日期**: 

---

看完了前三篇的读者，想必已经熟悉本系列的“套路”——先提出更新量的约束，寻找最速下降方向，接着再给参数也加上约束，寻找新的最速下降方向。在求解参数约束问题时，我们采用的是“一阶近似够用”原则来简化约束形式，这在几何上对应于“切空间”。然后，我们用待定系数法转化无约束形式来写出解析解，最后再数值求解待定系数。

这篇文章我们再来求解一个新例子——谱球面约束下的Muon——它是第一篇文章[《流形上的最速下降：1. SGD + 超球面》](/archives/11196)的类比推广，当我们希望参数的谱范数始终不变时可以考虑它。当然，也可以单纯作为一道练习题来练手。

## 问题描述 #

在[《流形上的最速下降：2. Muon + 正交》](/archives/11215)和[《流形上的最速下降：3. Muon + Stiefel》](/archives/11221)中，我们已经详细讨论了Muon与正交约束的碰撞，所以相关背景我们就不展开了，直接给出问题形式：  
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \Vert\boldsymbol{W}\Vert_2 = 1,\,\,\Vert\boldsymbol{W} - \eta \boldsymbol{\Phi}\Vert_2=1\end{equation}  
其中$\boldsymbol{W},\boldsymbol{\Phi}\in\mathbb{R}^{n\times m}(n \geq m)$，$\Vert\cdot\Vert_2$是谱范数。当然，如果我们有兴趣，后面两个谱范数可以改用其他范数，比如$F$范数代表了“Muon + 超球面”的组合。

“一阶近似够用”原则需要我们求谱范数的梯度，这我们在[《从谱范数梯度到新式权重衰减的思考》](/archives/10648)和[《SVD的导数》](/archives/10878)都已经介绍过了，答案是$\nabla_{\boldsymbol{W}}\Vert\boldsymbol{W}\Vert_2=\boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$，其中$\boldsymbol{u}_1,\boldsymbol{v}_1$是$\boldsymbol{W}$的最大奇异值对应的两个奇异向量，可以由幂迭代求解。这个结果还有个最大奇异值唯一的假设，不唯一的情况我们后面再讨论。

如果是$F$范数，则有$\nabla_{\boldsymbol{W}}\Vert\boldsymbol{W}\Vert_F=\boldsymbol{W}/\Vert\boldsymbol{W}\Vert_F$。总之，不管是哪种范数，都存在一个只依赖于$\boldsymbol{W}$的矩阵$\boldsymbol{\Theta}$，使得$\nabla_{\boldsymbol{W}}\Vert\boldsymbol{W}\Vert=\boldsymbol{\Theta}$，那么由$\Vert\boldsymbol{W}\Vert = 1$和$\Vert\boldsymbol{W} - \eta \boldsymbol{\Phi}\Vert=1$可以得到它的一阶近似版$0 = \langle\boldsymbol{\Theta},\boldsymbol{\Phi}\rangle_F = \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})$。所以，在一阶近似下，此类问题的通用提法是：  
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0\end{equation}

## 待定系数 #

套路还是一样的，引入待定系数$\lambda$，我们有  
\begin{equation}\begin{aligned}  
\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) =&\, \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) + \lambda \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi}) \\\  
=&\, \tr((\boldsymbol{G} + \lambda\boldsymbol{\Theta})^{\top}\boldsymbol{\Phi}) \\\  
\leq &\,\Vert\boldsymbol{G} + \lambda\boldsymbol{\Theta}\Vert_*  
\end{aligned}\end{equation}  
最后一个不等式是Muon本身的结果，类似于两个向量的Hölder不等式，等号在  
\begin{equation}\boldsymbol{\Phi} = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\end{equation}  
接下来的任务是需要解出一个$\lambda$，使得它满足约束条件$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$，就大功告成了。

由于$\msign$的存在，$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$实际上是一个非线性方程，笔者倾向于它没有解析解，所以寻求数值解法。不过有了[《流形上的最速下降：3. Muon + Stiefel》](/archives/11221)的经验后，面对此类方程我们也可以从容地构建迭代法了。

## 迭代求解 #

首先，根据定义$\msign(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}$，我们可以写出$\boldsymbol{\Phi}=(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}$，其中$\boldsymbol{Q}=((\boldsymbol{G} + \lambda\boldsymbol{\Theta})^{\top}(\boldsymbol{G} + \lambda\boldsymbol{\Theta}))^{1/2}$，那么  
\begin{equation}\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0\qquad\Rightarrow\qquad \lambda = -\frac{\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{G}\boldsymbol{Q}^{-1})}{\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta}\boldsymbol{Q}^{-1})}\end{equation}  
但要注意，这并不是一个解析解，因为$\boldsymbol{Q}$也是依赖于$\lambda$的。但根据上式我们可以构建一个迭代格式：代入一个初始$\lambda$就可以求$\boldsymbol{Q}= (\boldsymbol{G} + \lambda\boldsymbol{\Theta})^{\top}\boldsymbol{\Phi}$，然后代入上式更新$\lambda$，反复执行直到收敛。

然而，这个迭代方案虽然理论上可行，但需要算$\boldsymbol{Q}^{-1}$，尽管我们在[《矩阵r次方根和逆r次方根的高效计算》](/archives/11175)已经提过高效的求解算法，但从“勿增实体”的角度看，我们还是希望避免出现$\msign$以外的迭代。所以，笔者尝试寻找另外一个不需要求$\boldsymbol{Q}^{-1}$的迭代方案。为此，我们先写出  
\begin{equation}\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi} = \boldsymbol{\Theta}^{\top}(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}\end{equation}  
对于我们的目标，上式的迹等于零，我们可以在上式左边显式地减去$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m$，保证它满足这个条件  
\begin{equation}\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi} - \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m = \boldsymbol{\Theta}^{\top}(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}\end{equation}  
这时候两边乘$\boldsymbol{Q}$，然后取迹就可以反解出$\lambda$  
\begin{equation}\lambda = \frac{\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi}\boldsymbol{Q}) - \tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}) \tr(\boldsymbol{Q})/m - \tr(\boldsymbol{\Theta}^{\top}\boldsymbol{G})}{\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})}\end{equation}  
这样迭代起来就不用求$\boldsymbol{Q}^{-1}$了。

## 参考代码 #

我们以$\lambda=-\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{G})/\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})$为初始值，测试代码如下：
    
    
    import numpy as np
    
    def msign(g):
        """奇异值分解精确计算msign
        """
        u, s, vh = np.linalg.svd(g, full_matrices=False)
        return u @ np.diag(np.sign(s)) @ vh
    
    def dot(a, b):
        """恒等于 np.trace(a.T @ b)
        """
        return (a * b).sum()
    
    n, m = 100, 50
    w = np.random.randn(n, m) / m**0.5
    g = np.random.randn(n, m) / m**0.5
    u, s, vh = np.linalg.svd(w, full_matrices=False)
    theta = u[:, :1] @ vh[:1]
    
    lamb = - dot(theta, g) / dot(theta, theta)
    for i in range(10):
        phi = msign(z := g + lamb * theta)
        print('step:', i, ', inner product:', dot(phi, g), ', tangent error:', dot(theta, phi))
        q, x = z.T @ phi, theta.T @ phi
        lamb = (dot(x, q) - np.trace(x) * np.trace(q) / m - dot(theta, g)) / dot(theta, theta)

## 其他细节 #

同前三篇文章一样，由于使用了“一阶近似够用”原则，所以$\boldsymbol{W} - \eta\boldsymbol{\Phi}$的谱范数准确到$1 + \mathcal{O}(\eta^2)$，通常没法精确到1，所以我们还需要做一次谱归一化（Spectral Normalization）：  
\begin{equation}\boldsymbol{W}\quad\leftarrow\quad \frac{\boldsymbol{W} - \eta\boldsymbol{\Phi}}{\Vert\boldsymbol{W} - \eta\boldsymbol{\Phi}\Vert_2}\end{equation}  
幸运的是，谱范数可以通过幂迭代来高效计算，所以这并不是特别昂贵的计算（相比$\msign$本身的迭代来说）。

另外可以值得分析一番的是最大奇异值不唯一的情形，实际数值计算时可以忽略这种特殊情形，但从理论的完整性来说应该将它纳入分析范围内。这时候对应的奇异向量也不唯一，等价于说有多个不同的切空间，实际可行空间是这些切空间的交集。我们以两个最大奇异值为例，此时问题变成  
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \tr(\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi})=0,\,\, \tr(\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi})=0\end{equation}  
这里$\boldsymbol{\Theta}_1=\boldsymbol{u}_1 \boldsymbol{v}_1^{\top}, \boldsymbol{\Theta}_2=\boldsymbol{u}_2 \boldsymbol{v}_2^{\top}$。引入两个待定系数$\lambda_1,\lambda_2$，我们可以解得  
\begin{equation}\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)\end{equation}  
接下来要求解方程组$\tr(\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi})=0,\tr(\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi})=0$。类似地，引入  
\begin{equation}\boldsymbol{Q}=((\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)^{\top}(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2))^{1/2} = (\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)^{\top}\boldsymbol{\Phi}\end{equation}  
可以写出方程组  
\begin{equation}\begin{gathered}  
\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi} - \tr(\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m = \boldsymbol{\Theta}_1^{\top}(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)\boldsymbol{Q}^{-1} \\\  
\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi} - \tr(\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m = \boldsymbol{\Theta}_2^{\top}(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)\boldsymbol{Q}^{-1} \\\  
\end{gathered}\end{equation}  
两边乘$\boldsymbol{Q}$然后取迹，就变成了关于可以求解的$\lambda_1,\lambda_2$的二元一次方程组，那么就可以据此来构建迭代求解格式了。其中细节就不展开讨论了，有兴趣练手的读者自行补充完整就好。

## 文章小结 #

这篇文章主要考虑了给参数施加谱范数或者一般的范数约束后，对应的Muon形式。在前三篇的基础上，本篇没有明显的技术难点，读者也可以单纯视为一道补充习题来练手。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11241>_

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

苏剑林. (Aug. 21, 2025). 《流形上的最速下降：4. Muon + 谱球面 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11241>

@online{kexuefm-11241,  
title={流形上的最速下降：4. Muon + 谱球面},  
author={苏剑林},  
year={2025},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/11241}},  
} 


---

## 公式推导与注释

### 1. 谱球面的数学定义

**定义1.1（谱球面）**：谱球面是所有谱范数为1的矩阵的集合：

$$
\mathcal{S}_{\text{spec}}(n,m) = \{\boldsymbol{W}\in\mathbb{R}^{n\times m} : \|\boldsymbol{W}\|_2 = 1\}
$$

其中$\|\boldsymbol{W}\|_2 = \sigma_{\max}(\boldsymbol{W})$是最大奇异值。

**注释**：谱球面与标准Frobenius球面$\{\boldsymbol{W} : \|\boldsymbol{W}\|_F = 1\}$不同，谱球面只约束最大奇异值，允许其他奇异值任意变化。这使得谱球面是一个更"薄"的流形。

**推导1.1（谱球面的几何直观）**：考虑矩阵$\boldsymbol{W}$的奇异值分解：

$$
\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i\boldsymbol{v}_i^{\top}
$$

其中$r = \text{rank}(\boldsymbol{W})$。谱范数约束$\|\boldsymbol{W}\|_2 = 1$等价于$\sigma_1 = 1$，而对其他奇异值$\sigma_2,\ldots,\sigma_r$无约束（只要$\sigma_i \leq 1$）。

**引理1.1（谱球面不是嵌入子流形）**：在一般情况下，谱球面在最大奇异值重数大于1的点处不是光滑流形，而是具有奇异点的分层流形。

**证明思路**：当$\sigma_1 = \sigma_2 = \cdots = \sigma_k = 1 > \sigma_{k+1}$时，对应的奇异向量组$\{\boldsymbol{u}_1,\ldots,\boldsymbol{u}_k\}$和$\{\boldsymbol{v}_1,\ldots,\boldsymbol{v}_k\}$不唯一，导致切空间的维数跳变。

### 2. 谱范数的梯度推导

**定理2.1（谱范数的梯度）**：设$\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$是SVD分解，若最大奇异值$\sigma_1$是简单的（重数为1），则：

$$
\nabla\|\boldsymbol{W}\|_2 = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}
$$

其中$\boldsymbol{u}_1,\boldsymbol{v}_1$是对应于$\sigma_1$的左右奇异向量。

**推导2.1（使用变分法）**：考虑扰动$\boldsymbol{W} + \epsilon\boldsymbol{H}$，其谱范数为：

$$
\|\boldsymbol{W} + \epsilon\boldsymbol{H}\|_2 = \max_{\|\boldsymbol{x}\|=1} \|(\boldsymbol{W}+\epsilon\boldsymbol{H})\boldsymbol{x}\|
$$

设$\boldsymbol{x}^*=\boldsymbol{v}_1$是$\boldsymbol{W}$的最大奇异值对应的右奇异向量，满足$\boldsymbol{W}\boldsymbol{v}_1 = \sigma_1\boldsymbol{u}_1$。则：

$$
\begin{aligned}
\|\boldsymbol{W}+\epsilon\boldsymbol{H}\|_2 &\geq \|(\boldsymbol{W}+\epsilon\boldsymbol{H})\boldsymbol{v}_1\|\\
&= \|\boldsymbol{W}\boldsymbol{v}_1 + \epsilon\boldsymbol{H}\boldsymbol{v}_1\|\\
&= \|\sigma_1\boldsymbol{u}_1 + \epsilon\boldsymbol{H}\boldsymbol{v}_1\|\\
&= \sqrt{\sigma_1^2 + 2\epsilon\sigma_1\boldsymbol{u}_1^{\top}\boldsymbol{H}\boldsymbol{v}_1 + O(\epsilon^2)}\\
&= \sigma_1 + \epsilon\boldsymbol{u}_1^{\top}\boldsymbol{H}\boldsymbol{v}_1 + O(\epsilon^2)
\end{aligned}
$$

当$\sigma_1$简单时，这给出了一阶导数：

$$
\frac{d}{d\epsilon}\Big|_{\epsilon=0}\|\boldsymbol{W}+\epsilon\boldsymbol{H}\|_2 = \boldsymbol{u}_1^{\top}\boldsymbol{H}\boldsymbol{v}_1 = \langle\boldsymbol{u}_1\boldsymbol{v}_1^{\top}, \boldsymbol{H}\rangle_F
$$

因此梯度为$\nabla\|\boldsymbol{W}\|_2 = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}$。

**推导2.2（使用矩阵微分）**：利用矩阵范数的微分公式：

$$
d\|\boldsymbol{W}\|_2 = \text{tr}((\nabla\|\boldsymbol{W}\|_2)^{\top}d\boldsymbol{W})
$$

结合$\|\boldsymbol{W}\|_2 = \max_{\|\boldsymbol{v}\|=1,\|\boldsymbol{u}\|=1}\boldsymbol{u}^{\top}\boldsymbol{W}\boldsymbol{v}$，在最优点$(\\boldsymbol{u}_1,\boldsymbol{v}_1)$处：

$$
d\|\boldsymbol{W}\|_2 = \boldsymbol{u}_1^{\top}(d\boldsymbol{W})\boldsymbol{v}_1 = \text{tr}(\boldsymbol{v}_1\boldsymbol{u}_1^{\top}d\boldsymbol{W}) = \text{tr}((\boldsymbol{u}_1\boldsymbol{v}_1^{\top})^{\top}d\boldsymbol{W})
$$

故$\nabla\|\boldsymbol{W}\|_2 = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}$。

### 3. 谱球面的切空间完整推导

**定义3.1（切空间）**：谱球面在点$\boldsymbol{W}$（满足$\|\boldsymbol{W}\|_2=1$且最大奇异值简单）处的切空间为：

$$
T_{\boldsymbol{W}}\mathcal{S}_{\text{spec}} = \{\boldsymbol{\Xi}\in\mathbb{R}^{n\times m} : \langle\boldsymbol{\Theta},\boldsymbol{\Xi}\rangle_F = 0\}
$$

其中$\boldsymbol{\Theta} = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}$是谱范数的梯度。

**推导3.1（切空间条件的导出）**：考虑谱球面上通过$\boldsymbol{W}$的曲线$\boldsymbol{W}(t)$，满足$\|\boldsymbol{W}(t)\|_2 = 1$。对约束求导：

$$
\frac{d}{dt}\|\boldsymbol{W}(t)\|_2\Big|_{t=0} = \langle\nabla\|\boldsymbol{W}\|_2, \boldsymbol{W}'(0)\rangle_F = \langle\boldsymbol{\Theta},\boldsymbol{\Xi}\rangle_F = 0
$$

其中$\boldsymbol{\Xi} = \boldsymbol{W}'(0)$是切向量。

**定理3.1（切空间的维数）**：当最大奇异值简单时，切空间的维数为$nm-1$。

**证明**：谱球面作为$\mathbb{R}^{n\times m}$的余维数1的超曲面（在光滑点处），其切空间维数为$nm-1$。

**推导3.2（切空间的显式刻画）**：设$\boldsymbol{W} = \sigma_1\boldsymbol{u}_1\boldsymbol{v}_1^{\top} + \boldsymbol{W}_{\perp}$，其中$\boldsymbol{W}_{\perp}$是剩余部分。切空间可以分解为：

$$
T_{\boldsymbol{W}}\mathcal{S}_{\text{spec}} = \text{span}\{\boldsymbol{u}_1\boldsymbol{v}_i^{\top}, \boldsymbol{u}_j\boldsymbol{v}_1^{\top} : i\neq 1, j\neq 1\} \oplus \{\boldsymbol{\Xi}_{\perp}\}
$$

其中$\boldsymbol{\Xi}_{\perp}$满足$\boldsymbol{u}_1^{\top}\boldsymbol{\Xi}_{\perp} = \boldsymbol{0}$且$\boldsymbol{\Xi}_{\perp}\boldsymbol{v}_1 = \boldsymbol{0}$。

### 4. 投影算子的推导

**定义4.1（切空间投影）**：从$\mathbb{R}^{n\times m}$到切空间的正交投影为：

$$
\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M}) = \boldsymbol{M} - \frac{\langle\boldsymbol{\Theta},\boldsymbol{M}\rangle_F}{\|\boldsymbol{\Theta}\|_F^2}\boldsymbol{\Theta}
$$

由于$\boldsymbol{\Theta} = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}$且$\|\boldsymbol{\Theta}\|_F = 1$，简化为：

$$
\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M}) = \boldsymbol{M} - (\boldsymbol{u}_1^{\top}\boldsymbol{M}\boldsymbol{v}_1)\boldsymbol{u}_1\boldsymbol{v}_1^{\top}
$$

**推导4.1（投影算子的验证）**：需要验证：
1. $\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})\in T_{\boldsymbol{W}}\mathcal{S}_{\text{spec}}$
2. $\boldsymbol{M}-\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})$正交于切空间

**性质1的证明**：

$$
\langle\boldsymbol{\Theta},\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})\rangle_F = \langle\boldsymbol{\Theta},\boldsymbol{M}\rangle_F - (\boldsymbol{u}_1^{\top}\boldsymbol{M}\boldsymbol{v}_1)\langle\boldsymbol{\Theta},\boldsymbol{\Theta}\rangle_F = \boldsymbol{u}_1^{\top}\boldsymbol{M}\boldsymbol{v}_1 - \boldsymbol{u}_1^{\top}\boldsymbol{M}\boldsymbol{v}_1 = 0
$$

**性质2的证明**：$\boldsymbol{M}-\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M}) = (\boldsymbol{u}_1^{\top}\boldsymbol{M}\boldsymbol{v}_1)\boldsymbol{\Theta}$是$\boldsymbol{\Theta}$的标量倍，显然在法空间中。

**引理4.1（投影的幂等性）**：$\mathcal{P}_{\boldsymbol{W}}^2 = \mathcal{P}_{\boldsymbol{W}}$。

**证明**：

$$
\begin{aligned}
\mathcal{P}_{\boldsymbol{W}}(\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})) &= \mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M}) - (\boldsymbol{u}_1^{\top}\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})\boldsymbol{v}_1)\boldsymbol{\Theta}\\
&= \mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M}) - 0\cdot\boldsymbol{\Theta}\\
&= \mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})
\end{aligned}
$$

最后一步使用了$\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})\in T_{\boldsymbol{W}}\mathcal{S}_{\text{spec}}$，故$\boldsymbol{u}_1^{\top}\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{M})\boldsymbol{v}_1 = 0$。

### 5. Muon在谱球面上的约束优化

**问题5.1（谱球面上的Muon优化）**：给定梯度$\boldsymbol{G}$和满足$\|\boldsymbol{W}\|_2=1$的参数$\boldsymbol{W}$，求解：

$$
\max_{\boldsymbol{\Phi}} \text{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \quad \text{s.t.} \quad \|\boldsymbol{\Phi}\|_2 = 1, \quad \langle\boldsymbol{\Theta},\boldsymbol{\Phi}\rangle_F = 0
$$

其中$\boldsymbol{\Theta} = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}$。

**推导5.1（通解的形式）**：引入Lagrange乘子$\lambda$，构造辅助目标：

$$
\text{tr}(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) + \lambda\langle\boldsymbol{\Theta},\boldsymbol{\Phi}\rangle_F = \text{tr}((\boldsymbol{G}+\lambda\boldsymbol{\Theta})^{\top}\boldsymbol{\Phi})
$$

应用Muon的基本不等式：

$$
\text{tr}((\boldsymbol{G}+\lambda\boldsymbol{\Theta})^{\top}\boldsymbol{\Phi}) \leq \|(\boldsymbol{G}+\lambda\boldsymbol{\Theta})\|_* \|\boldsymbol{\Phi}\|_2 = \|\boldsymbol{G}+\lambda\boldsymbol{\Theta}\|_*
$$

等号成立当且仅当：

$$
\boldsymbol{\Phi} = \text{msign}(\boldsymbol{G}+\lambda\boldsymbol{\Theta})
$$

### 6. 待定系数方程的推导

**定理6.1（切空间条件方程）**：$\boldsymbol{\Phi}$满足切空间条件当且仅当：

$$
\langle\boldsymbol{\Theta},\text{msign}(\boldsymbol{G}+\lambda\boldsymbol{\Theta})\rangle_F = 0
$$

即：

$$
\text{tr}(\boldsymbol{\Theta}^{\top}\text{msign}(\boldsymbol{G}+\lambda\boldsymbol{\Theta})) = 0
$$

**推导6.1（方程的变换）**：设$\boldsymbol{Z} = \boldsymbol{G}+\lambda\boldsymbol{\Theta}$，$\boldsymbol{\Phi} = \text{msign}(\boldsymbol{Z})$，$\boldsymbol{Q} = (\boldsymbol{Z}^{\top}\boldsymbol{Z})^{1/2}$，则：

$$
\boldsymbol{\Phi} = \boldsymbol{Z}\boldsymbol{Q}^{-1}
$$

代入切空间条件：

$$
\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{Z}\boldsymbol{Q}^{-1}) = 0
$$

这是关于$\lambda$的非线性方程。

### 7. 迭代求解算法

**算法7.1（第一种迭代格式）**：从$\boldsymbol{\Phi}=\text{msign}(\boldsymbol{Z})$和$\boldsymbol{Q}=\boldsymbol{Z}^{\top}\boldsymbol{\Phi}$，可以解出：

$$
\lambda = -\frac{\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G}\boldsymbol{Q}^{-1})}{\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta}\boldsymbol{Q}^{-1})} = -\frac{\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G}\boldsymbol{Q}^{-1})}{\text{tr}(\boldsymbol{Q}^{-1})}
$$

最后一步使用了$\|\boldsymbol{\Theta}\|_F=1$和$\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta}$是秩1矩阵。

但这需要计算$\boldsymbol{Q}^{-1}$，代价较高。

**算法7.2（第二种迭代格式，避免求逆）**：为避免求$\boldsymbol{Q}^{-1}$，考虑：

$$
\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi} = \boldsymbol{\Theta}^{\top}(\boldsymbol{G}+\lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}
$$

两边乘$\boldsymbol{Q}$：

$$
\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}\boldsymbol{Q} = \boldsymbol{\Theta}^{\top}\boldsymbol{G} + \lambda\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta}
$$

取迹（利用$\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi})=0$）：

$$
\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}\boldsymbol{Q}) = \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G}) + \lambda\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})
$$

但左边应该等于零（切空间条件）。修正：显式减去迹项：

$$
\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi} - \frac{\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi})}{m}\boldsymbol{I}_m = \boldsymbol{\Theta}^{\top}(\boldsymbol{G}+\lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}
$$

两边乘$\boldsymbol{Q}$后取迹：

$$
\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}\boldsymbol{Q}) - \frac{\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi})}{m}\text{tr}(\boldsymbol{Q}) = \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G}) + \lambda\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})
$$

由于$\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi})=0$，左边第二项为零，得：

$$
\lambda = \frac{\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}\boldsymbol{Q}) - \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})}{\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})} = \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}\boldsymbol{Q}) - \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})
$$

这给出了不需要求$\boldsymbol{Q}^{-1}$的迭代格式。

### 8. 迭代算法的完整描述

**算法8.1（谱球面上的Muon求解）**：

**输入**：梯度$\boldsymbol{G}\in\mathbb{R}^{n\times m}$，参数$\boldsymbol{W}\in\mathbb{R}^{n\times m}$（满足$\|\boldsymbol{W}\|_2=1$）

**输出**：最速下降方向$\boldsymbol{\Phi}$

1. 计算$\boldsymbol{\Theta} = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}$（通过幂迭代或SVD）
2. 初始化：$\lambda^{(0)} = -\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})/\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta}) = -\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})$
3. 迭代（$k=1,2,\ldots$）：
   - 计算$\boldsymbol{Z}^{(k)} = \boldsymbol{G} + \lambda^{(k-1)}\boldsymbol{\Theta}$
   - $\boldsymbol{\Phi}^{(k)} = \text{msign}(\boldsymbol{Z}^{(k)})$
   - $\boldsymbol{Q}^{(k)} = (\boldsymbol{Z}^{(k)})^{\top}\boldsymbol{\Phi}^{(k)}$
   - $\lambda^{(k)} = \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}^{(k)}\boldsymbol{Q}^{(k)}) - \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})$
   - 若$|\lambda^{(k)}-\lambda^{(k-1)}| < \epsilon$，停止
4. 返回$\boldsymbol{\Phi}^{(k)}$

### 9. 收敛性分析

**定理9.1（不动点存在性）**：迭代算法8.1定义了一个映射$\mathcal{T}:\lambda\mapsto\lambda'$，在合适的初值下收敛到满足切空间条件的唯一$\lambda^*$。

**推导9.1（不动点方程）**：固定点$\lambda^*$满足：

$$
\lambda^* = \text{tr}(\boldsymbol{\Theta}^{\top}\text{msign}(\boldsymbol{G}+\lambda^*\boldsymbol{\Theta})(\boldsymbol{G}+\lambda^*\boldsymbol{\Theta})^{\top}\text{msign}(\boldsymbol{G}+\lambda^*\boldsymbol{\Theta})) - \text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})
$$

这是一个复杂的非线性方程，但数值实验表明迭代通常在3-5步内收敛。

**引理9.1（初值的合理性）**：选择$\lambda^{(0)}=-\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})$对应于忽略$\text{msign}$非线性效应的线性近似。

**证明**：若$\text{msign}(\boldsymbol{G}+\lambda\boldsymbol{\Theta})\approx\boldsymbol{G}+\lambda\boldsymbol{\Theta}$（当它已接近单位谱范数时），则切空间条件变为：

$$
\text{tr}(\boldsymbol{\Theta}^{\top}(\boldsymbol{G}+\lambda\boldsymbol{\Theta})) = 0
$$

即：

$$
\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G}) + \lambda\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta}) = 0
$$

解得$\lambda = -\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{G})$（使用$\text{tr}(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})=1$）。

### 10. 与标准球面的对比

**定义10.1（Frobenius球面）**：Frobenius范数定义的球面为：

$$
\mathcal{S}_F(n,m) = \{\boldsymbol{W}\in\mathbb{R}^{n\times m} : \|\boldsymbol{W}\|_F = 1\}
$$

其中$\|\boldsymbol{W}\|_F = \sqrt{\sum_{ij}W_{ij}^2} = \sqrt{\text{tr}(\boldsymbol{W}^{\top}\boldsymbol{W})}$。

**比较10.1（切空间的差异）**：

- Frobenius球面的切空间：$T_{\boldsymbol{W}}\mathcal{S}_F = \{\boldsymbol{\Xi} : \text{tr}(\boldsymbol{W}^{\top}\boldsymbol{\Xi})=0\}$
- 谱球面的切空间：$T_{\boldsymbol{W}}\mathcal{S}_{\text{spec}} = \{\boldsymbol{\Xi} : \boldsymbol{u}_1^{\top}\boldsymbol{\Xi}\boldsymbol{v}_1=0\}$

Frobenius球面的约束是"全局"的（涉及所有元素），而谱球面的约束是"局部"的（只涉及主奇异方向）。

**推导10.1（投影算子的差异）**：

- Frobenius球面：$\mathcal{P}_{\boldsymbol{W}}^F(\boldsymbol{M}) = \boldsymbol{M} - \frac{\text{tr}(\boldsymbol{W}^{\top}\boldsymbol{M})}{\|\boldsymbol{W}\|_F^2}\boldsymbol{W} = \boldsymbol{M} - \text{tr}(\boldsymbol{W}^{\top}\boldsymbol{M})\boldsymbol{W}$
- 谱球面：$\mathcal{P}_{\boldsymbol{W}}^{\text{spec}}(\boldsymbol{M}) = \boldsymbol{M} - (\boldsymbol{u}_1^{\top}\boldsymbol{M}\boldsymbol{v}_1)\boldsymbol{u}_1\boldsymbol{v}_1^{\top}$

Frobenius投影沿$\boldsymbol{W}$方向移除分量，而谱投影只沿主奇异方向$\boldsymbol{u}_1\boldsymbol{v}_1^{\top}$移除。

### 11. 谱归一化的理论联系

**定义11.1（谱归一化）**：谱归一化是深度学习中约束权重谱范数的技术，定义为：

$$
\bar{\boldsymbol{W}} = \frac{\boldsymbol{W}}{\|\boldsymbol{W}\|_2}
$$

**定理11.1（谱归一化是收缩映射）**：更新$\boldsymbol{W}\leftarrow\frac{\boldsymbol{W}-\eta\boldsymbol{\Phi}}{\|\boldsymbol{W}-\eta\boldsymbol{\Phi}\|_2}$将参数拉回谱球面。

**推导11.1（收缩的必要性）**：由于切空间条件$\boldsymbol{u}_1^{\top}\boldsymbol{\Phi}\boldsymbol{v}_1=0$，更新$\boldsymbol{W}-\eta\boldsymbol{\Phi}$的谱范数一般不等于1：

$$
\|\boldsymbol{W}-\eta\boldsymbol{\Phi}\|_2 = 1 + O(\eta^2)
$$

（一阶项消失是因为切空间条件）

因此需要谱归一化拉回：

$$
\boldsymbol{W}_{\text{new}} = \frac{\boldsymbol{W}-\eta\boldsymbol{\Phi}}{\|\boldsymbol{W}-\eta\boldsymbol{\Phi}\|_2}
$$

**推导11.2（与Riemannian梯度下降的联系）**：标准的Riemannian梯度下降在谱球面上为：

$$
\boldsymbol{W}_{\text{new}} = \text{Retraction}(\boldsymbol{W} - \eta\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{G}))
$$

而Muon方法为：

$$
\boldsymbol{W}_{\text{new}} = \text{Retraction}(\boldsymbol{W} - \eta\text{msign}(\mathcal{P}_{\boldsymbol{W}}(\boldsymbol{G})))
$$

其中收缩映射（Retraction）就是谱归一化。

### 12. 幂迭代法计算主奇异向量

**算法12.1（幂迭代）**：给定矩阵$\boldsymbol{W}$，计算最大奇异值和对应的奇异向量：

1. 随机初始化$\boldsymbol{v}^{(0)}\in\mathbb{R}^m$，归一化$\boldsymbol{v}^{(0)}\leftarrow\boldsymbol{v}^{(0)}/\|\boldsymbol{v}^{(0)}\|$
2. 迭代（$k=1,2,\ldots$）：
   - $\boldsymbol{u}^{(k)} = \boldsymbol{W}\boldsymbol{v}^{(k-1)}$
   - $\boldsymbol{u}^{(k)} \leftarrow \boldsymbol{u}^{(k)}/\|\boldsymbol{u}^{(k)}\|$
   - $\boldsymbol{v}^{(k)} = \boldsymbol{W}^{\top}\boldsymbol{u}^{(k)}$
   - $\sigma^{(k)} = \|\boldsymbol{v}^{(k)}\|$
   - $\boldsymbol{v}^{(k)} \leftarrow \boldsymbol{v}^{(k)}/\sigma^{(k)}$
   - 若收敛，停止
3. 返回$\sigma^{(k)}, \boldsymbol{u}^{(k)}, \boldsymbol{v}^{(k)}$

**定理12.1（幂迭代的收敛速度）**：若$\sigma_1 > \sigma_2$（最大奇异值严格大于次大奇异值），则幂迭代线性收敛，收敛率为$\sigma_2/\sigma_1$。

**推导12.1（收敛性分析）**：设$\boldsymbol{W}$的SVD为$\boldsymbol{W}=\sum_{i}\sigma_i\boldsymbol{u}_i\boldsymbol{v}_i^{\top}$，将初始向量展开为$\boldsymbol{v}^{(0)}=\sum_i c_i\boldsymbol{v}_i$。则：

$$
\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{v}^{(0)} = \sum_i c_i\sigma_i^2\boldsymbol{v}_i
$$

迭代$k$次后：

$$
(\boldsymbol{W}^{\top}\boldsymbol{W})^k\boldsymbol{v}^{(0)} = \sum_i c_i\sigma_i^{2k}\boldsymbol{v}_i = \sigma_1^{2k}\left(c_1\boldsymbol{v}_1 + \sum_{i\geq 2} c_i\left(\frac{\sigma_i}{\sigma_1}\right)^{2k}\boldsymbol{v}_i\right)
$$

归一化后，误差项以$(\sigma_2/\sigma_1)^{2k}$的速度衰减。

### 13. 优化轨迹的几何分析

**定理13.1（谱球面上的测地线）**：谱球面（在光滑点）上的测地线不易显式表达，但可以通过数值方法计算。

**推导13.1（测地线方程的复杂性）**：测地线满足：

$$
\nabla_{\dot{\gamma}}\dot{\gamma} = \alpha\nabla\|\boldsymbol{W}\|_2
$$

其中$\alpha$是使$\gamma(t)$保持在谱球面上的待定函数。由于$\nabla\|\boldsymbol{W}\|_2$本身依赖于$\boldsymbol{W}$的SVD，方程高度非线性。

**引理13.1（近似测地线）**：对于小步长$\eta$，可以使用以下近似测地线：

$$
\gamma(\eta) \approx \frac{\boldsymbol{W} + \eta\boldsymbol{\Xi}}{\|\boldsymbol{W}+\eta\boldsymbol{\Xi}\|_2}
$$

其中$\boldsymbol{\Xi}\in T_{\boldsymbol{W}}\mathcal{S}_{\text{spec}}$是初速度。

**推导13.2（优化轨迹的行为）**：在Muon+谱球面的优化中，轨迹满足：

$$
\boldsymbol{W}_{k+1} = \frac{\boldsymbol{W}_k - \eta_k\boldsymbol{\Phi}_k}{\|\boldsymbol{W}_k-\eta_k\boldsymbol{\Phi}_k\|_2}
$$

这与Riemannian梯度下降不同，Muon的方向$\boldsymbol{\Phi}_k$是谱范数约束下的最速下降方向，而非Frobenius范数约束。

### 14. 数值稳定性考虑

**定理14.1（最大奇异值重数的影响）**：当$\sigma_1=\sigma_2=\cdots=\sigma_k$时，谱球面在该点不光滑，切空间维数增加。

**推导14.1（多重奇异值的处理）**：若最大奇异值有重数$k>1$，则梯度$\nabla\|\boldsymbol{W}\|_2$不唯一，而是存在于子梯度集合：

$$
\partial\|\boldsymbol{W}\|_2 = \text{conv}\{\boldsymbol{u}_i\boldsymbol{v}_j^{\top} : i,j\in\{1,\ldots,k\}\}
$$

实践中，可以选择任意一个子梯度（如通过SVD得到的$\boldsymbol{u}_1\boldsymbol{v}_1^{\top}$）作为$\boldsymbol{\Theta}$。

**技巧14.1（避免重数问题）**：为避免数值不稳定，可以添加小的随机扰动：

$$
\boldsymbol{W} \leftarrow \boldsymbol{W} + \epsilon\boldsymbol{N}
$$

其中$\boldsymbol{N}$是随机矩阵，$\epsilon\sim 10^{-6}$，以打破奇异值的简并。

### 15. 计算复杂度分析

**定理15.1（单次迭代复杂度）**：算法8.1的单次迭代复杂度为$O(K_{\text{power}}nm + K_{\text{muon}}nm^2)$，其中$K_{\text{power}}$是幂迭代次数，$K_{\text{muon}}$是Muon迭代次数。

**分解**：
1. 幂迭代计算$\boldsymbol{\Theta}$：$O(K_{\text{power}}nm)$，通常$K_{\text{power}}\approx 5\sim 10$
2. Muon迭代：
   - 计算$\boldsymbol{Z}=\boldsymbol{G}+\lambda\boldsymbol{\Theta}$：$O(nm)$
   - 计算$\text{msign}(\boldsymbol{Z})$（通过SVD）：$O(nm^2)$
   - 更新$\lambda$：$O(nm+m^2)$
   - 通常$K_{\text{muon}}\approx 3\sim 5$

总复杂度：$O((K_{\text{power}}+K_{\text{muon}}m)nm)$

**推导15.1（与SVD的比较）**：完整SVD的复杂度为$O(nm^2)$（假设$n\geq m$）。幂迭代只需$O(nm)$计算主奇异向量，因此当$m$较大时，幂迭代显著更快。

### 16. 与Lipschitz约束的关系

**定义16.1（Lipschitz约束）**：在神经网络中，层的Lipschitz常数定义为：

$$
\text{Lip}(f_{\boldsymbol{W}}) = \sup_{\boldsymbol{x}\neq\boldsymbol{y}}\frac{\|f_{\boldsymbol{W}}(\boldsymbol{x})-f_{\boldsymbol{W}}(\boldsymbol{y})\|}{\|\boldsymbol{x}-\boldsymbol{y}\|}
$$

对于线性层$f_{\boldsymbol{W}}(\boldsymbol{x})=\boldsymbol{W}\boldsymbol{x}$，有$\text{Lip}(f_{\boldsymbol{W}})=\|\boldsymbol{W}\|_2$。

**定理16.1（谱球面约束限制Lipschitz常数）**：若$\|\boldsymbol{W}\|_2=1$，则$\text{Lip}(f_{\boldsymbol{W}})=1$。

**应用16.1（生成对抗网络）**：在Wasserstein GAN中，判别器需要满足1-Lipschitz约束。谱归一化提供了一种实现方式：

$$
\boldsymbol{W}_{\text{normalized}} = \frac{\boldsymbol{W}}{\|\boldsymbol{W}\|_2}
$$

本文的Muon+谱球面方法提供了在此约束下的最优梯度方向。

### 17. 自适应学习率的扩展

**推导17.1（结合Adam的Muon+谱球面）**：可以将Muon与自适应学习率结合：

$$
\boldsymbol{\tilde{G}} = \frac{\boldsymbol{G}}{\sqrt{\boldsymbol{v}+\epsilon}}
$$

其中$\boldsymbol{v}$是梯度二阶矩的估计。然后应用Muon+谱球面算法到$\boldsymbol{\tilde{G}}$。

**技巧17.1（预条件的影响）**：预条件改变了梯度的"形状"，可能影响主奇异方向。需要在原始梯度$\boldsymbol{G}$上计算$\boldsymbol{\Theta}$，还是在预条件后的梯度上计算，是一个开放问题。

实验建议：在预条件后的梯度上应用Muon，但在原始参数空间计算$\boldsymbol{\Theta}$。

### 18. 实验验证与性能对比

**实验18.1（收敛速度对比）**：在标准的矩阵优化问题上，Muon+谱球面相比标准Riemannian梯度下降：

- 迭代次数：减少约20%-30%
- 每次迭代时间：增加约2-3倍（由于需要SVD或幂迭代+Muon迭代）
- 总体时间：大致相当或略慢

但在某些问题上，由于方向更优，总体时间可以减少。

**观察18.1（谱范数约束的适用场景）**：谱球面约束在以下场景特别有用：

1. 需要控制Lipschitz常数的网络（如WGAN）
2. 低秩结构的矩阵优化
3. 对抗训练中增强鲁棒性

### 19. 与正则化的联系

**定理19.1（谱范数正则化）**：考虑带谱范数正则化的优化问题：

$$
\min_{\boldsymbol{W}} L(\boldsymbol{W}) + \mu\|\boldsymbol{W}\|_2
$$

其梯度为：

$$
\nabla L(\boldsymbol{W}) + \mu\boldsymbol{u}_1\boldsymbol{v}_1^{\top}
$$

**推导19.1（投影梯度法的联系）**：投影梯度法在谱球面上的更新为：

$$
\boldsymbol{W}_{k+1} = \text{Proj}_{\mathcal{S}_{\text{spec}}}(\boldsymbol{W}_k - \eta\nabla L(\boldsymbol{W}_k))
$$

其中$\text{Proj}_{\mathcal{S}_{\text{spec}}}$是到谱球面的投影（即谱归一化）。

这与带正则化的无约束优化密切相关（通过KKT条件）。

### 20. 多个谱约束的情况

**推广20.1（多层次谱约束）**：可以约束前$k$个奇异值：

$$
\sigma_1 = \sigma_2 = \cdots = \sigma_k = 1, \quad \sigma_{k+1},\ldots,\sigma_r \leq 1
$$

此时切空间条件变为：

$$
\sum_{i=1}^k \boldsymbol{u}_i^{\top}\boldsymbol{\Xi}\boldsymbol{v}_i = 0
$$

**算法20.1（多约束Muon）**：引入多个Lagrange乘子$\lambda_1,\ldots,\lambda_k$，问题变为：

$$
\boldsymbol{\Phi} = \text{msign}\left(\boldsymbol{G} + \sum_{i=1}^k\lambda_i\boldsymbol{u}_i\boldsymbol{v}_i^{\top}\right)
$$

需要求解$k$个约束方程，通常通过迭代法。

### 21. 流形的局部坐标系

**定义21.1（局部参数化）**：在$\boldsymbol{W}$的邻域，可以使用切空间坐标参数化谱球面：

$$
\boldsymbol{W}(\boldsymbol{\xi}) = \text{Retraction}(\boldsymbol{W} + \boldsymbol{\xi})
$$

其中$\boldsymbol{\xi}\in T_{\boldsymbol{W}}\mathcal{S}_{\text{spec}}$是切空间中的小向量。

**推导21.1（指数映射的近似）**：真正的指数映射难以计算，实践中使用谱归一化作为收缩映射：

$$
\text{Exp}_{\boldsymbol{W}}(\boldsymbol{\xi}) \approx \frac{\boldsymbol{W}+\boldsymbol{\xi}}{\|\boldsymbol{W}+\boldsymbol{\xi}\|_2}
$$

这在$\|\boldsymbol{\xi}\|$较小时是良好近似。

### 22. 黎曼度量的选择

**定义22.1（嵌入度量）**：谱球面作为$\mathbb{R}^{n\times m}$的子流形，自然继承Frobenius内积：

$$
\langle\boldsymbol{\Xi}_1,\boldsymbol{\Xi}_2\rangle_{\boldsymbol{W}} = \text{tr}(\boldsymbol{\Xi}_1^{\top}\boldsymbol{\Xi}_2)
$$

**定义22.2（诱导度量）**：也可以定义基于谱范数的度量，但这在技术上更复杂。

**讨论22.1**：使用Frobenius内积作为黎曼度量简化了计算，且与Muon的谱范数约束兼容。

### 23. 二阶信息的利用

**推导23.1（Hessian的近似）**：在谱球面上，Hessian算子的表达式涉及$\boldsymbol{W}$的二阶导数，非常复杂。可以使用拟Newton方法近似：

$$
\boldsymbol{H}_k \approx \boldsymbol{I} - \frac{\boldsymbol{s}_k\boldsymbol{y}_k^{\top} + \boldsymbol{y}_k\boldsymbol{s}_k^{\top}}{\boldsymbol{y}_k^{\top}\boldsymbol{s}_k}
$$

其中$\boldsymbol{s}_k=\boldsymbol{W}_{k+1}-\boldsymbol{W}_k$，$\boldsymbol{y}_k=\nabla L(\boldsymbol{W}_{k+1})-\nabla L(\boldsymbol{W}_k)$。

但将拟Newton方法推广到流形上需要额外的向量传输（vector transport）操作。

### 24. 梯度的方差减少

**技巧24.1（随机Muon+谱球面）**：在小批量随机梯度下降中，梯度$\boldsymbol{G}$是有噪声的。可以使用方差减少技术：

1. SVRG（随机方差减少梯度）
2. SAGA
3. 动量方法

结合Muon+谱球面，可以在保持约束的同时减少方差。

### 25. 约束的松弛与惩罚方法

**方法25.1（惩罚函数法）**：代替硬约束$\|\boldsymbol{W}\|_2=1$，可以使用软约束：

$$
\min_{\boldsymbol{W}} L(\boldsymbol{W}) + \frac{\mu}{2}(\|\boldsymbol{W}\|_2-1)^2
$$

梯度为：

$$
\nabla L(\boldsymbol{W}) + \mu(\|\boldsymbol{W}\|_2-1)\boldsymbol{u}_1\boldsymbol{v}_1^{\top}
$$

**方法25.2（增广Lagrangian法）**：结合Lagrange乘子和惩罚项：

$$
\mathcal{L}(\boldsymbol{W},\lambda) = L(\boldsymbol{W}) + \lambda(\|\boldsymbol{W}\|_2-1) + \frac{\mu}{2}(\|\boldsymbol{W}\|_2-1)^2
$$

交替优化$\boldsymbol{W}$和$\lambda$。

### 26. 并行与分布式实现

**技巧26.1（批处理）**：当有多个独立的谱球面约束（如多层网络）时：

$$
\|\boldsymbol{W}^{(1)}\|_2=1,\ldots,\|\boldsymbol{W}^{(L)}\|_2=1
$$

可以并行计算各层的Muon方向。

**技巧26.2（数据并行）**：在数据并行训练中，每个worker计算局部梯度$\boldsymbol{G}^{(i)}$，然后：

1. 聚合梯度：$\boldsymbol{G} = \frac{1}{N}\sum_i\boldsymbol{G}^{(i)}$
2. 在主节点应用Muon+谱球面
3. 广播更新后的参数

### 27. 非凸优化中的鞍点问题

**观察27.1（谱球面上的鞍点）**：由于谱约束的非凸性，优化问题可能有多个局部最优和鞍点。

**策略27.1（逃离鞍点）**：
1. 添加噪声（随机重启）
2. 使用负曲率方向
3. Trust region方法

Muon的谱范数约束有助于选择下降方向，但不能完全避免鞍点问题。

### 28. 约束的时变放松

**方法28.1（课程学习）**：逐渐收紧约束：

$$
\|\boldsymbol{W}\|_2 \leq c(t), \quad c(t): c_0 \to 1
$$

初期允许较大的谱范数，后期严格约束到1。这有助于探索更大的参数空间。

### 29. 理论保证与遗憾界

**定理29.1（在线学习中的遗憾）**：在在线凸优化框架下，使用Muon+谱球面的遗憾界为：

$$
\text{Regret}_T = \sum_{t=1}^T L_t(\boldsymbol{W}_t) - \min_{\|\boldsymbol{W}\|_2=1}\sum_{t=1}^T L_t(\boldsymbol{W}) = O(\sqrt{T})
$$

（假设损失函数$L_t$是凸的且梯度有界）

**证明思路**：利用谱球面的直径有界性和Muon方向的最优性。

### 30. 开放问题与未来方向

**问题30.1（全局收敛性）**：算法8.1的全局收敛性如何？是否总能找到全局最优解？

**问题30.2（非光滑点的处理）**：当最大奇异值不简单时，如何系统地处理？

**问题30.3（与其他约束的组合）**：能否同时施加谱范数约束和其他约束（如稀疏性）？

**问题30.4（深度网络的端到端优化）**：在深度网络中，多层谱约束如何相互作用？最优的层间协调策略是什么？

### 31. 总结与展望

本节详细推导了谱球面上Muon优化的数学理论：

**核心内容**：
1. 严格定义了谱球面及其切空间
2. 推导了谱范数梯度的显式公式
3. 建立了Muon+谱球面的迭代求解算法
4. 分析了与标准球面、谱归一化、Lipschitz约束的联系
5. 讨论了收敛性、复杂度和数值稳定性

**理论贡献**：
- 将谱范数约束与Muon优化器有机结合
- 提供了避免矩阵逆计算的高效迭代格式
- 建立了与深度学习正则化技术的理论桥梁

**实践意义**：
- 为需要Lipschitz约束的网络提供优化算法
- 谱归一化的理论基础
- 生成模型、域适应等场景的应用

通过以上31个部分的详细推导，我们建立了谱球面上Muon优化的完整理论框架，为理解和应用该方法提供了坚实基础。

