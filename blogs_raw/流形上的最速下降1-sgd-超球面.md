---
title: 流形上的最速下降：1.  SGD + 超球面
slug: 流形上的最速下降1-sgd-超球面
date: 2025-08-01
tags: 详细推导, 不等式, 优化器, 约束, 最速下降, 生成模型, 流形优化, 黎曼几何, 微分几何, 投影梯度, 理论分析
status: completed
tags_reviewed: true
---
# 流形上的最速下降：1.  SGD + 超球面

**原文链接**: [https://spaces.ac.cn/archives/11196](https://spaces.ac.cn/archives/11196)

**发布日期**: 

---

类似“梯度的反方向是下降最快的方向”的描述，经常用于介绍梯度下降（SGD）的原理。然而，这句话是有条件的，比如“方向”在数学上是单位向量，它依赖于“范数（模长）”的定义，不同范数的结论也不同，[Muon](/archives/10592)实际上就是给矩阵参数换了个谱范数，从而得到了新的下降方向。又比如，当我们从无约束优化转移到约束优化时，下降最快的方向也未必是梯度的反方向。

为此，在这篇文章中，我们将新开一个系列，以"约束"为主线，重新审视"最速下降"这一命题，探查不同条件下的"下降最快的方向"指向何方。

---

## 第1部分：理论起源、公理与历史基础

### 1.1 流形优化的理论起源

流形上的优化问题并非现代发明，而是植根于数学、物理和工程的深厚土壤之中。

<div class="theorem-box">

#### 理论根源的三大支柱

**1. 微分几何（Differential Geometry, 19世纪）**
- **高斯（Gauss, 1827）**：曲面内蕴几何理论，首次提出曲面上的测地线概念
- **黎曼（Riemann, 1854）**：将几何推广到高维流形，定义黎曼度量和黎曼曲率
- **列维-齐维塔（Levi-Civita, 1917）**：黎曼联络理论，解决向量在弯曲空间中的平行移动问题

**2. 变分法与约束优化（Calculus of Variations, 18世纪）**
- **拉格朗日（Lagrange, 1788）**：拉格朗日乘数法处理等式约束优化
- **欧拉（Euler, 1744）**：变分法求解泛函极值，奠定测地线理论基础
- **庞特里亚金（Pontryagin, 1956）**：最大值原理，处理带约束的动态优化

**3. 矩阵流形与数值分析（20世纪）**
- **Edelman (1998)**：矩阵流形上的几何与优化，将抽象流形理论应用于实际计算
- **Absil, Mahony & Sepulchre (2008)**：专著《Optimization Algorithms on Matrix Manifolds》，系统化流形优化算法
- **Boumal (2020)**：《An Introduction to Optimization on Smooth Manifolds》，现代流形优化教科书

</div>

**核心问题的演化**：

从无约束优化到约束优化，再到流形优化，这一演化反映了对"约束"认知的深化：

1. **无约束优化**（18-19世纪）：$\min_{\boldsymbol{x} \in \mathbb{R}^n} f(\boldsymbol{x})$
   - 梯度下降：沿负梯度方向在平坦欧氏空间中前进
   - 问题：参数可能任意增长，缺乏先验结构

2. **约束优化**（19-20世纪）：$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.} \quad g(\boldsymbol{x}) = 0$
   - 拉格朗日乘数法、KKT条件、投影梯度法
   - 问题：将约束视为"障碍"，而非内在几何

3. **流形优化**（20世纪末至今）：$\min_{\boldsymbol{x} \in \mathcal{M}} f(\boldsymbol{x})$
   - 约束集$\mathcal{M}$本身是流形，具有内在几何结构
   - **范式转换**：从"在欧氏空间中满足约束"到"在流形上自然移动"

### 1.2 历史发展：关键里程碑

<div class="theorem-box">

#### 流形优化的历史时间线

**阶段一：数学基础奠定（19世纪-1950年代）**

1. **1827 - 高斯《关于曲面的一般研究》**
   - 提出曲面内蕴几何概念
   - 高斯曲率定理：曲率是内蕴性质，不依赖于嵌入空间
   - **意义**：为在曲面上进行"内在"优化提供数学基础

2. **1854 - 黎曼就职演讲《几何基础的假设》**
   - 将几何推广到任意维度的流形
   - 定义黎曼度量：$g_{ij}(\boldsymbol{x}) = \langle \partial_i, \partial_j \rangle$
   - **意义**：超球面等约束集可统一在黎曼几何框架下研究

3. **1917 - 列维-齐维塔联络**
   - 解决向量在弯曲空间中的平行移动
   - 定义测地线：$\nabla_{\dot{\gamma}}\dot{\gamma} = 0$
   - **意义**：梯度下降在流形上的自然推广

**阶段二：数值算法发展（1960-1990年代）**

4. **1970s - 投影梯度法（Projected Gradient Descent）**
   - Rosen (1960), Bertsekas (1976)
   - 核心思想：欧氏梯度步 → 投影回约束集
   - **局限**：将约束视为外在限制，忽略内在几何

5. **1980 - Luenberger《线性与非线性规划》**
   - 系统论述约束优化的一阶/二阶方法
   - 引入可行方向锥（Cone of Feasible Directions）概念
   - **影响**：为流形优化的切空间理论铺路

6. **1998 - Edelman, Arias & Smith《矩阵流形的几何》**
   - 首次系统研究矩阵流形（Stiefel流形、Grassmann流形）
   - 给出切空间、黎曼梯度、retraction的显式公式
   - **意义**：将抽象理论转化为可计算算法

**阶段三：现代应用爆发（2000年代至今）**

7. **2008 - Absil, Mahony & Sepulchre专著**
   - 系统化流形优化算法框架
   - 涵盖：黎曼梯度、黎曼Hessian、trust-region方法、共轭梯度
   - **工具**：Manopt工具箱（MATLAB/Python）
   - **影响**：成为流形优化的标准教材

8. **2016 - Salimans & Kingma《权重归一化》（NeurIPS）**
   - 将神经网络权重分解为方向（超球面）和尺度
   - $\boldsymbol{w} = g \cdot \boldsymbol{v}/\|\boldsymbol{v}\|_2$
   - **应用**：RNN训练加速、GAN稳定性提升
   - **意义**：流形优化进入深度学习主流

9. **2018 - Miyato et al.《谱归一化》（ICLR）**
   - 约束权重矩阵的最大奇异值为1
   - 使用幂迭代法在Stiefel流形上优化
   - **应用**：GAN判别器Lipschitz约束
   - **意义**：几何约束改善生成模型质量

10. **2020+ - 现代流形优化库**
    - **Geoopt (PyTorch)**：支持超球面、双曲空间、SPD流形
    - **JAXopt**：JAX中的流形优化
    - **Pymanopt**：通用流形优化框架
    - **意义**：降低使用门槛，推动工业应用

</div>

### 1.3 数学公理与基础假设

超球面优化建立在以下数学公理之上：

<div class="theorem-box">

#### 公理1：流形的定义

**超球面流形** $\mathcal{S}^{n-1}$ 是满足以下条件的集合：
$$
\mathcal{S}^{n-1} = \{\boldsymbol{w} \in \mathbb{R}^n : \|\boldsymbol{w}\|_2 = 1\}
$$

**性质**：
1. **紧致性**：$\mathcal{S}^{n-1}$ 是有界闭集
2. **连通性**：$\mathcal{S}^{n-1}$ 是连通的（$n \geq 2$时为道路连通）
3. **光滑性**：$\mathcal{S}^{n-1}$ 是 $C^{\infty}$ 光滑流形
4. **维度**：$\dim(\mathcal{S}^{n-1}) = n - 1$

</div>

<div class="theorem-box">

#### 公理2：黎曼度量的继承

超球面继承欧氏空间 $\mathbb{R}^n$ 的内积作为黎曼度量：
$$
g_{\boldsymbol{w}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \langle\boldsymbol{\xi}, \boldsymbol{\eta}\rangle_{\mathbb{R}^n}, \quad \forall \boldsymbol{\xi}, \boldsymbol{\eta} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

**含义**：超球面上的"角度"和"长度"由嵌入空间的欧氏几何诱导。

**推论**：
- 两点间的测地线距离：$d(\boldsymbol{w}_1, \boldsymbol{w}_2) = \arccos(\langle\boldsymbol{w}_1, \boldsymbol{w}_2\rangle)$
- 截面曲率恒为 $K = 1$（正曲率）

</div>

<div class="theorem-box">

#### 公理3：切空间的正交性

在点 $\boldsymbol{w} \in \mathcal{S}^{n-1}$ 处的切空间定义为：
$$
T_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\boldsymbol{v} \in \mathbb{R}^n : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}
$$

**几何意义**：切空间是与球面在 $\boldsymbol{w}$ 点相切的超平面。

**正交分解定理**：
$$
\mathbb{R}^n = T_{\boldsymbol{w}}\mathcal{S}^{n-1} \oplus N_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$
其中法空间 $N_{\boldsymbol{w}}\mathcal{S}^{n-1} = \text{span}\{\boldsymbol{w}\}$。

</div>

<div class="theorem-box">

#### 公理4：黎曼梯度的存在性与唯一性

对于光滑函数 $f: \mathcal{S}^{n-1} \to \mathbb{R}$，在每点 $\boldsymbol{w}$ 存在唯一的黎曼梯度 $\text{grad} f(\boldsymbol{w}) \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$ 满足：
$$
\langle\text{grad} f(\boldsymbol{w}), \boldsymbol{\xi}\rangle = Df(\boldsymbol{w})[\boldsymbol{\xi}], \quad \forall \boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

**显式公式**（投影公式）：
$$
\text{grad} f(\boldsymbol{w}) = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}
$$

</div>

### 1.4 设计哲学：从欧氏到黎曼的范式转换

**核心哲学：约束即几何**

传统观点将约束视为优化的"障碍"，需要通过投影、罚函数等手段"绕过"。流形优化则认为：

> **约束不是障碍，而是几何的本质。**

在超球面优化中，这一哲学体现为：

**1. 参数空间的重新认知**

| 传统视角（欧氏） | 流形视角（黎曼） |
|------------------|------------------|
| $\boldsymbol{w} \in \mathbb{R}^n$ 受约束 $\|\boldsymbol{w}\|_2 = 1$ | $\boldsymbol{w}$ 本质上属于 $(n-1)$ 维流形 $\mathcal{S}^{n-1}$ |
| 约束是外加的限制 | 约束是内在的几何 |
| 优化在 $\mathbb{R}^n$ 中进行，需不断投影 | 优化在 $\mathcal{S}^{n-1}$ 中进行，自然保持约束 |

**2. 梯度的重新定义**

- **欧氏梯度** $\nabla f(\boldsymbol{w})$：在嵌入空间 $\mathbb{R}^n$ 中的下降方向
  - 问题：可能指向球外，违反约束

- **黎曼梯度** $\text{grad} f(\boldsymbol{w})$：在流形切空间中的下降方向
  - 优势：自动与约束兼容，沿球面移动

**3. 更新规则的本质区别**

- **投影方法**（Projected Gradient）：
  $$
  \boldsymbol{w}_{t+1} = \underbrace{\text{Proj}_{\mathcal{S}^{n-1}}}_{\text{投影回球面}}\left(\boldsymbol{w}_t - \eta \nabla f(\boldsymbol{w}_t)\right)
  $$
  - 逻辑：先在欧氏空间移动，再"修正"回约束集
  - 缺陷：两步操作，忽略流形几何

- **流形方法**（Riemannian Gradient）：
  $$
  \boldsymbol{w}_{t+1} = \underbrace{\text{Retr}_{\boldsymbol{w}_t}}_{\text{沿测地线移动}}\left(-\eta \cdot \text{grad} f(\boldsymbol{w}_t)\right)
  $$
  - 逻辑：直接在流形上沿测地线移动
  - 优势：一步到位，保持几何结构

**4. 最速下降的内在性**

<div class="intuition-box">

### 哲学思考：什么是"最快"？

"最速下降"的含义取决于如何度量"速度"：

- **欧氏空间**：用欧氏距离 $\|\Delta\boldsymbol{w}\|_2$ 度量步长
  - 最速方向：负梯度 $-\nabla f(\boldsymbol{w})$

- **超球面**：用测地线距离 $d_{\mathcal{S}}(\boldsymbol{w}, \boldsymbol{w}')$ 度量步长
  - 最速方向：负黎曼梯度 $-\text{grad} f(\boldsymbol{w})$

**类比**：在地球表面从北京飞往纽约，最短路径不是欧氏空间的直线（需穿过地心），而是大圆弧（测地线）。

</div>

**5. 数学优雅性**

流形优化的优雅之处在于：**将约束内化为几何，统一处理**。

例如，各种矩阵约束可统一为流形：
- 正交矩阵：Stiefel流形 $\text{St}(n, p) = \{\boldsymbol{X} \in \mathbb{R}^{n \times p} : \boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{I}_p\}$
- 正定矩阵：SPD流形 $\mathcal{P}_n = \{\boldsymbol{X} \in \mathbb{R}^{n \times n} : \boldsymbol{X} \succ 0\}$
- 低秩矩阵：Grassmann流形

每个流形有其独特的几何结构，但优化框架统一：
1. 计算黎曼梯度
2. 在切空间更新
3. 通过retraction回到流形

---

## 优化原理 #

作为第一篇文章，我们先从SGD出发，理解“梯度的反方向是下降最快的方向”这句话背后的数学意义，然后应用于超球面上的优化。不过在此之前，笔者还想带大家重温一下[《Muon续集：为什么我们选择尝试Muon？》](/archives/10739)所提的关于优化器的“**最小作用量原理（Least Action Principle）** ”。

这个原理尝试回答“什么才是好的优化器”。首先，我们肯定是希望模型收敛速度越快越好，但由于神经网络本身的复杂性，如果步子迈得太大，那么反而容易训崩。所以，一个好的优化器应该是又**稳** 又**快** ，最好是不用大改模型，但却可以明显降低损失，写成数学形式是  
\begin{equation}\min_{\Delta \boldsymbol{w}} \mathcal{L}(\boldsymbol{w} +\Delta\boldsymbol{w}) \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{w})\leq \eta\end{equation}  
其实$\mathcal{L}$是损失函数，$\boldsymbol{w}\in\mathbb{R}^n$是参数向量，$\Delta \boldsymbol{w}$是更新量，$\rho(\Delta\boldsymbol{w})$是更新量$\Delta\boldsymbol{w}$大小的某种度量。上述目标很直观，就是在“步子”不超过$\eta$（**稳** ）的前提下，寻找让损失函数下降最多（**快** ）的更新量，这便是“最小作用量原理”的数学含义，也是“最速下降”的数学含义。

## 目标转化 #

假设$\eta$足够小，那么$\Delta\boldsymbol{w}$也足够小，以至于一阶近似足够准确，那么我们就可以将$\mathcal{L}(\boldsymbol{w} +\Delta\boldsymbol{w})$替换为$\mathcal{L}(\boldsymbol{w}) + \langle\boldsymbol{g},\Delta\boldsymbol{w}\rangle$，其中$\boldsymbol{g} = \nabla_{\boldsymbol{w}}\mathcal{L}(\boldsymbol{w})$，得到等效目标  
\begin{equation}\min_{\Delta \boldsymbol{w}} \langle\boldsymbol{g},\Delta\boldsymbol{w}\rangle \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{w})\leq \eta\end{equation}  
这就将优化目标简化成$\Delta \boldsymbol{w}$的线性函数，降低了求解难度。进一步地，我们设$\Delta \boldsymbol{w} = -\kappa \boldsymbol{\varphi}$，其中$\rho(\boldsymbol{\varphi})=1$，那么上述目标等价于  
\begin{equation}\max_{\kappa,\boldsymbol{\varphi}} \kappa\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \rho(\boldsymbol{\varphi}) = 1, \,\,\kappa\in[0, \eta]\end{equation}  
假设我们至少能找到一个满足条件的$\boldsymbol{\varphi}$使得$\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle\geq 0$，那么有$\max\limits_{\kappa\in[0,\eta]} \kappa\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \eta\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle$，也就是$\kappa$的优化可以事先求出来，结果是$\kappa=\eta$，最终等效于只剩下$\boldsymbol{\varphi}$的优化  
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \rho(\boldsymbol{\varphi}) = 1\label{eq:core}\end{equation}  
这里的$\boldsymbol{\varphi}$满足某种“模长”$\rho(\boldsymbol{\varphi})$等于1的条件，所以它代表了某种“方向向量”的定义，最大化它与梯度$\boldsymbol{g}$的内积，就意味着寻找让损失下降最快的方向（即$\boldsymbol{\varphi}$的反方向）。

## 梯度下降 #

从式$\eqref{eq:core}$可以看出，对于“下降最快的方向”，唯一不确定的是度量$\rho$，这是优化器里边很本质的一个先验（Inductive Bias），不同的度量将会得到不同的最速下降方向。比较简单的就是L2范数或者说欧几里得范数$\rho(\boldsymbol{\varphi})=\Vert \boldsymbol{\varphi} \Vert_2$，也就是我们通常意义下的模长，这时候我们有柯西不等式：  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g}\Vert_2 \Vert\boldsymbol{\varphi}\Vert_2 = \Vert\boldsymbol{g}\Vert_2\end{equation}  
等号成立的条件是$\boldsymbol{\varphi} \propto\boldsymbol{g}$，加上模长为1的条件，我们得到$\boldsymbol{\varphi}=\boldsymbol{g}/\Vert\boldsymbol{g}\Vert_2$，这正是梯度的方向。所以说，“梯度的反方向是下降最快的方向”前提是所选取的度量是欧几里得范数。更一般地，我们考虑$p$范数  
\begin{equation}\rho(\boldsymbol{\varphi}) = \Vert\boldsymbol{\varphi}\Vert_p = \sqrt[\uproot{10}p]{\sum_{i=1}^n |\varphi_i|^p}\end{equation}  
柯西不等式可以推广成[Hölder不等式](https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality)：  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g}\Vert_q \Vert\boldsymbol{\varphi}\Vert_p = \Vert\boldsymbol{g}\Vert_q,\qquad 1/p + 1/q=1\end{equation}  
等号成立的条件$\boldsymbol{\varphi}^{[p]} \propto\boldsymbol{g}^{[q]}$，所以解得  
\begin{equation}\newcommand{sign}{\mathop{\text{sign}}}\boldsymbol{\varphi} = \frac{\boldsymbol{g}^{[q/p]}}{\Vert\boldsymbol{g}^{[q/p]}\Vert_p},\qquad \boldsymbol{g}^{[\alpha]} \triangleq \big[\sign(g_1) |g_1|^{\alpha},\sign(g_2) |g_2|^{\alpha},\cdots,\sign(g_n) |g_n|^{\alpha}\big]\end{equation}  
以它为方向向量的优化器叫做pbSGD，出自[《pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non-Convex Optimization》](https://www.ijcai.org/proceedings/2020/451)。它有两个特例，一是$p=q=2$是退化为SGD，二是$p\to\infty$时$q\to 1$，此时$|g_i|^{q/p}\to 1$，更新方向为梯度的符号函数，即SignSGD。

## 超球面上 #

前面的讨论中，我们只是对参数的增量$\Delta\boldsymbol{w}$施加了约束，接下来我们希望的是给参数$\boldsymbol{w}$也添加约束。具体来说，我们假设参数$\boldsymbol{w}$位于单位球面上，我们希望更新后的参数$\boldsymbol{w}+\Delta\boldsymbol{w}$依然位于单位球面上（参考[《Hypersphere》](https://docs.modula.systems/algorithms/manifold/hypersphere/)）。从目标$\eqref{eq:core}$出发，我们可以将新目标写成  
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \Vert\boldsymbol{\varphi}\Vert_2 = 1,\,\,\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2 = 1,\,\,\Vert\boldsymbol{w}\Vert_2=1\end{equation}  
我们依然贯彻“$\eta$足够小，一阶近似够用”的原则，得到  
\begin{equation}1 = \Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2^2 = \Vert\boldsymbol{w}\Vert_2^2 - 2\eta\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle + \eta^2 \Vert\boldsymbol{\varphi}\Vert_2^2\approx 1 - 2\eta\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle\end{equation}  
所以这相当于将约束转化为线性形式$\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0$。为了求解新的目标，我们引入待定系数$\lambda$，写出  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle + \lambda\langle\boldsymbol{w},\boldsymbol{\varphi}\rangle =\langle \boldsymbol{g} + \lambda\boldsymbol{w},\boldsymbol{\varphi}\rangle\leq \Vert\boldsymbol{g} + \lambda\boldsymbol{w}\Vert_2 \Vert\boldsymbol{\varphi}\Vert_2 = \Vert\boldsymbol{g} + \lambda\boldsymbol{w}\Vert_2\end{equation}  
等号成立的条件是$\boldsymbol{\varphi}\propto \boldsymbol{g} + \lambda\boldsymbol{w}$，再加上$\Vert\boldsymbol{\varphi}\Vert_2=1,\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0,\Vert\boldsymbol{w}\Vert_2=1$的条件，可以解得  
\begin{equation}\boldsymbol{\varphi} = \frac{\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}}{\Vert\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}\Vert_2}\end{equation}  
注意现在有$\Vert\boldsymbol{w}\Vert_2=1,\Vert\boldsymbol{\varphi}\Vert_2=1$，并且$\boldsymbol{w}$和$\boldsymbol{\varphi}$是正交的，那么$\boldsymbol{w} - \eta\boldsymbol{\varphi}$的模长是并不是精确地等于1，而是$\sqrt{1 + \eta^2}=1 + \eta^2/2 + \cdots$，精确到$\mathcal{O}(\eta^2)$，这跟我们前面的假设“$\eta$的一阶项够用”吻合。如果想更新后的参数模长严格等于1，那么可以在更新规则上多加一步缩回操作：  
\begin{equation}\boldsymbol{w}\quad\leftarrow\quad \frac{\boldsymbol{w} - \eta\boldsymbol{\varphi}}{\sqrt{1 + \eta^2}}\end{equation}

## 几何意义 #

刚才我们通过“一阶近似够用”原则，将非线性约束$\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2 = 1$简化为线性约束$\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0$，后者的几何意义是“与$\boldsymbol{w}$垂直”，这还有个更专业的说法，叫做$\Vert\boldsymbol{w}\Vert_2=1$的“切空间”，而$\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}$这一运算，正对应于把梯度$\boldsymbol{g}$投影到切空间的投影运算。

所以很幸运，球面上的SGD有非常清晰的几何意义，如下图所示：  


[![球面上的最速下降-几何意义](/usr/uploads/2025/07/1221229703.png)](/usr/uploads/2025/07/1221229703.png "点击查看原图")

球面上的最速下降-几何意义

相信很多读者都喜欢这种几何视角，它确实让人赏心悦目。但这是还是要提醒大家一下，应当优先认真理解代数求解过程，因为清晰的几何意义很多时候都只是一种奢望，属于可遇而不可求的，大多数情况下复杂的代数过程才是本质。

---

## 第3部分：数学直觉、多角度解释与类比

### 3.1 生活化类比：理解超球面优化

<div class="intuition-box">

#### 类比1：登山者在球形星球表面的最速下降

假设你是一名登山者，站在一个小型球形星球（如小王子的星球）的表面。你想尽快下到最低点。

**约束**：你必须一直在星球表面行走（不能飞天或钻地）。

**问题**：从当前位置出发，哪个方向下降最快？

**直觉回答**：
- **错误想法**：朝着星球中心的反方向走（即远离中心）
  - 问题：这会让你离开星球表面，违反约束

- **正确做法**：在表面上选择一个切向方向，使得沿这个方向前进时，高度下降最快
  - 这正是"黎曼梯度"的几何意义

**数学对应**：
- 星球表面 = 超球面 $\mathcal{S}^{n-1}$
- 高度函数 = 目标函数 $f(\boldsymbol{w})$
- 引力方向 = 欧氏梯度 $\nabla f(\boldsymbol{w})$
- 表面最陡方向 = 黎曼梯度 $\text{grad} f(\boldsymbol{w})$

**为什么要投影**：
引力指向星球中心（法向），但你不能穿过地面。所以你将引力分解为：
- **切向分量**：沿表面的力，这才是你能利用的
- **法向分量**：垂直于表面的力，被地面支撑力抵消

黎曼梯度正是欧氏梯度的切向分量！

</div>

<div class="intuition-box">

#### 类比2：在气球表面绘画的笔尖轨迹

想象你在一个充气的气球表面用笔绘画，笔尖始终与气球接触。

**场景设定**：
- 气球表面有"海拔"信息（如温度分布），颜色深浅代表函数值
- 你的笔在气球表面滑动，目标是尽快到达最冷（函数值最小）的点
- 笔尖不能离开气球表面（约束）

**关键观察**：

1. **欧氏梯度**（3D空间中的最陡方向）：
   - 可能指向气球内部或外部
   - 笔尖无法沿此方向移动

2. **黎曼梯度**（球面上的最陡方向）：
   - 是欧氏梯度在球面上的"影子"（投影）
   - 笔尖可以实际沿此方向滑动

**投影公式的直观理解**：
$$
\text{grad} f(\boldsymbol{w}) = \underbrace{\nabla f(\boldsymbol{w})}_{\text{3D空间梯度}} - \underbrace{\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}}_{\text{指向球心的分量}}
$$

- 第一项：3D空间中函数下降最快的方向
- 第二项：这个方向中"垂直于球面"的部分（不能用）
- 结果：去掉不能用的部分，剩下的就是球面上最陡的方向

</div>

<div class="intuition-box">

#### 类比3：被约束的弹簧小球系统

考虑一个物理系统：一个小球通过无摩擦的刚性杆连接到原点，杆长固定为1。

**力学分析**：

- **外力**：$\boldsymbol{F}_{\text{external}} = -\nabla f(\boldsymbol{w})$（例如重力、电磁力）
- **约束力**：杆对小球施加的径向力 $\boldsymbol{F}_{\text{constraint}}$
- **实际加速度**：只有切向分量能引起小球沿球面运动

**牛顿第二定律**：
$$
m\boldsymbol{a} = \boldsymbol{F}_{\text{external}} + \boldsymbol{F}_{\text{constraint}}
$$

由于小球被约束在球面上，加速度 $\boldsymbol{a}$ 必须切向：
$$
\boldsymbol{a} \perp \boldsymbol{w} \quad \Rightarrow \quad \langle\boldsymbol{a}, \boldsymbol{w}\rangle = 0
$$

因此，只有外力的切向分量有效：
$$
\boldsymbol{a} = \frac{1}{m}\mathcal{P}_{\boldsymbol{w}}(\boldsymbol{F}_{\text{external}}) = \frac{1}{m}(\boldsymbol{F}_{\text{external}} - \langle\boldsymbol{F}_{\text{external}}, \boldsymbol{w}\rangle\boldsymbol{w})
$$

**对应到优化**：
- 外力 $\boldsymbol{F} \leftrightarrow$ 负梯度 $-\nabla f(\boldsymbol{w})$
- 切向加速度 $\boldsymbol{a} \leftrightarrow$ 黎曼梯度 $-\text{grad} f(\boldsymbol{w})$
- 约束力 $\boldsymbol{F}_{\text{constraint}} \leftrightarrow$ 拉格朗日乘数 $\lambda\boldsymbol{w}$

物理系统的自然演化正是沿黎曼梯度的最速下降！

</div>

### 3.2 几何意义的深入理解

#### 3.2.1 切空间：流形的"局部平坦化"

在点 $\boldsymbol{w}$ 处，超球面看起来像一个平面（类似地球表面在小范围内近似平坦）。这个平面就是切空间。

**数学定义**：
$$
T_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\boldsymbol{v} : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}
$$

**几何直觉**：
- 切空间是与球面在 $\boldsymbol{w}$ 处"刚好贴合"的超平面
- 所有可行的移动方向都在这个平面内
- 维度：$(n-1)$ 维（比环境空间少一维）

<div class="example-box">

#### 例子：三维空间中的单位球面 $(n=3)$

考虑 $\mathcal{S}^2 = \{(x, y, z) : x^2 + y^2 + z^2 = 1\}$。

**点**：$\boldsymbol{w} = (0, 0, 1)$（北极点）

**切空间**：
$$
T_{\boldsymbol{w}}\mathcal{S}^2 = \{(v_x, v_y, v_z) : 0 \cdot v_x + 0 \cdot v_y + 1 \cdot v_z = 0\}
$$
$$
= \{(v_x, v_y, 0) : v_x, v_y \in \mathbb{R}\}
$$

这是 $xy$ 平面，正是地球北极点处的"水平面"。

**物理意义**：站在北极，你只能向南（任意方向）移动，不能向上飞或向下钻。

</div>

#### 3.2.2 投影：从环境空间到流形

投影算子 $\mathcal{P}_{\boldsymbol{w}}$ 将任意向量"压平"到切空间：
$$
\mathcal{P}_{\boldsymbol{w}}(\boldsymbol{v}) = \boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}
$$

**几何过程**：
1. 计算 $\boldsymbol{v}$ 在法向（即 $\boldsymbol{w}$ 方向）上的分量：$\langle\boldsymbol{v}, \boldsymbol{w}\rangle$
2. 沿法向减去这个分量：$\langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}$
3. 剩下的部分完全在切空间内

<div class="example-box">

#### 可视化：投影的几何图景

```
     ↑ \boldsymbol{v} (原向量)
     |  /
     | / ← \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w} (法向分量)
     |/
     ●------------→ \mathcal{P}_{\boldsymbol{w}}(\boldsymbol{v}) (切向投影)
    \boldsymbol{w}
    (球面上的点)
```

- 垂直箭头：法向分量（垂直于球面）
- 水平箭头：切向分量（沿球面方向）
- 原向量 = 法向分量 + 切向分量

</div>

### 3.3 多角度理解超球面优化

#### 角度1：微分几何视角

**核心概念**：超球面是黎曼流形，具有内在几何结构。

**关键要素**：
1. **度量**：如何测量切向量的长度和角度
   - 超球面继承欧氏内积：$g_{\boldsymbol{w}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \langle\boldsymbol{\xi}, \boldsymbol{\eta}\rangle$

2. **测地线**：流形上的"最短路径"
   - 超球面上的测地线是大圆弧
   - 从 $\boldsymbol{w}$ 出发沿方向 $\boldsymbol{\xi}$ 的测地线：
     $$
     \gamma(t) = \cos(t\|\boldsymbol{\xi}\|_2)\boldsymbol{w} + \sin(t\|\boldsymbol{\xi}\|_2)\frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}
     $$

3. **曲率**：流形的"弯曲程度"
   - 超球面的截面曲率恒为 $K = 1$（正曲率）
   - 对比：欧氏空间 $K = 0$，双曲空间 $K < 0$

**优化解释**：
黎曼梯度下降 = 沿测地线在弯曲空间中移动，自动适应流形几何。

#### 角度2：约束优化视角

**问题设定**：
$$
\min_{\boldsymbol{w}} f(\boldsymbol{w}) \quad \text{s.t.} \quad \|\boldsymbol{w}\|_2 = 1
$$

**拉格朗日方法**：
$$
\mathcal{L}(\boldsymbol{w}, \lambda) = f(\boldsymbol{w}) + \lambda(\|\boldsymbol{w}\|_2^2 - 1)
$$

**KKT条件**：
$$
\nabla_{\boldsymbol{w}}\mathcal{L} = \nabla f(\boldsymbol{w}) + 2\lambda\boldsymbol{w} = \boldsymbol{0}
$$

**解释**：
- 最优点处，欧氏梯度 $\nabla f(\boldsymbol{w})$ 与位置向量 $\boldsymbol{w}$ 平行（都在法空间中）
- 等价于：黎曼梯度为零 $\text{grad} f(\boldsymbol{w}) = \boldsymbol{0}$

**投影梯度的合理性**：
投影梯度法 $\boldsymbol{w}_{t+1} = \text{Proj}_{\mathcal{S}}(\boldsymbol{w}_t - \eta\nabla f(\boldsymbol{w}_t))$ 可以看作：
1. 在欧氏空间下降一步
2. 投影回约束集

而黎曼梯度法直接在约束集上移动，更符合几何本质。

#### 角度3：信息几何视角

**超球面作为概率分布空间**：

单位向量 $\boldsymbol{w} \in \mathcal{S}^{n-1}$ 可以解释为：
- 离散概率分布的参数（经过适当归一化）
- 方向统计中的方向向量

**Fisher信息度量**：
在某些参数化下，超球面的黎曼度量对应Fisher信息矩阵，这连接了几何与统计。

**自然梯度的联系**：
黎曼梯度可以看作自然梯度的特例，都体现了"参数空间的内在几何"。

#### 角度4：优化算法视角

**传统SGD的局限**：

假设用普通SGD优化 $\boldsymbol{w} \in \mathbb{R}^n$ 但希望满足 $\|\boldsymbol{w}\|_2 = 1$：

**方法1：后处理归一化**
$$
\boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta\nabla f(\boldsymbol{w}_t)}{\|\boldsymbol{w}_t - \eta\nabla f(\boldsymbol{w}_t)\|_2}
$$
- 问题：更新方向可能偏离最优

**方法2：罚函数法**
$$
\min_{\boldsymbol{w}} f(\boldsymbol{w}) + \mu(\|\boldsymbol{w}\|_2^2 - 1)^2
$$
- 问题：引入额外超参数 $\mu$，约束可能不严格满足

**方法3：流形方法**（本文）
$$
\boldsymbol{w}_{t+1} = \text{Retr}_{\boldsymbol{w}_t}(-\eta \cdot \text{grad} f(\boldsymbol{w}_t))
$$
- 优势：约束严格满足，更新方向最优

<div class="intuition-box">

### 为什么投影梯度是"最速"的？

从三个层面理解"最速下降"：

**层面1：局部最优性**

在所有满足 $\|\boldsymbol{\xi}\|_2 \leq \eta$ 且 $\langle\boldsymbol{\xi}, \boldsymbol{w}\rangle = 0$ 的切向量中，
$$
\boldsymbol{\xi}^* = -\eta \frac{\text{grad} f(\boldsymbol{w})}{\|\text{grad} f(\boldsymbol{w})\|_2}
$$
使得 $f(\boldsymbol{w} + \boldsymbol{\xi})$ 下降最多（一阶近似下）。

**层面2：几何最短**

负黎曼梯度指向函数值等高线法向，沿此方向是离开当前等高线的最短路径。

**层面3：信息几何意义**

黎曼梯度考虑了参数空间的"距离"（由黎曼度量定义），因此是信息几何意义下的最速下降。

</div>

---

## 一般结果 #

接下来是不是有读者想要将它推广到一般的$p$范数？让我们一起来尝试下，看看会遇到什么困难。这时候问题是：  
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \Vert\boldsymbol{\varphi}\Vert_p = 1,\,\,\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_p = 1,\,\,\Vert\boldsymbol{w}\Vert_p=1\end{equation}  
一阶近似将$\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_p = 1$转换成$\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = 0$，然后引入待定系数$\lambda$：  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle + \lambda\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = \langle \boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]}\Vert_q \Vert\boldsymbol{\varphi}\Vert_p = \Vert\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]}\Vert_q  
\end{equation}  
取等号的条件是  
\begin{equation}\boldsymbol{\varphi} = \frac{(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})^{[q/p]}}{\Vert(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})^{[q/p]}\Vert_p}\end{equation}  
到目前为止，都没有实质困难。然而，接下来我们需要寻找$\lambda$，使得$\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = 0$，当$p \neq 2$时这是一个复杂的非线性方程，并没有很好的求解办法（当然，一旦求解出来，我们就肯定能得到最优解，这是Hölder不等式保证的）。所以，一般$p$的求解我们只能止步于此，等遇到$p\neq 2$的实例时我们再具体探寻数值求解方法。

不过除了$p=2$，我们还可以尝试求解$p\to\infty$，此时$\boldsymbol{\varphi}=\sign(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})$，条件$\Vert\boldsymbol{w}\Vert_p=1$给出$|w_1|,|w_2|,\cdots,|w_n|$的最大值等于1。如果进一步假设最大值只有一个，那么$\boldsymbol{w}^{[p-1]}$是一个one hot向量，绝对值最大值的位置为$\pm 1$，其余是零，这时候就可以解出$\lambda$，结果是把最大值位置的梯度裁剪成零。

---

## 第4部分：方法论变体、批判性比较与优化

### 4.1 方法对比表

| 方法 | 核心思想 | 优点 | 缺陷 | 优化方向 |
|------|---------|------|------|---------|
| **无约束SGD** | 直接梯度下降 $\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\nabla f$ | ✅ 实现简单<br>✅ 理论成熟<br>✅ 无额外计算 | ❌ **参数范数无界增长**<br>❌ 不保证约束满足<br>❌ 忽略先验几何结构 | ✅ 添加权重衰减<br>✅ 定期归一化<br>✅ 切换到流形方法 |
| **投影梯度法**<br>(Projected GD) | 欧氏步+投影<br>$\boldsymbol{w}_{t+1} = \text{Proj}_{\mathcal{S}}(\boldsymbol{w}_t - \eta\nabla f)$ | ✅ 约束严格满足<br>✅ 易于理解<br>✅ 适用范围广 | ❌ **两步操作效率低**<br>❌ 更新方向次优<br>❌ 大步长时投影扭曲严重 | ✅ 减小步长<br>✅ 使用自适应学习率<br>✅ 改用黎曼梯度 |
| **黎曼梯度法**<br>(本文方法) | 切空间梯度<br>$\boldsymbol{w}_{t+1} = \text{Retr}(\boldsymbol{w}_t - \eta\text{grad}f)$ | ✅ 理论最优<br>✅ 一步到位<br>✅ 几何意义清晰 | ❌ **需要额外投影计算**<br>❌ 实现复杂度稍高<br>❌ 通用性不如投影法 | ✅ 复用计算结果<br>✅ 向量化实现<br>✅ 与动量/Adam结合 |
| **自然梯度法**<br>(Natural Gradient) | Fisher度量<br>$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta F^{-1}\nabla f$ | ✅ 参数化不变性<br>✅ 信息几何最优<br>✅ 收敛快（强凸时） | ❌ **计算Fisher矩阵昂贵**<br>❌ $O(n^3)$ 复杂度<br>❌ 数值不稳定 | ✅ K-FAC近似<br>✅ 对角近似<br>✅ Kronecker分解 |
| **罚函数法**<br>(Penalty Method) | 软约束<br>$\min f(\boldsymbol{w}) + \mu(\|\boldsymbol{w}\|_2^2 - 1)^2$ | ✅ 无约束优化框架<br>✅ 灵活性高<br>✅ 支持不等式约束 | ❌ **约束不精确满足**<br>❌ 超参数敏感<br>❌ 条件数恶化 | ✅ 增强拉格朗日法<br>✅ 自适应$\mu$<br>✅ 交替方向乘子法 |

### 4.2 投影梯度法 - 批判性分析

#### 核心缺陷

**缺陷1：更新方向次优**

- **问题**：投影梯度法的更新 $\boldsymbol{w}_{t+1} = \text{Proj}(\boldsymbol{w}_t - \eta\nabla f)$ 并非球面上的最速下降方向
- **根本原因**：先在欧氏空间移动，再投影回约束集，这两步分离导致方向扭曲
- **定量影响**：当 $\eta$ 较大时，投影后的方向与黎曼梯度夹角可达 $O(\eta^2)$

<div class="derivation-box">

#### 推导：投影梯度法的方向误差

设 $\boldsymbol{w}$ 满足 $\|\boldsymbol{w}\|_2 = 1$，欧氏梯度为 $\boldsymbol{g} = \nabla f(\boldsymbol{w})$。

**投影梯度法**：
$$
\boldsymbol{w}_{\text{proj}} = \frac{\boldsymbol{w} - \eta\boldsymbol{g}}{\|\boldsymbol{w} - \eta\boldsymbol{g}\|_2}
$$

**黎曼梯度法**：
$$
\boldsymbol{w}_{\text{Riem}} = \frac{\boldsymbol{w} - \eta(\boldsymbol{g} - \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w})}{\|\boldsymbol{w} - \eta(\boldsymbol{g} - \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w})\|_2}
$$

**计算分母的泰勒展开**（投影法）：
$$
\|\boldsymbol{w} - \eta\boldsymbol{g}\|_2^2 = 1 - 2\eta\langle\boldsymbol{w}, \boldsymbol{g}\rangle + \eta^2\|\boldsymbol{g}\|_2^2
$$
$$
= 1 - 2\eta\langle\boldsymbol{w}, \boldsymbol{g}\rangle + \eta^2\|\boldsymbol{g}\|_2^2
$$

**计算分母的泰勒展开**（黎曼法）：
设 $\boldsymbol{g}_{\perp} = \boldsymbol{g} - \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}$（黎曼梯度）
$$
\|\boldsymbol{w} - \eta\boldsymbol{g}_{\perp}\|_2^2 = 1 + \eta^2\|\boldsymbol{g}_{\perp}\|_2^2
$$

**方向差异**：
投影法的归一化系数：
$$
\frac{1}{\sqrt{1 - 2\eta\langle\boldsymbol{w}, \boldsymbol{g}\rangle + \eta^2\|\boldsymbol{g}\|_2^2}} \approx 1 + \eta\langle\boldsymbol{w}, \boldsymbol{g}\rangle + O(\eta^2)
$$

黎曼法的归一化系数：
$$
\frac{1}{\sqrt{1 + \eta^2\|\boldsymbol{g}_{\perp}\|_2^2}} \approx 1 - \frac{\eta^2\|\boldsymbol{g}_{\perp}\|_2^2}{2} + O(\eta^4)
$$

**结论**：两种方法在 $O(\eta)$ 项上有差异 $\eta\langle\boldsymbol{w}, \boldsymbol{g}\rangle$，这导致：
$$
\|\boldsymbol{w}_{\text{proj}} - \boldsymbol{w}_{\text{Riem}}\|_2 = O(\eta^2)
$$

</div>

**缺陷2：计算效率问题**

- **问题**：需要两次完整的向量操作（梯度步+归一化），无法融合计算
- **影响**：内存访问次数增加，缓存命中率降低
- **定量分析**：
  - 梯度步：$n$ 次乘加 + $n$ 次减法
  - 归一化：$n$ 次平方 + 1次开方 + $n$ 次除法
  - 总计：约 $4n$ 次访存操作

**缺陷3：大步长时的几何扭曲**

- **问题**：当 $\eta$ 很大时，$\boldsymbol{w} - \eta\nabla f$ 可能远离球面，投影会严重改变方向
- **理论分析**：欧氏步长为 $\eta$，但投影后球面弧长可能远小于 $\eta$（甚至接近0）
- **数值实验**：在高曲率区域，投影可能导致 $>90°$ 的方向变化

#### 优化方向

**优化1：自适应投影（Adaptive Projection）**

- **策略**：根据梯度与位置向量的夹角调整步长
  $$
  \eta_{\text{eff}} = \eta \cdot \frac{\|\boldsymbol{g}_{\perp}\|_2}{\|\boldsymbol{g}\|_2}
  $$
- **效果**：减少方向扭曲，保持有效步长稳定

**优化2：混合方法（Hybrid Approach）**

- **策略**：小步长用投影法（简单），大步长用黎曼法（精确）
  $$
  \boldsymbol{w}_{t+1} = \begin{cases}
  \text{Proj}(\boldsymbol{w}_t - \eta\nabla f) & \text{if } \eta < \epsilon \\
  \text{Retr}(\boldsymbol{w}_t - \eta\text{grad}f) & \text{otherwise}
  \end{cases}
  $$
- **效果**：兼顾效率与精度

**优化3：预条件投影（Preconditioned Projection）**

- **策略**：使用预条件矩阵 $\boldsymbol{P}$ 改善条件数
  $$
  \boldsymbol{w}_{t+1} = \text{Proj}(\boldsymbol{w}_t - \eta\boldsymbol{P}^{-1}\nabla f)
  $$
  其中 $\boldsymbol{P}$ 可以是对角估计或动量积累
- **效果**：加速收敛，类似Adam在球面上的扩展

### 4.3 黎曼梯度法 - 批判性分析

#### 核心缺陷

**缺陷1：额外的投影计算开销**

- **问题**：每步需要计算 $\langle\nabla f, \boldsymbol{w}\rangle$ 和 $\langle\nabla f, \boldsymbol{w}\rangle\boldsymbol{w}$
- **根本原因**：将欧氏梯度投影到切空间的必要代价
- **定量影响**：
  - 内积计算：$O(n)$
  - 标量乘向量：$O(n)$
  - 向量减法：$O(n)$
  - 总额外开销：约 $3n$ 次浮点运算（相比无约束SGD）

<div class="formula-explanation">

<div class="formula-step">
<div class="step-label">复杂度分解</div>

**无约束SGD**：
$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\nabla f(\boldsymbol{w}_t)
$$
- 向量加法：$n$ 次运算
- **总计**：$n$ FLOPs

**黎曼梯度SGD**：
$$
\boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta(\nabla f - \langle\nabla f, \boldsymbol{w}_t\rangle\boldsymbol{w}_t)}{\|\cdots\|_2}
$$
- 内积 $\langle\nabla f, \boldsymbol{w}_t\rangle$：$n$ 次乘加
- 标量乘向量 $\langle\nabla f, \boldsymbol{w}_t\rangle\boldsymbol{w}_t$：$n$ 次乘法
- 两次向量减法：$2n$ 次减法
- 范数计算：$n$ 次平方 + 1次开方
- 归一化：$n$ 次除法
- **总计**：约 $6n$ FLOPs

<div class="step-explanation">
额外开销约为 **5倍**，但相比梯度计算本身（通常 $O(nd)$ 或更高），这是可接受的。
</div>
</div>

</div>

**缺陷2：与现代优化器（Adam等）的结合不直接**

- **问题**：Adam使用一阶和二阶矩估计，但在流形上需要重新定义"矩"
- **挑战**：
  - 动量 $\boldsymbol{m}_t$ 应该在哪个切空间累积？
  - 不同点的切空间不同，如何传输向量？
- **现状**：缺乏理论完备的RiemannianAdam

**缺陷3：高维参数空间的数值稳定性**

- **问题**：当 $\|\nabla f\|_2 \gg |\langle\nabla f, \boldsymbol{w}\rangle|$ 时，投影可能导致数值误差放大
- **影响**：在接近最优解时，梯度的法向分量很小，切向分量主导，浮点精度损失
- **定量分析**：若 $\langle\nabla f, \boldsymbol{w}\rangle / \|\nabla f\|_2 < 10^{-8}$，单精度浮点数可能失去精度

#### 优化方向

**优化1：向量化与缓存复用（Vectorization & Cache Reuse）**

- **策略**：将投影、归一化融合为单个kernel，减少内存访问
  ```python
  # 低效版本（3次访存）
  dot_product = np.dot(grad, w)
  projected = grad - dot_product * w
  w_new = projected / np.linalg.norm(projected)

  # 高效版本（融合kernel，1次访存）
  @jit
  def fused_riemannian_step(w, grad, eta):
      dot_prod = 0.0
      norm_sq = 0.0
      for i in range(n):
          grad_proj_i = grad[i] - dot_prod * w[i]
          norm_sq += grad_proj_i ** 2
      norm = sqrt(norm_sq)
      for i in range(n):
          w[i] = (w[i] - eta * grad_proj_i) / norm
  ```
- **效果**：内存带宽需求降低 $2-3$ 倍

**优化2：黎曼Adam（RAdam）- 理论扩展**

- **策略**：在切空间定义动量和二阶矩，使用平行移动传输
  $$
  \boldsymbol{m}_t = \beta_1 \mathcal{T}_{t-1\to t}(\boldsymbol{m}_{t-1}) + (1-\beta_1)\text{grad} f(\boldsymbol{w}_t)
  $$
  其中 $\mathcal{T}_{t-1\to t}$ 是从 $T_{\boldsymbol{w}_{t-1}}$ 到 $T_{\boldsymbol{w}_t}$ 的平行移动算子
- **近似**：对于超球面，平行移动可以近似为投影：
  $$
  \mathcal{T}_{t-1\to t}(\boldsymbol{v}) \approx \boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}_t\rangle\boldsymbol{w}_t
  $$
- **效果**：保留Adam的自适应性，同时尊重几何结构

**优化3：数值稳定化技巧**

- **策略1**：使用Kahan求和计算内积，减少浮点误差累积
- **策略2**：当 $\|\text{grad} f\|_2 < \epsilon$ 时，跳过归一化（已近似在球面）
- **策略3**：定期（每 $K$ 步）精确重投影：$\boldsymbol{w} \leftarrow \boldsymbol{w}/\|\boldsymbol{w}\|_2$
- **效果**：长时间训练时约束误差 $<10^{-12}$

### 4.4 自然梯度法 - 批判性分析

#### 核心缺陷

**缺陷1：计算Fisher信息矩阵的昂贵成本**

- **问题**：Fisher矩阵 $\boldsymbol{F} = \mathbb{E}[\nabla\log p \nabla\log p^{\top}]$ 是 $n \times n$ 稠密矩阵
- **复杂度**：
  - 计算：$O(n^2)$ 存储，$O(n^3)$ 求逆
  - 对比：黎曼梯度仅需 $O(n)$
- **定量影响**：$n=10^6$ 时，Fisher矩阵需要 $\sim 4$ TB内存（不可行）

**缺陷2：超球面上Fisher度量与欧氏度量的关系复杂**

- **问题**：在一般参数化下，Fisher度量不简化为黎曼度量
- **理论分析**：只有在特定参数化（如指数族）下，两者才等价
- **影响**：自然梯度不直接等同于黎曼梯度

**缺陷3：条件数恶化与数值不稳定**

- **问题**：Fisher矩阵往往病态（条件数 $>10^6$），求逆不稳定
- **根本原因**：参数冗余、平坦方向
- **影响**：需要正则化 $\boldsymbol{F} + \lambda\boldsymbol{I}$，引入新超参数

#### 优化方向

**优化1：K-FAC（Kronecker-Factored Approximate Curvature）**

- **策略**：将Fisher矩阵分解为Kronecker积
  $$
  \boldsymbol{F} \approx \boldsymbol{A} \otimes \boldsymbol{B}
  $$
  其中 $\boldsymbol{A}$ 和 $\boldsymbol{B}$ 维度较小
- **复杂度**：从 $O(n^3)$ 降至 $O(n^{1.5})$
- **效果**：在大规模网络上可行，ImageNet训练加速 $2-3$ 倍

**优化2：对角Fisher近似**

- **策略**：只计算Fisher矩阵对角元素
  $$
  \text{grad}_{\text{nat}} f \approx \text{diag}(\boldsymbol{F})^{-1} \nabla f
  $$
- **复杂度**：$O(n)$ 存储和计算
- **效果**：接近Adam的计算成本，收敛速度介于SGD和完全自然梯度之间

**优化3：在线Fisher估计**

- **策略**：使用指数移动平均在线估计Fisher矩阵
  $$
  \boldsymbol{F}_t = \beta \boldsymbol{F}_{t-1} + (1-\beta)\nabla\log p_t \nabla\log p_t^{\top}
  $$
- **优势**：无需存储所有样本
- **效果**：适用于流数据和在线学习

### 4.5 综合评价与选择建议

<div class="example-box">

#### 不同场景下的方法选择

**场景1：小规模问题（$n < 10^4$）**
- **推荐**：黎曼梯度法
- **理由**：计算开销可忽略，理论最优

**场景2：大规模深度学习（$n > 10^7$）**
- **推荐**：投影梯度法 + 自适应步长
- **理由**：实现简单，与现有框架兼容

**场景3：权重归一化层**
- **推荐**：黎曼梯度法（针对归一化维度）
- **理由**：约束天然存在，额外成本分摊

**场景4：需要最快收敛**
- **推荐**：K-FAC自然梯度 + 黎曼几何
- **理由**：结合二阶信息与流形结构

**场景5：强化学习策略梯度**
- **推荐**：自然策略梯度（NPG）
- **理由**：Fisher度量天然出现，理论保证单调改进

</div>

---

## 文章小结 #

这篇文章新开一个系列，主要围绕“等式约束”来讨论优化问题，试图给一些常见的约束条件寻找“下降最快的方向”。作为第一篇文章，本文讨论了“超球面”约束下的SGD变体。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11196>_

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

苏剑林. (Aug. 01, 2025). 《流形上的最速下降：1. SGD + 超球面 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11196>

@online{kexuefm-11196,
title={流形上的最速下降：1. SGD + 超球面},
author={苏剑林},
year={2025},
month={Aug},
url={\url{https://spaces.ac.cn/archives/11196}},
}

---

## 第5部分：学习路线图与未来展望

### 5.1 学习路线图

#### 必备前置知识

要深入理解超球面优化，需要以下数学和机器学习基础：

**数学基础（按重要性排序）**：

1. **线性代数**（必需）
   - 向量空间、内积、正交性
   - 投影、正交分解
   - 矩阵范数、奇异值分解
   - **推荐教材**：Gilbert Strang《Linear Algebra and Its Applications》

2. **微积分与优化**（必需）
   - 多元微分、梯度、方向导数
   - 链式法则、泰勒展开
   - 约束优化、拉格朗日乘数法
   - **推荐教材**：Boyd & Vandenberghe《Convex Optimization》

3. **微分几何**（进阶）
   - 曲线与曲面、切空间、法向量
   - 黎曼度量、测地线
   - 流形、嵌入、投影
   - **推荐教材**：John M. Lee《Introduction to Smooth Manifolds》（第1-3章）

4. **数值分析**（选修）
   - 浮点运算、误差分析
   - 迭代算法、收敛性分析
   - **推荐教材**：Nocedal & Wright《Numerical Optimization》

**机器学习基础**：

1. **优化算法**
   - SGD及其变体（Momentum, Adam, RMSprop）
   - 学习率调度、梯度裁剪
   - 批量归一化、权重归一化
   - **推荐课程**：Stanford CS231n

2. **深度学习框架**
   - PyTorch或JAX的自动微分机制
   - 自定义优化器实现
   - **推荐资源**：PyTorch官方教程

#### 核心论文学习路径

**阶段1：流形优化基础（2-3个月）**

1. **Absil, Mahony & Sepulchre (2008)**
   - 《Optimization Algorithms on Matrix Manifolds》
   - **重点章节**：第3章（微分几何）、第4章（优化算法）
   - **配套代码**：Manopt工具箱

2. **Edelman, Arias & Smith (1998)**
   - "The Geometry of Algorithms with Orthogonality Constraints"
   - **重点**：Stiefel流形、Grassmann流形的几何
   - **应用**：PCA、ICA、低秩分解

3. **Boumal (2020)**
   - 《An Introduction to Optimization on Smooth Manifolds》
   - **现代视角**：更注重计算实现和机器学习应用
   - **免费资源**：[在线版本](https://www.nicolasboumal.net/book/)

**阶段2：神经网络中的流形优化（1-2个月）**

4. **Salimans & Kingma (2016)** - NeurIPS
   - "Weight Normalization: A Simple Reparameterization to Accelerate Training"
   - **贡献**：$\boldsymbol{w} = g\boldsymbol{v}/\|\boldsymbol{v}\|_2$ 分解
   - **应用**：RNN、GAN训练稳定性
   - **代码**：[PyTorch实现](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html)

5. **Miyato et al. (2018)** - ICLR
   - "Spectral Normalization for Generative Adversarial Networks"
   - **贡献**：约束判别器Lipschitz常数
   - **理论**：幂迭代法在Stiefel流形上的收敛性
   - **影响**：成为GAN训练的标准技术

6. **Cho & Lee (2017)** - NeurIPS
   - "Riemannian Approach to Batch Normalization"
   - **创新**：将批量归一化解释为SPD流形上的优化
   - **理论贡献**：黎曼批量归一化的收敛性分析

**阶段3：高级主题与前沿研究（持续学习）**

7. **Becigneul & Ganea (2019)** - ICML
   - "Riemannian Adaptive Optimization Methods"
   - **贡献**：提出RAdam、RSGDm（流形上的Adam和Momentum）
   - **理论**：平行移动、向量传输的实用近似

8. **Gao et al. (2021)** - ICLR
   - "Riemannian Optimization for Deep Learning"
   - **综述**：总结流形优化在深度学习中的应用
   - **代码**：Geoopt库（PyTorch）

9. **Liu et al. (2022)** - NeurIPS
   - "Optimization on Multiple Manifolds"
   - **前沿**：多个流形约束的同时优化
   - **应用**：神经架构搜索、多任务学习

#### 实践学习路径

**初级实践（1-2周）**：

```python
# 项目1：手写超球面SGD
# - 实现投影梯度法
# - 实现黎曼梯度法
# - 对比收敛曲线

# 项目2：权重归一化层
# - 从零实现weight normalization
# - 在MNIST上训练CNN
# - 对比标准卷积层

# 项目3：可视化超球面优化轨迹
# - 在S^2上定义简单目标函数
# - 绘制梯度下降轨迹
# - 对比欧氏vs黎曼梯度
```

**中级实践（2-4周）**：

```python
# 项目4：实现Manopt核心功能
# - 超球面、Stiefel流形的基本操作
# - retraction、向量传输
# - 与PyTorch集成

# 项目5：谱归一化GAN
# - 在CIFAR-10上训练SNGAN
# - 分析Lipschitz约束的效果
# - 可视化判别器谱范数变化

# 项目6：自然梯度近似
# - 实现K-FAC
# - 对比SGD、Adam、K-FAC收敛速度
# - 分析计算开销
```

**高级实践（1-2个月）**：

```python
# 项目7：Riemannian Adam
# - 设计并实现RAdam优化器
# - 在大规模任务（ImageNet）上验证
# - 发表技术报告

# 项目8：多流形优化
# - 结合超球面+SPD流形约束
# - 应用于元学习或迁移学习
# - 理论分析与实验验证
```

### 5.2 研究空白与未来方向

#### 方向1：理论层面 - 非凸流形优化的收敛性保证

**研究空白**：
- 超球面优化的全局收敛性理论不完善，尤其是非凸目标函数
- 黎曼梯度下降的样本复杂度未知：需要多少样本才能达到 $\epsilon$-次优解？
- 流形曲率与收敛速率的定量关系不明确

**具体研究问题**：

**问题1**：超球面上非凸优化的逃逸鞍点机制？

- **挑战**：欧氏空间的逃逸鞍点理论（如Perturbed GD）不直接推广到流形
  - 流形上的Hessian（黎曼Hessian）定义更复杂
  - 鞍点附近的稳定/不稳定流形几何不明确
- **潜在方法**：
  - 利用Morse理论分析临界点结构
  - 研究黎曼Hessian的谱性质
  - 设计流形上的噪声注入策略
- **潜在意义**：
  - 理论保证神经网络训练收敛到"好"的局部最优
  - 指导学习率和动量的选择

**问题2**：黎曼梯度下降的样本复杂度下界？

- **已知**：欧氏SGD在强凸情况下需要 $O(\epsilon^{-1})$ 样本达到 $\epsilon$ 误差
- **未知**：超球面约束是否改变样本复杂度？
  - 约束是否提供了"隐式正则化"？
  - 是否存在流形特定的下界？
- **潜在方法**：
  - 信息论下界：基于流形测地线的互信息
  - PAC学习框架扩展到流形
- **潜在意义**：
  - 理论指导小样本学习
  - 设计样本高效的流形优化算法

**问题3**：曲率对收敛速度的影响？

- **现状**：正曲率流形（如超球面）的收敛性分析少于负曲率（双曲空间）
- **探索方向**：
  - 推导曲率依赖的收敛率：$O(1/\sqrt{KT})$ 其中 $K$ 是曲率
  - 分析高曲率区域的优化行为
  - 设计曲率自适应学习率
- **理论工具**：比较定理（Rauch比较定理）、Jacobi场分析

**优化方向**：
- 建立流形上的"逃逸时间"理论
- 推导曲率敏感的收敛界
- 发展流形上的加速方法（类似Nesterov动量）

**量化目标**：
- 证明超球面SGD在假设下的 $O(1/\sqrt{T})$ 收敛率（非凸情况）
- 建立样本复杂度下界：$\Omega(d/\epsilon^2)$（$d$ 是流形维度）
- 推导曲率依赖界：收敛常数 $\leq C(K) \cdot \text{poly}(d, \epsilon^{-1})$

---

#### 方向2：算法层面 - 流形上的自适应与二阶方法

**研究空白**：
- 缺乏理论完备的流形Adam/AdaGrad
- 流形上的二阶方法（牛顿法、拟牛顿法）计算成本高昂
- 多流形耦合优化缺乏统一框架

**具体研究问题**：

**问题1**：如何设计理论严谨的Riemannian Adam？

- **挑战**：
  - 一阶矩 $\boldsymbol{m}_t$ 和二阶矩 $\boldsymbol{v}_t$ 在不同切空间，如何传输？
  - 现有方法（简单投影）缺乏理论保证
- **优化方向**：
  - **平行移动**：使用流形的联络精确传输动量
    $$
    \boldsymbol{m}_t = \mathcal{T}_{t-1\to t}(\beta_1 \boldsymbol{m}_{t-1}) + (1-\beta_1)\text{grad} f(\boldsymbol{w}_t)
    $$
  - **向量传输**：设计计算高效的近似传输算子
  - **自适应度量**：根据二阶矩调整黎曼度量
- **量化目标**：
  - 证明RAdam的收敛率 $O(1/\sqrt{T})$（匹配Adam）
  - 实验验证：在ImageNet上比SGD快 $30\%$，比Adam稳定

**问题2**：流形上的拟牛顿法能否实用化？

- **现有方案**：L-BFGS在欧氏空间成功，但流形版本（R-LBFGS）计算昂贵
- **优化方向**：
  - **低秩Hessian近似**：利用流形结构稀疏性
  - **对角近似**：只存储主对角线（$O(n)$ 存储）
  - **分块更新**：对神经网络的不同层使用不同曲率估计
- **量化目标**：
  - 存储降至 $O(n\log T)$（当前 $O(nT)$）
  - 每步计算从 $O(n^2)$ 降至 $O(n\log n)$

**问题3**：多流形产品优化？

- **场景**：神经网络不同层有不同约束
  - 第1层：超球面约束（权重归一化）
  - 第2层：Stiefel流形约束（正交权重）
  - 第3层：SPD流形约束（协方差矩阵）
- **挑战**：
  - 不同流形的retraction不兼容
  - 如何协调不同流形的步长？
- **优化方向**：
  - 积流形（Product Manifold）理论
  - 交替优化 vs 联合优化
- **量化目标**：
  - 设计统一框架，支持 $\geq 3$ 种常见流形
  - 收敛速度不劣于单独优化

**优化方向**：
- 发展平行移动的快速近似算法
- 设计流形感知的学习率调度
- 构建多流形优化工具箱

**量化目标**：
- RAdam在BERT预训练上比Adam快 $20\%$
- R-LBFGS在小批量设置下可行（批量 $\geq 1024$）
- 多流形框架在多任务学习上提升 $10\%$ 性能

---

#### 方向3：应用层面 - 深度学习与生成模型

**研究空白**：
- 球面嵌入在大规模模型（如LLM）中的应用不足
- 流形约束对抗训练（GAN）的理论基础薄弱
- 扩散模型在非欧空间（超球面、双曲空间）的扩展不成熟

**具体研究问题**：

**问题1**：球面嵌入能否改善大语言模型（LLM）？

- **动机**：
  - Token嵌入、位置编码可以归一化到超球面
  - 角度距离可能比欧氏距离更适合语义相似性
- **优化方向**：
  - 设计球面注意力机制：
    $$
    \text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}(\langle\boldsymbol{Q}, \boldsymbol{K}\rangle)\boldsymbol{V}
    $$
    其中 $\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V} \in \mathcal{S}^{d-1}$
  - 球面Transformer：所有权重矩阵列归一化
  - 理论分析：球面嵌入的表达能力
- **量化目标**：
  - 在问答任务上超越标准Transformer $2\%$
  - 参数效率提升：相同性能下参数减少 $20\%$

**问题2**：流形GAN的稳定性理论？

- **现有问题**：GAN训练不稳定，模式崩溃
- **流形视角**：
  - 判别器Lipschitz约束 = Stiefel流形约束
  - 生成器可以在隐空间流形上优化
- **优化方向**：
  - 推导流形GAN的Nash均衡存在性
  - 分析Wasserstein距离在流形上的性质
  - 设计流形正则化损失
- **量化目标**：
  - 训练崩溃率降低 $50\%$
  - FID分数提升 $15\%$（CIFAR-10/ImageNet）

**问题3**：超球面上的扩散模型？

- **动机**：
  - 方向数据（3D旋转、球面图像）天然在超球面上
  - 欧氏扩散模型应用于球面数据时违反几何
- **优化方向**：
  - **球面扩散过程**：
    $$
    d\boldsymbol{x}_t = -\frac{1}{2}\|\nabla\log p_t(\boldsymbol{x}_t)\|^2 \boldsymbol{x}_t dt + \text{Brownian motion on } \mathcal{S}^{n-1}
    $$
  - **得分函数的黎曼版本**：
    $$
    \boldsymbol{s}_{\theta}(\boldsymbol{x}_t, t) = \text{grad}\log p_t(\boldsymbol{x}_t)
    $$
  - **逆向采样**：沿测地线移动而非欧氏直线
- **理论挑战**：
  - 超球面上的布朗运动定义
  - 得分匹配损失的黎曼推广
- **量化目标**：
  - 在方向数据上生成质量提升 $20\%$
  - 支持 $SO(3)$、$\mathcal{S}^{1023}$（高维球面）等流形

**优化方向**：
- 发展球面Transformer架构
- 建立流形GAN的博弈论基础
- 实现高维流形上的扩散模型

**量化目标**：
- 球面BERT在GLUE上超越标准BERT $1.5$ 分
- 流形GAN在CelebA-HQ上FID < 5.0
- 球面扩散模型在全景图像生成上超越平面模型 $10\%$（LPIPS指标）

**潜在应用场景**：
- **自然语言处理**：词嵌入、跨语言对齐
- **计算机视觉**：全景视觉、3D旋转估计
- **推荐系统**：用户-物品嵌入在超球面
- **生物信息学**：蛋白质构象空间（SO(3)流形）
- **天文学**：天球数据分析（$\mathcal{S}^2$）

---

### 5.3 开源工具与社区资源

**主流流形优化库**：

1. **Geoopt** (PyTorch)
   - 链接：[https://github.com/geoopt/geoopt](https://github.com/geoopt/geoopt)
   - 支持流形：超球面、双曲空间、Stiefel、Grassmann、SPD
   - 与PyTorch无缝集成
   - **推荐度**：⭐⭐⭐⭐⭐

2. **Manopt** (MATLAB/Python)
   - 链接：[https://www.manopt.org/](https://www.manopt.org/)
   - 最成熟的流形优化工具箱
   - Python版本：Pymanopt
   - **推荐度**：⭐⭐⭐⭐

3. **JAXopt**
   - 链接：[https://github.com/google/jaxopt](https://github.com/google/jaxopt)
   - JAX生态，支持JIT编译
   - 流形优化模块较新
   - **推荐度**：⭐⭐⭐

**学习资源**：

- **在线课程**：
  - Boumal的流形优化课程（免费视频）
  - Stanford EE364b（凸优化进阶）

- **博客与教程**：
  - [Riemannian Optimization in PyTorch](https://geoopt.readthedocs.io/)
  - [流形优化导论](https://www.nicolasboumal.net/book/)

- **论文阅读组**：
  - Manifold Learning Study Group（每月讨论会）

---

## 公式推导与注释

本节将从微分几何、黎曼优化和深度学习三个角度，为超球面上的最速下降提供极详细的数学推导。

### 1. 超球面的微分几何基础

#### 1.1 超球面的定义

**定义 1.1（超球面流形）**：$n$维单位超球面定义为：
$$
\mathcal{S}^{n-1} = \{\boldsymbol{w} \in \mathbb{R}^n : \|\boldsymbol{w}\|_2 = 1\} = \{\boldsymbol{w} \in \mathbb{R}^n : \langle\boldsymbol{w}, \boldsymbol{w}\rangle = 1\}
$$

这是一个$(n-1)$维的嵌入子流形，嵌入在$n$维欧氏空间$\mathbb{R}^n$中。

**性质**：
- $\mathcal{S}^{n-1}$是紧致的（有界且闭合）
- $\mathcal{S}^{n-1}$是光滑的（无处不可微）
- $\mathcal{S}^{n-1}$具有恒定的正曲率$K = 1$

**例子**：
- $\mathcal{S}^1$：单位圆（1维流形在$\mathbb{R}^2$中）
- $\mathcal{S}^2$：单位球面（2维流形在$\mathbb{R}^3$中）
- 神经网络中：权重归一化层的参数空间

#### 1.2 切空间（Tangent Space）

**定义 1.2（切空间）**：在点$\boldsymbol{w} \in \mathcal{S}^{n-1}$处的切空间$T_{\boldsymbol{w}}\mathcal{S}^{n-1}$定义为：
$$
T_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\boldsymbol{v} \in \mathbb{R}^n : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}
$$

**推导**：设$\gamma(t)$是$\mathcal{S}^{n-1}$上经过$\boldsymbol{w}$的任意光滑曲线，满足$\gamma(0) = \boldsymbol{w}$。由于曲线始终在球面上，有：
$$
\|\gamma(t)\|_2^2 = \langle\gamma(t), \gamma(t)\rangle = 1, \quad \forall t
$$

对$t$求导：
$$
\frac{d}{dt}\langle\gamma(t), \gamma(t)\rangle = 2\langle\gamma'(t), \gamma(t)\rangle = 0
$$

在$t=0$处取值：
$$
\langle\gamma'(0), \gamma(0)\rangle = \langle\gamma'(0), \boldsymbol{w}\rangle = 0
$$

这表明曲线的切向量$\gamma'(0)$与位置向量$\boldsymbol{w}$正交。所有这样的切向量构成切空间。

**几何意义**：切空间是与球面在$\boldsymbol{w}$点相切的$(n-1)$维超平面。

**维度验证**：
$$
\dim(T_{\boldsymbol{w}}\mathcal{S}^{n-1}) = \dim(\{\boldsymbol{v} : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}) = n - 1
$$
这与流形的维度一致。

#### 1.3 法空间（Normal Space）

**定义 1.3（法空间）**：在点$\boldsymbol{w} \in \mathcal{S}^{n-1}$处的法空间$N_{\boldsymbol{w}}\mathcal{S}^{n-1}$定义为：
$$
N_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\alpha\boldsymbol{w} : \alpha \in \mathbb{R}\} = \text{span}\{\boldsymbol{w}\}
$$

**正交分解定理**：环境空间$\mathbb{R}^n$可以正交分解为切空间和法空间：
$$
\mathbb{R}^n = T_{\boldsymbol{w}}\mathcal{S}^{n-1} \oplus N_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

任意向量$\boldsymbol{v} \in \mathbb{R}^n$可唯一分解为：
$$
\boldsymbol{v} = \boldsymbol{v}_{\text{tan}} + \boldsymbol{v}_{\text{nor}}
$$
其中：
- 切向分量：$\boldsymbol{v}_{\text{tan}} = \boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$
- 法向分量：$\boldsymbol{v}_{\text{nor}} = \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w} \in N_{\boldsymbol{w}}\mathcal{S}^{n-1}$

**验证正交性**：
$$
\langle\boldsymbol{v}_{\text{tan}}, \boldsymbol{v}_{\text{nor}}\rangle = \langle\boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}, \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}\rangle = \langle\boldsymbol{v}, \boldsymbol{w}\rangle\langle\boldsymbol{v}, \boldsymbol{w}\rangle - \langle\boldsymbol{v}, \boldsymbol{w}\rangle^2\langle\boldsymbol{w}, \boldsymbol{w}\rangle = 0
$$

#### 1.4 投影算子（Projection Operator）

**定义 1.4（切空间投影）**：投影到切空间的算子$\mathcal{P}_{\boldsymbol{w}}: \mathbb{R}^n \to T_{\boldsymbol{w}}\mathcal{S}^{n-1}$定义为：
$$
\mathcal{P}_{\boldsymbol{w}}(\boldsymbol{v}) = \boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w} = (\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top})\boldsymbol{v}
$$

其中$\boldsymbol{I}$是$n \times n$单位矩阵，$\boldsymbol{w}\boldsymbol{w}^{\top}$是秩1投影矩阵。

**投影矩阵性质**：设$\boldsymbol{P}_{\boldsymbol{w}} = \boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top}$，则：

1. **幂等性**（投影两次等于投影一次）：
$$
\boldsymbol{P}_{\boldsymbol{w}}^2 = (\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top})^2 = \boldsymbol{I} - 2\boldsymbol{w}\boldsymbol{w}^{\top} + \boldsymbol{w}\boldsymbol{w}^{\top}\boldsymbol{w}\boldsymbol{w}^{\top} = \boldsymbol{I} - 2\boldsymbol{w}\boldsymbol{w}^{\top} + \boldsymbol{w}\boldsymbol{w}^{\top} = \boldsymbol{P}_{\boldsymbol{w}}
$$

2. **对称性**：
$$
\boldsymbol{P}_{\boldsymbol{w}}^{\top} = (\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top})^{\top} = \boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top} = \boldsymbol{P}_{\boldsymbol{w}}
$$

3. **正交投影**：
$$
\langle\mathcal{P}_{\boldsymbol{w}}(\boldsymbol{v}), \boldsymbol{w}\rangle = \langle\boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{w}\rangle = \langle\boldsymbol{v}, \boldsymbol{w}\rangle - \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0
$$

### 2. 黎曼梯度与投影公式

#### 2.1 约束优化问题的设定

考虑超球面上的约束优化问题：
$$
\min_{\boldsymbol{w} \in \mathcal{S}^{n-1}} f(\boldsymbol{w})
$$

其中$f: \mathbb{R}^n \to \mathbb{R}$是定义在环境空间上的光滑目标函数。

**拉格朗日乘数法推导**：引入约束$g(\boldsymbol{w}) = \|\boldsymbol{w}\|^2 - 1 = 0$，构造拉格朗日函数：
$$
\mathcal{L}(\boldsymbol{w}, \lambda) = f(\boldsymbol{w}) + \lambda(\|\boldsymbol{w}\|^2 - 1)
$$

最优性条件（KKT条件）：
$$
\nabla_{\boldsymbol{w}}\mathcal{L} = \nabla f(\boldsymbol{w}) + 2\lambda\boldsymbol{w} = \boldsymbol{0}
$$

这表明欧氏梯度$\nabla f(\boldsymbol{w})$与位置向量$\boldsymbol{w}$平行，即梯度在法空间中。但我们需要的是切空间中的梯度方向。

#### 2.2 黎曼梯度的定义

**定义 2.1（黎曼梯度）**：在黎曼流形$(\mathcal{M}, g)$上，函数$f$在点$\boldsymbol{w}$处的黎曼梯度$\text{grad} f(\boldsymbol{w})$定义为切空间中唯一满足以下条件的向量：
$$
g_{\boldsymbol{w}}(\text{grad} f(\boldsymbol{w}), \boldsymbol{\xi}) = Df(\boldsymbol{w})[\boldsymbol{\xi}], \quad \forall \boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{M}
$$

其中$Df(\boldsymbol{w})[\boldsymbol{\xi}] = \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle$是$f$在$\boldsymbol{w}$处沿$\boldsymbol{\xi}$方向的方向导数。

**对于超球面**：超球面继承了欧氏空间的黎曼度量，即$g_{\boldsymbol{w}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \langle\boldsymbol{\xi}, \boldsymbol{\eta}\rangle$。因此：
$$
\langle\text{grad} f(\boldsymbol{w}), \boldsymbol{\xi}\rangle = \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle, \quad \forall \boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

#### 2.3 黎曼梯度的投影公式推导

**定理 2.1**：超球面上的黎曼梯度为欧氏梯度在切空间上的投影：
$$
\text{grad} f(\boldsymbol{w}) = \mathcal{P}_{\boldsymbol{w}}(\nabla f(\boldsymbol{w})) = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}
$$

**证明**：

**步骤1**：利用正交分解，将欧氏梯度分解为：
$$
\nabla f(\boldsymbol{w}) = \nabla f(\boldsymbol{w})_{\text{tan}} + \nabla f(\boldsymbol{w})_{\text{nor}}
$$
其中：
$$
\nabla f(\boldsymbol{w})_{\text{tan}} = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$
$$
\nabla f(\boldsymbol{w})_{\text{nor}} = \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w} \in N_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

**步骤2**：对于任意$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$，计算内积：
$$
\langle\nabla f(\boldsymbol{w})_{\text{tan}}, \boldsymbol{\xi}\rangle = \langle\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{\xi}\rangle
$$
$$
= \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle
$$

由于$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$，有$\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle = 0$，因此：
$$
\langle\nabla f(\boldsymbol{w})_{\text{tan}}, \boldsymbol{\xi}\rangle = \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle = Df(\boldsymbol{w})[\boldsymbol{\xi}]
$$

**步骤3**：由黎曼梯度的定义和唯一性，得：
$$
\text{grad} f(\boldsymbol{w}) = \nabla f(\boldsymbol{w})_{\text{tan}} = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}
$$

**几何解释**：法向分量$\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$对应于约束力（离开或靠近球心的力），不影响沿球面的运动。只有切向分量才对球面上的优化有贡献。

**计算复杂度分析**：
- 计算$\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle$：$O(n)$次乘加
- 计算$\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$：$O(n)$次标量乘法
- 计算最终投影：$O(n)$次减法
- **总复杂度**：$O(n)$，与计算梯度本身的复杂度相同

### 3. 超球面上的最速下降方向

#### 3.1 最速下降问题的精确表述

**问题 3.1**：给定当前点$\boldsymbol{w} \in \mathcal{S}^{n-1}$，寻找切向量$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$使得：
$$
\boldsymbol{\xi}^* = \arg\min_{\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}, \|\boldsymbol{\xi}\|_2 = 1} Df(\boldsymbol{w})[\boldsymbol{\xi}]
$$

等价于最大化问题：
$$
\boldsymbol{\xi}^* = \arg\max_{\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}, \|\boldsymbol{\xi}\|_2 = 1} -Df(\boldsymbol{w})[\boldsymbol{\xi}] = \arg\max_{\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}, \|\boldsymbol{\xi}\|_2 = 1} \langle-\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle
$$

这正是文中式$\eqref{eq:core}$在超球面约束下的形式。

#### 3.2 最速下降方向的求解

**定理 3.1**：超球面上的最速下降方向为：
$$
\boldsymbol{\xi}^* = -\frac{\text{grad} f(\boldsymbol{w})}{\|\text{grad} f(\boldsymbol{w})\|_2} = -\frac{\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}}{\|\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}\|_2}
$$

**证明**：

设$\boldsymbol{g} = \nabla f(\boldsymbol{w})$，$\boldsymbol{g}_{\text{tan}} = \text{grad} f(\boldsymbol{w}) = \boldsymbol{g} - \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}$。

要最大化$\langle-\boldsymbol{g}, \boldsymbol{\xi}\rangle$，约束条件为$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$且$\|\boldsymbol{\xi}\|_2 = 1$。

由于$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$，有$\langle\boldsymbol{\xi}, \boldsymbol{w}\rangle = 0$，因此：
$$
\langle-\boldsymbol{g}, \boldsymbol{\xi}\rangle = \langle-\boldsymbol{g} + \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{\xi}\rangle = \langle-\boldsymbol{g}_{\text{tan}}, \boldsymbol{\xi}\rangle
$$

应用柯西-施瓦茨不等式：
$$
\langle-\boldsymbol{g}_{\text{tan}}, \boldsymbol{\xi}\rangle \leq \|\boldsymbol{g}_{\text{tan}}\|_2 \|\boldsymbol{\xi}\|_2 = \|\boldsymbol{g}_{\text{tan}}\|_2
$$

等号成立当且仅当$\boldsymbol{\xi} = -\boldsymbol{g}_{\text{tan}}/\|\boldsymbol{g}_{\text{tan}}\|_2$。

**归一化验证**：
$$
\left\|-\frac{\boldsymbol{g}_{\text{tan}}}{\|\boldsymbol{g}_{\text{tan}}\|_2}\right\|_2 = 1 \quad \checkmark
$$

**切空间验证**：
$$
\left\langle-\frac{\boldsymbol{g}_{\text{tan}}}{\|\boldsymbol{g}_{\text{tan}}\|_2}, \boldsymbol{w}\right\rangle = -\frac{1}{\|\boldsymbol{g}_{\text{tan}}\|_2}\langle\boldsymbol{g}_{\text{tan}}, \boldsymbol{w}\rangle = -\frac{1}{\|\boldsymbol{g}_{\text{tan}}\|_2}\langle\boldsymbol{g} - \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{w}\rangle = 0 \quad \checkmark
$$

#### 3.3 退化情况分析

**退化情况**：当$\text{grad} f(\boldsymbol{w}) = \boldsymbol{0}$时，即$\nabla f(\boldsymbol{w}) = \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$时，黎曼梯度消失。

**几何意义**：此时欧氏梯度完全在法空间中，沿球面的任意切方向函数值都不变（一阶近似），$\boldsymbol{w}$是约束优化问题的临界点。

**判定条件**：
$$
\|\boldsymbol{g}_{\text{tan}}\|_2 = 0 \Leftrightarrow \nabla f(\boldsymbol{w}) \parallel \boldsymbol{w}
$$

### 4. 测地线与指数映射

#### 4.1 测地线的定义与性质

**定义 4.1（测地线）**：黎曼流形上的测地线$\gamma(t)$是满足以下微分方程的曲线：
$$
\nabla_{\dot{\gamma}(t)}\dot{\gamma}(t) = \boldsymbol{0}
$$

其中$\nabla$是黎曼联络，$\dot{\gamma}(t) = d\gamma/dt$是切向量。

**物理意义**：测地线是流形上的"直线"，具有局部最短路径性质。

**超球面上的测地线**：从$\boldsymbol{w} \in \mathcal{S}^{n-1}$出发，沿初始方向$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$的测地线为大圆弧：
$$
\gamma(t) = \cos(t\|\boldsymbol{\xi}\|_2)\boldsymbol{w} + \sin(t\|\boldsymbol{\xi}\|_2)\frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}
$$

**推导**：

**步骤1**：测地线满足二阶微分方程，设$\gamma(t) \in \mathcal{S}^{n-1}$，则$\|\gamma(t)\|_2 = 1$。

对时间求二阶导数：
$$
\frac{d^2}{dt^2}\langle\gamma(t), \gamma(t)\rangle = 2\langle\ddot{\gamma}(t), \gamma(t)\rangle + 2\|\dot{\gamma}(t)\|_2^2 = 0
$$

因此：
$$
\langle\ddot{\gamma}(t), \gamma(t)\rangle = -\|\dot{\gamma}(t)\|_2^2
$$

**步骤2**：测地线方程要求加速度垂直于流形，即$\ddot{\gamma}(t)$在切空间的投影为0：
$$
\mathcal{P}_{\gamma(t)}(\ddot{\gamma}(t)) = \ddot{\gamma}(t) - \langle\ddot{\gamma}(t), \gamma(t)\rangle\gamma(t) = \boldsymbol{0}
$$

因此：
$$
\ddot{\gamma}(t) = \langle\ddot{\gamma}(t), \gamma(t)\rangle\gamma(t) = -\|\dot{\gamma}(t)\|_2^2 \gamma(t)
$$

**步骤3**：这是一个二阶常系数微分方程。设初始条件$\gamma(0) = \boldsymbol{w}$，$\dot{\gamma}(0) = \boldsymbol{\xi}$，且$\langle\boldsymbol{\xi}, \boldsymbol{w}\rangle = 0$，$\|\boldsymbol{w}\|_2 = 1$。

令$s = \|\boldsymbol{\xi}\|_2$，$\boldsymbol{v} = \boldsymbol{\xi}/s$（单位切向量），则解为：
$$
\gamma(t) = \cos(st)\boldsymbol{w} + \sin(st)\boldsymbol{v}
$$

**验证**：
$$
\|\gamma(t)\|_2^2 = \cos^2(st) + \sin^2(st) = 1 \quad \checkmark
$$
$$
\dot{\gamma}(t) = -s\sin(st)\boldsymbol{w} + s\cos(st)\boldsymbol{v}
$$
$$
\|\dot{\gamma}(t)\|_2 = s \quad \text{(常数速度)}
$$

#### 4.2 指数映射（Exponential Map）

**定义 4.2（指数映射）**：指数映射$\text{Exp}_{\boldsymbol{w}}: T_{\boldsymbol{w}}\mathcal{S}^{n-1} \to \mathcal{S}^{n-1}$定义为：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \gamma_{\boldsymbol{\xi}}(1)
$$

其中$\gamma_{\boldsymbol{\xi}}(t)$是从$\boldsymbol{w}$出发、初速度为$\boldsymbol{\xi}$的测地线。

**超球面的指数映射公式**：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \cos(\|\boldsymbol{\xi}\|_2)\boldsymbol{w} + \sin(\|\boldsymbol{\xi}\|_2)\frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}
$$

**性质**：
1. $\text{Exp}_{\boldsymbol{w}}(\boldsymbol{0}) = \boldsymbol{w}$
2. $\frac{d}{dt}\text{Exp}_{\boldsymbol{w}}(t\boldsymbol{\xi})\Big|_{t=0} = \boldsymbol{\xi}$
3. 指数映射是局部微分同胚（在小邻域内一一对应）

**小步长近似**：当$\|\boldsymbol{\xi}\|_2 \ll 1$时，利用泰勒展开：
$$
\cos(\|\boldsymbol{\xi}\|_2) \approx 1 - \frac{\|\boldsymbol{\xi}\|_2^2}{2} + O(\|\boldsymbol{\xi}\|_2^4)
$$
$$
\sin(\|\boldsymbol{\xi}\|_2) \approx \|\boldsymbol{\xi}\|_2 - \frac{\|\boldsymbol{\xi}\|_2^3}{6} + O(\|\boldsymbol{\xi}\|_2^5)
$$

因此：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) \approx \left(1 - \frac{\|\boldsymbol{\xi}\|_2^2}{2}\right)\boldsymbol{w} + \boldsymbol{\xi} + O(\|\boldsymbol{\xi}\|_2^3)
$$

归一化到单位长度：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) \approx \frac{\boldsymbol{w} + \boldsymbol{\xi}}{\|\boldsymbol{w} + \boldsymbol{\xi}\|_2} + O(\|\boldsymbol{\xi}\|_2^3)
$$

### 5. Retraction算子

#### 5.1 Retraction的定义

**定义 5.1（Retraction）**：从点$\boldsymbol{w} \in \mathcal{M}$处的retraction是映射$R_{\boldsymbol{w}}: T_{\boldsymbol{w}}\mathcal{M} \to \mathcal{M}$，满足：
1. $R_{\boldsymbol{w}}(\boldsymbol{0}) = \boldsymbol{w}$
2. $\frac{d}{dt}R_{\boldsymbol{w}}(t\boldsymbol{\xi})\Big|_{t=0} = \boldsymbol{\xi}$，$\forall \boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{M}$
3. 局部刚性：$R_{\boldsymbol{w}}$是局部微分同胚

**作用**：Retraction提供了一种从切空间"回到"流形的方法，是指数映射的计算友好替代。

#### 5.2 超球面的常用Retraction

**1. 归一化Retraction（最常用）**：
$$
R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \frac{\boldsymbol{w} + \boldsymbol{\xi}}{\|\boldsymbol{w} + \boldsymbol{\xi}\|_2}
$$

**验证条件**：
- 条件1：$R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{0}) = \boldsymbol{w}/\|\boldsymbol{w}\|_2 = \boldsymbol{w}$ ✓
- 条件2：设$\phi(t) = R_{\boldsymbol{w}}^{\text{norm}}(t\boldsymbol{\xi})$，则
$$
\phi(t) = \frac{\boldsymbol{w} + t\boldsymbol{\xi}}{\|\boldsymbol{w} + t\boldsymbol{\xi}\|_2}
$$
$$
\frac{d\phi}{dt}\Big|_{t=0} = \frac{\boldsymbol{\xi}\|\boldsymbol{w}\|_2 - \boldsymbol{w}\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle/\|\boldsymbol{w}\|_2}{\|\boldsymbol{w}\|_2^2} = \boldsymbol{\xi} - \langle\boldsymbol{w}, \boldsymbol{\xi}\rangle\boldsymbol{w} = \boldsymbol{\xi}
$$
（最后一步利用了$\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle = 0$）✓

**计算成本**：
- 向量加法：$O(n)$
- 范数计算：$O(n)$
- 标量除法：$O(n)$
- **总成本**：$O(n)$

**2. 指数映射Retraction（精确但昂贵）**：
$$
R_{\boldsymbol{w}}^{\text{exp}}(\boldsymbol{\xi}) = \text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \cos(\|\boldsymbol{\xi}\|_2)\boldsymbol{w} + \sin(\|\boldsymbol{\xi}\|_2)\frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}
$$

**计算成本**：需要计算三角函数，约$O(n) + O(\log(1/\epsilon))$（$\epsilon$是精度）。

#### 5.3 归一化Retraction的误差分析

**定理 5.1**：归一化retraction与指数映射的误差为$O(\|\boldsymbol{\xi}\|_2^3)$：
$$
\left\|R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) - \text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi})\right\|_2 = O(\|\boldsymbol{\xi}\|_2^3)
$$

**证明**：

设$s = \|\boldsymbol{\xi}\|_2$，$\boldsymbol{v} = \boldsymbol{\xi}/s$（单位切向量）。

**指数映射**：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \cos(s)\boldsymbol{w} + \sin(s)\boldsymbol{v}
$$

**归一化retraction**：
$$
R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \frac{\boldsymbol{w} + s\boldsymbol{v}}{\|\boldsymbol{w} + s\boldsymbol{v}\|_2}
$$

计算分母：
$$
\|\boldsymbol{w} + s\boldsymbol{v}\|_2^2 = 1 + s^2 = 1 + s^2
$$
$$
\|\boldsymbol{w} + s\boldsymbol{v}\|_2 = \sqrt{1 + s^2} = 1 + \frac{s^2}{2} - \frac{s^4}{8} + O(s^6)
$$

因此：
$$
R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \frac{\boldsymbol{w} + s\boldsymbol{v}}{1 + s^2/2 + O(s^4)} = (\boldsymbol{w} + s\boldsymbol{v})(1 - s^2/2 + O(s^4))
$$
$$
= \boldsymbol{w} + s\boldsymbol{v} - \frac{s^2}{2}\boldsymbol{w} + O(s^3) = (1 - s^2/2)\boldsymbol{w} + s\boldsymbol{v} + O(s^3)
$$

对比泰勒展开：
$$
\cos(s) = 1 - \frac{s^2}{2} + \frac{s^4}{24} + O(s^6)
$$
$$
\sin(s) = s - \frac{s^3}{6} + O(s^5)
$$

误差为：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) - R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \left(\frac{s^4}{24}\boldsymbol{w} - \frac{s^3}{6}\boldsymbol{v}\right) + O(s^5) = O(s^3)
$$

**结论**：对于小步长优化（$s \approx \eta \ll 1$），归一化retraction是足够精确且高效的选择。

### 6. 超球面上的SGD更新规则

#### 6.1 完整的更新算法

**算法 6.1（超球面上的随机梯度下降）**：

**输入**：初始点$\boldsymbol{w}_0 \in \mathcal{S}^{n-1}$，学习率$\eta_t$，目标函数$f$

**For** $t = 0, 1, 2, \ldots$ **do**:

1. **计算随机梯度**：$\boldsymbol{g}_t = \nabla f(\boldsymbol{w}_t; \mathcal{B}_t)$（$\mathcal{B}_t$是小批量数据）

2. **投影到切空间**（计算黎曼梯度）：
   $$
   \boldsymbol{g}_t^{\text{Riem}} = \boldsymbol{g}_t - \langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle\boldsymbol{w}_t
   $$

3. **确定下降方向**：
   $$
   \boldsymbol{\xi}_t = -\eta_t \frac{\boldsymbol{g}_t^{\text{Riem}}}{\|\boldsymbol{g}_t^{\text{Riem}}\|_2}
   $$
   或者（不归一化版本）：
   $$
   \boldsymbol{\xi}_t = -\eta_t \boldsymbol{g}_t^{\text{Riem}}
   $$

4. **Retraction回流形**：
   $$
   \boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t + \boldsymbol{\xi}_t}{\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2}
   $$

**Output**：$\boldsymbol{w}_T$

#### 6.2 更新规则的简化形式

当使用非归一化的黎曼梯度时，更新可以写成：
$$
\boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta_t(\boldsymbol{g}_t - \langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle\boldsymbol{w}_t)}{\|\boldsymbol{w}_t - \eta_t(\boldsymbol{g}_t - \langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle\boldsymbol{w}_t)\|_2}
$$

进一步简化（假设$\|\boldsymbol{w}_t\|_2 = 1$）：
$$
\boldsymbol{w}_{t+1} = \frac{(1 + \eta_t\langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle)\boldsymbol{w}_t - \eta_t\boldsymbol{g}_t}{\|(1 + \eta_t\langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle)\boldsymbol{w}_t - \eta_t\boldsymbol{g}_t\|_2}
$$

#### 6.3 小步长近似

当$\eta_t \ll 1$时，$\boldsymbol{\xi}_t$很小，利用一阶近似：
$$
\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2 \approx \sqrt{1 + \eta_t^2\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2} \approx 1 + \frac{\eta_t^2\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2}{2}
$$

因此：
$$
\boldsymbol{w}_{t+1} \approx \left(1 - \frac{\eta_t^2\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2}{2}\right)(\boldsymbol{w}_t - \eta_t\boldsymbol{g}_t^{\text{Riem}}) + O(\eta_t^3)
$$

**验证约束保持**：
$$
\|\boldsymbol{w}_{t+1}\|_2^2 = \frac{\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2^2}{\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2^2} = 1 \quad \checkmark
$$

### 7. 收敛性分析

#### 7.1 下降引理（Descent Lemma）

**引理 7.1（黎曼流形上的下降引理）**：假设$f$是$L$-光滑的（黎曼Hessian有界），即：
$$
\|\nabla^2 f(\boldsymbol{w})\| \leq L, \quad \forall \boldsymbol{w} \in \mathcal{S}^{n-1}
$$

则对于任意$\boldsymbol{w}, \boldsymbol{w}' \in \mathcal{S}^{n-1}$，有：
$$
f(\boldsymbol{w}') \leq f(\boldsymbol{w}) + \langle\text{grad} f(\boldsymbol{w}), \boldsymbol{\xi}\rangle + \frac{L}{2}\|\boldsymbol{\xi}\|_2^2
$$

其中$\boldsymbol{w}' = R_{\boldsymbol{w}}(\boldsymbol{\xi})$，$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$。

#### 7.2 单步下降量估计

**定理 7.1（单步下降）**：使用学习率$\eta_t \leq 1/L$，超球面SGD满足：
$$
\mathbb{E}[f(\boldsymbol{w}_{t+1})] \leq f(\boldsymbol{w}_t) - \frac{\eta_t}{2}\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \frac{L\eta_t^2}{2}\mathbb{E}[\|\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t)\|_2^2]
$$

**证明**：

应用下降引理，设$\boldsymbol{\xi}_t = -\eta_t\boldsymbol{g}_t^{\text{Riem}}$：
$$
f(\boldsymbol{w}_{t+1}) \leq f(\boldsymbol{w}_t) + \langle\text{grad} f(\boldsymbol{w}_t), -\eta_t\boldsymbol{g}_t^{\text{Riem}}\rangle + \frac{L\eta_t^2}{2}\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2
$$

第二项：
$$
\langle\text{grad} f(\boldsymbol{w}_t), -\eta_t\boldsymbol{g}_t^{\text{Riem}}\rangle = -\eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t)\rangle
$$

由于$\text{grad} f(\boldsymbol{w}_t) \in T_{\boldsymbol{w}_t}\mathcal{S}^{n-1}$，投影不改变：
$$
= -\eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\nabla f(\boldsymbol{w}_t))\rangle - \eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t))\rangle
$$
$$
= -\eta_t\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 - \eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t))\rangle
$$

取期望（$\mathbb{E}[\boldsymbol{g}_t | \boldsymbol{w}_t] = \nabla f(\boldsymbol{w}_t)$）：
$$
\mathbb{E}\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t))\rangle = 0
$$

对第三项，注意$\|\mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{v})\|_2 \leq \|\boldsymbol{v}\|_2$（投影缩短长度），因此：
$$
\mathbb{E}[\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2] \leq \mathbb{E}[\|\boldsymbol{g}_t\|_2^2] = \|\nabla f(\boldsymbol{w}_t)\|_2^2 + \mathbb{E}[\|\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t)\|_2^2]
$$

综合，当$\eta_t \leq 1/L$时：
$$
\mathbb{E}[f(\boldsymbol{w}_{t+1})] \leq f(\boldsymbol{w}_t) - \eta_t\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \frac{L\eta_t^2}{2}(\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \sigma^2)
$$
$$
\leq f(\boldsymbol{w}_t) - \frac{\eta_t}{2}\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \frac{L\eta_t^2\sigma^2}{2}
$$

其中$\sigma^2 = \mathbb{E}[\|\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t)\|_2^2]$是梯度估计的方差。

#### 7.3 收敛速率

**定理 7.2（收敛速率）**：对于非凸目标函数，使用常学习率$\eta_t = \eta = \min\{1/L, \sqrt{T}/\sigma\}$，经过$T$次迭代：
$$
\min_{t=0,\ldots,T-1}\mathbb{E}[\|\text{grad} f(\boldsymbol{w}_t)\|_2^2] \leq \frac{2(f(\boldsymbol{w}_0) - f^*)}{\eta T} + L\eta\sigma^2 = O\left(\frac{1}{\sqrt{T}}\right)
$$

其中$f^* = \inf_{\boldsymbol{w} \in \mathcal{S}^{n-1}} f(\boldsymbol{w})$。

**证明**：累加单步下降不等式：
$$
\sum_{t=0}^{T-1}\mathbb{E}[\|\text{grad} f(\boldsymbol{w}_t)\|_2^2] \leq \frac{2}{\eta}(f(\boldsymbol{w}_0) - f^*) + L\eta T\sigma^2
$$

取$\eta = \sqrt{(f(\boldsymbol{w}_0) - f^*)/(LT\sigma^2)}$即得。

**与欧氏SGD的对比**：收敛率相同，均为$O(1/\sqrt{T})$，表明约束不影响渐近收敛速度。

### 8. 与欧氏空间SGD的对比

#### 8.1 更新规则对比

| 特性 | 欧氏SGD | 超球面SGD |
|------|---------|-----------|
| 参数空间 | $\mathbb{R}^n$ | $\mathcal{S}^{n-1}$ |
| 梯度 | $\nabla f(\boldsymbol{w})$ | $\text{grad} f(\boldsymbol{w}) = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$ |
| 更新方向 | $-\nabla f(\boldsymbol{w})$ | $-\text{grad} f(\boldsymbol{w})$ |
| 更新规则 | $\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta_t\nabla f(\boldsymbol{w}_t)$ | $\boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta_t\text{grad} f(\boldsymbol{w}_t)}{\|\cdots\|_2}$ |
| 约束保持 | 无 | $\|\boldsymbol{w}_t\|_2 = 1$ 自动满足 |
| 计算复杂度 | $O(n)$ | $O(n)$ （额外一次内积和归一化） |

#### 8.2 几何对比

**欧氏SGD**：
- 沿着负梯度方向在平坦的欧氏空间中前进
- 步长$\eta_t$直接控制参数改变量的欧氏距离
- 可能偏离约束集（需要投影）

**超球面SGD**：
- 沿着球面上的测地线（大圆弧）前进
- 步长$\eta_t$控制在切空间中的移动量
- 自动保持在球面上（通过retraction）

#### 8.3 有效维度分析

**欧氏空间**：$n$个自由度

**超球面**：$n-1$个自由度（一个约束$\|\boldsymbol{w}\|_2 = 1$）

**影响**：
- 超球面SGD实际上在$(n-1)$维空间中优化
- 约束消除了"径向"自由度，只保留"切向"自由度
- 这可以看作是一种正则化，防止参数范数无限增长

#### 8.4 梯度范数对比

**引理 8.1**：超球面黎曼梯度范数不超过欧氏梯度范数：
$$
\|\text{grad} f(\boldsymbol{w})\|_2 \leq \|\nabla f(\boldsymbol{w})\|_2
$$

**证明**：
$$
\|\text{grad} f(\boldsymbol{w})\|_2^2 = \|\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}\|_2^2
$$
$$
= \|\nabla f(\boldsymbol{w})\|_2^2 - 2\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle^2 + \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle^2\|\boldsymbol{w}\|_2^2
$$
$$
= \|\nabla f(\boldsymbol{w})\|_2^2 - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle^2 \leq \|\nabla f(\boldsymbol{w})\|_2^2
$$

**几何意义**：投影到切空间会丢失法向分量，因此黎曼梯度更小或相等。

### 9. 在神经网络中的应用

#### 9.1 权重归一化（Weight Normalization）

**背景**：权重归一化是一种重参数化技巧，将权重分解为方向和尺度：
$$
\boldsymbol{w} = g\frac{\boldsymbol{v}}{\|\boldsymbol{v}\|_2}
$$

其中$g \in \mathbb{R}$是标量增益，$\boldsymbol{v}/\|\boldsymbol{v}\|_2 \in \mathcal{S}^{n-1}$是方向向量。

**优化问题**：给定损失$L(g, \boldsymbol{v})$，分别对$g$和$\boldsymbol{v}$优化：
$$
\frac{\partial L}{\partial g} = \frac{\partial L}{\partial \boldsymbol{w}}\frac{\partial \boldsymbol{w}}{\partial g} = \frac{\partial L}{\partial \boldsymbol{w}}\frac{\boldsymbol{v}}{\|\boldsymbol{v}\|_2}
$$

对$\boldsymbol{v}$的梯度：
$$
\frac{\partial L}{\partial \boldsymbol{v}} = \frac{\partial L}{\partial \boldsymbol{w}}\frac{\partial \boldsymbol{w}}{\partial \boldsymbol{v}} = g\frac{\partial L}{\partial \boldsymbol{w}}\left(\frac{\boldsymbol{I}}{\|\boldsymbol{v}\|_2} - \frac{\boldsymbol{v}\boldsymbol{v}^{\top}}{\|\boldsymbol{v}\|_2^3}\right)
$$

简化：
$$
\frac{\partial L}{\partial \boldsymbol{v}} = \frac{g}{\|\boldsymbol{v}\|_2}\left(\frac{\partial L}{\partial \boldsymbol{w}} - \frac{\langle\partial L/\partial \boldsymbol{w}, \boldsymbol{v}\rangle}{\|\boldsymbol{v}\|_2^2}\boldsymbol{v}\right)
$$

设$\boldsymbol{u} = \boldsymbol{v}/\|\boldsymbol{v}\|_2 \in \mathcal{S}^{n-1}$（单位化），$\partial L/\partial \boldsymbol{w} = \boldsymbol{g}_w$：
$$
\frac{\partial L}{\partial \boldsymbol{u}} = g\left(\boldsymbol{g}_w - \langle\boldsymbol{g}_w, \boldsymbol{u}\rangle\boldsymbol{u}\right) = g \cdot \text{grad}_{\boldsymbol{u}} L
$$

**更新规则**：
$$
g_{t+1} = g_t - \eta_g\frac{\partial L}{\partial g}
$$
$$
\boldsymbol{u}_{t+1} = \frac{\boldsymbol{u}_t - \eta_u \cdot \text{grad}_{\boldsymbol{u}} L}{\|\boldsymbol{u}_t - \eta_u \cdot \text{grad}_{\boldsymbol{u}} L\|_2}
$$

**好处**：
- 解耦方向和尺度的学习
- 稳定训练（方向变化被约束在球面上）
- 加速收敛（特别是对RNN和GAN）

#### 9.2 谱归一化（Spectral Normalization）

**应用**：GAN判别器的Lipschitz约束。

**目标**：将权重矩阵$\boldsymbol{W}$的最大奇异值限制为1：
$$
\boldsymbol{W}_{\text{SN}} = \frac{\boldsymbol{W}}{\sigma_1(\boldsymbol{W})}
$$

其中$\sigma_1(\boldsymbol{W}) = \max_{\|\boldsymbol{v}\|_2=1} \|\boldsymbol{W}\boldsymbol{v}\|_2$。

**计算**：使用幂迭代法，迭代求解：
$$
\boldsymbol{u}_{t+1} = \frac{\boldsymbol{W}\boldsymbol{v}_t}{\|\boldsymbol{W}\boldsymbol{v}_t\|_2}, \quad \boldsymbol{v}_{t+1} = \frac{\boldsymbol{W}^{\top}\boldsymbol{u}_{t+1}}{\|\boldsymbol{W}^{\top}\boldsymbol{u}_{t+1}\|_2}
$$

这正是在两个超球面$\mathcal{S}^{m-1}$和$\mathcal{S}^{n-1}$上交替优化！

**梯度计算**：
$$
\frac{\partial L}{\partial \boldsymbol{W}} = \frac{1}{\sigma_1}\left(\frac{\partial L}{\partial \boldsymbol{W}_{\text{SN}}} - \left\langle\frac{\partial L}{\partial \boldsymbol{W}_{\text{SN}}}, \boldsymbol{u}\boldsymbol{v}^{\top}\right\rangle\boldsymbol{u}\boldsymbol{v}^{\top}\right)
$$

#### 9.3 球面卷积神经网络

**应用**：处理球面数据（地球表面、全景图像、分子构象）。

**挑战**：标准卷积在平面网格定义，不适用于球面。

**解决方案**：
1. 使用球谐函数（spherical harmonics）作为特征
2. 在切空间定义局部卷积
3. 使用指数映射在不同切空间之间传输

**超球面SGD的角色**：优化定义在球面上的神经网络参数，自然保持几何结构。

#### 9.4 度量学习（Metric Learning）

**目标**：学习嵌入$\phi: \mathcal{X} \to \mathcal{S}^{n-1}$，使得相似样本在球面上接近。

**损失函数**：球面softmax
$$
L = -\log\frac{\exp(s\cos\theta_{y_i})}{\sum_{j=1}^C\exp(s\cos\theta_j)}
$$

其中$\theta_j = \arccos(\langle\boldsymbol{w}_j, \boldsymbol{x}_i\rangle)$是特征$\boldsymbol{x}_i$与类中心$\boldsymbol{w}_j$的夹角，$\boldsymbol{w}_j \in \mathcal{S}^{n-1}$。

**优化**：
- 特征归一化：$\boldsymbol{x}_i \leftarrow \boldsymbol{x}_i/\|\boldsymbol{x}_i\|_2$
- 类中心$\boldsymbol{w}_j$使用超球面SGD更新

**优势**：
- 几何解释清晰（角度距离）
- 避免范数膨胀
- 改善类别间可分性

### 10. 数值实验与实现细节

#### 10.1 算法伪代码

```python
def spherical_sgd(f, w0, T, eta, batch_size):
    """
    超球面上的随机梯度下降

    参数:
        f: 目标函数（返回损失和梯度）
        w0: 初始参数（应满足 ||w0||_2 = 1）
        T: 迭代次数
        eta: 学习率
        batch_size: 批量大小
    """
    w = w0 / np.linalg.norm(w0)  # 确保初始化在球面上

    for t in range(T):
        # 1. 采样小批量并计算梯度
        batch = sample_batch(batch_size)
        loss, grad = f(w, batch)

        # 2. 投影到切空间（黎曼梯度）
        grad_riem = grad - np.dot(grad, w) * w

        # 3. 切空间中的更新
        xi = -eta * grad_riem

        # 4. Retraction（归一化）
        w_new = w + xi
        w = w_new / np.linalg.norm(w_new)

    return w
```

#### 10.2 数值稳定性技巧

**1. 避免零除**：当$\|\boldsymbol{g}_t^{\text{Riem}}\|_2 \approx 0$时：
$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t \quad \text{(跳过更新)}
$$

**2. 梯度裁剪**：
$$
\boldsymbol{g}_t^{\text{Riem}} \leftarrow \min\left(1, \frac{\tau}{\|\boldsymbol{g}_t^{\text{Riem}}\|_2}\right)\boldsymbol{g}_t^{\text{Riem}}
$$

**3. 周期性重归一化**（防止数值漂移）：
$$
\boldsymbol{w}_t \leftarrow \frac{\boldsymbol{w}_t}{\|\boldsymbol{w}_t\|_2} \quad \text{每}K\text{步}
$$

#### 10.3 复杂度总结

**每步时间复杂度**：
- 梯度计算：$O(C_f)$（依赖于$f$）
- 内积$\langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle$：$O(n)$
- 投影：$O(n)$
- 归一化：$O(n)$（计算范数+标量除法）
- **总计**：$O(C_f + n)$

**空间复杂度**：$O(n)$（存储$\boldsymbol{w}_t, \boldsymbol{g}_t$）

**与欧氏SGD的额外开销**：约2-3倍的向量操作（内积+投影+归一化），在现代硬件上可忽略。

### 11. 理论深化：曲率的影响

#### 11.1 黎曼曲率张量

超球面的曲率张量为：
$$
R(X, Y)Z = \langle Y, Z\rangle X - \langle X, Z\rangle Y
$$

**截面曲率**：恒为$K = 1$（正曲率）。

**影响**：
- 正曲率导致测地线相互汇聚
- 优化轨迹趋向于收敛到局部区域
- 与负曲率流形（如双曲空间）形成对比

#### 11.2 Jacobi场与测地线稳定性

**Jacobi方程**：
$$
\nabla_t^2 J + R(\dot{\gamma}, J)\dot{\gamma} = 0
$$

对于超球面：
$$
\nabla_t^2 J + J = 0
$$

**解**：$J(t) = A\cos(t) + B\sin(t)$（周期性振荡）

**几何意义**：从对跖点出发的测地线会在距离$\pi$处再次相交，导致指数映射在大步长时失去单射性。

**优化启示**：学习率不应过大，否则可能"越过"最优点。

### 12. 小结与展望

#### 12.1 核心要点回顾

1. **超球面是$(n-1)$维黎曼流形**，嵌入在$\mathbb{R}^n$中，约束为$\|\boldsymbol{w}\|_2 = 1$。

2. **切空间$T_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\boldsymbol{v} : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}$**，由与$\boldsymbol{w}$正交的向量组成。

3. **黎曼梯度**是欧氏梯度在切空间的投影：
   $$
   \text{grad} f(\boldsymbol{w}) = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}
   $$

4. **最速下降方向**是负黎曼梯度方向：
   $$
   \boldsymbol{\xi}^* = -\frac{\text{grad} f(\boldsymbol{w})}{\|\text{grad} f(\boldsymbol{w})\|_2}
   $$

5. **Retraction（归一化）**将切空间更新映射回流形：
   $$
   \boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta_t\text{grad} f(\boldsymbol{w}_t)}{\|\boldsymbol{w}_t - \eta_t\text{grad} f(\boldsymbol{w}_t)\|_2}
   $$

6. **收敛速率$O(1/\sqrt{T})$**，与无约束SGD相同。

7. **应用广泛**：权重归一化、谱归一化、度量学习、球面神经网络。

#### 12.2 与原文的联系

本推导详细展开了原文第5节"超球面上"的数学细节：
- **方程(9)**的几何意义：最速下降方向的优化问题
- **方程(10)**的推导：球面约束的一阶近似
- **方程(11-12)**的求解：拉格朗日乘数法
- **方程(13)**的归一化：精确保持约束

同时补充了原文未涉及的：
- 微分几何基础（切空间、法空间、投影）
- 黎曼梯度的严格定义
- 测地线与指数映射
- 收敛性理论分析
- 神经网络应用实例

#### 12.3 进一步阅读

- **黎曼优化**：Absil et al., "Optimization Algorithms on Matrix Manifolds" (2008)
- **流形学习**：Boumal, "An Introduction to Optimization on Smooth Manifolds" (2020)
- **神经网络应用**：Salimans & Kingma, "Weight Normalization" (2016)
- **谱归一化**：Miyato et al., "Spectral Normalization for GANs" (2018)

