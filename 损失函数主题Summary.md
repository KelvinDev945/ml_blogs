# 损失函数主题深度Summary

> **涵盖文章**：9篇损失函数相关文章
> **主要内容**：交叉熵、CoSENT、GlobalPointer、EMO、多任务损失

---

## 1. 核心理论

### 1.1 交叉熵的信息论基础

**KL散度**：
$$D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} = H(P, Q) - H(P)$$

**交叉熵**：
$$H(P, Q) = -\sum_i P(i) \log Q(i)$$

**最小化交叉熵** = **最小化KL散度**（当 $P$ 固定）

### 1.2 CoSENT推导

**目标**：句向量相似度排序

**Circle Loss变体**：
$$\mathcal{L} = \log\left(1 + \sum_{(i,j) \in \mathcal{P}} \sum_{(i,k) \in \mathcal{N}} e^{\gamma(s_{ik} - s_{ij})}\right)$$

**简化**（CoSENT）：
$$\mathcal{L} = \log\left(1 + \sum_{i<j} e^{\lambda(s_{ij} - y_{ij})}\right)$$
其中 $y_{ij} = 1$ (相似) or $-1$ (不相似)

---

## 2. GlobalPointer

### 2.1 核心思想

将实体识别建模为**头尾配对**：
$$s(i, j) = \boldsymbol{q}_i^T \boldsymbol{k}_j$$

其中 $i$ 是起始位置，$j$ 是结束位置。

### 2.2 GlobalPointer损失

**多标签分类**（实体可重叠）：
$$\mathcal{L} = -\frac{1}{N} \sum_{c=1}^C \left[\sum_{(i,j) \in \mathcal{P}_c} \log \sigma(s_c(i,j)) + \sum_{(i,j) \notin \mathcal{P}_c} \log(1 - \sigma(s_c(i,j)))\right]$$

**改进**：Efficient GlobalPointer
- 减少头数（单头）
- 低秩分解：$\boldsymbol{q}_i = \boldsymbol{W}_1 \boldsymbol{h}_i$, $\boldsymbol{k}_j = \boldsymbol{W}_2 \boldsymbol{h}_j$

---

## 3. 多任务损失

### 3.1 朴素加权
$$\mathcal{L} = \sum_{i=1}^T \lambda_i \mathcal{L}_i$$

**缺陷**：
- ❌ $\lambda_i$ 难以调参
- ❌ 任务尺度不一致（损失值差异大）

### 3.2 不确定性加权

**核心思想**（Kendall & Gal, 2018）：
$$\mathcal{L} = \sum_i \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$$

其中 $\sigma_i$ 是可学习的任务不确定性。

**推导**：贝叶斯多任务学习的最大似然

### 3.3 梯度归一化

**GradNorm**：
$$\mathcal{L}_{grad} = \sum_i \left| \|\nabla_{\boldsymbol{W}} \mathcal{L}_i\| - \bar{G} \cdot r_i^{\alpha} \right|$$
其中 $r_i(t) = \mathcal{L}_i(t) / \mathcal{L}_i(0)$ 是相对损失下降率。

---

## 4. EMO最优传输损失

### 4.1 核心思想

将分类问题建模为**概率分布匹配**。

**Wasserstein距离**：
$$W_p(\mu, \nu) = \left(\inf_{\gamma \in \Gamma(\mu, \nu)} \int d(x, y)^p d\gamma(x, y)\right)^{1/p}$$

**EMO简化**（Sinkhorn迭代）：
$$\mathcal{L}_{EMO} = \langle \boldsymbol{C}, \boldsymbol{T} \rangle - \epsilon H(\boldsymbol{T})$$
其中 $\boldsymbol{T}$ 是最优传输矩阵，$\boldsymbol{C}$ 是代价矩阵。

---

## 5. 未来方向

**方向1：自适应损失**
- 根据训练阶段动态调整损失权重
- 强化学习选择损失函数

**方向2：元学习损失**
- 学习数据特定的损失函数
- Neural Loss Functions

**方向3：对抗性损失**
- 结合生成对抗思想
- Focal Loss的自适应版本

---

**撰写日期**：2025-11-18
**版本**：v1.0
