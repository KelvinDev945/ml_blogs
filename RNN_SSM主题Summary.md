# RNN/SSM主题深度Summary

> **涵盖文章**：6篇RNN/SSM相关文章
> **主要内容**：RNN梯度问题、SSM理论、S4、Mamba

---

## 1. RNN梯度消失/爆炸

### 1.1 梯度传播分析

**标准RNN**：
$$\boldsymbol{h}_t = \tanh(\boldsymbol{W}_h \boldsymbol{h}_{t-1} + \boldsymbol{W}_x \boldsymbol{x}_t)$$

**梯度链式法则**：
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_0} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_T} \prod_{t=1}^{T} \frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_{t-1}}$$

**雅可比矩阵**：
$$\boldsymbol{J}_t = \text{diag}(\tanh'(\cdot)) \boldsymbol{W}_h$$

**谱半径分析**：
$$\left\|\prod_{t=1}^{T} \boldsymbol{J}_t\right\| \approx \rho(\boldsymbol{W}_h)^T$$

- $\rho < 1$：梯度消失（$\to 0$）
- $\rho > 1$：梯度爆炸（$\to \infty$）

---

## 2. SSM核心理论

### 2.1 状态空间方程

**连续时间**：
$$\frac{d\boldsymbol{x}(t)}{dt} = \boldsymbol{A}\boldsymbol{x}(t) + \boldsymbol{B}u(t)$$
$$y(t) = \boldsymbol{C}\boldsymbol{x}(t) + Du(t)$$

**离散化**（Zero-Order Hold）：
$$\boldsymbol{x}_k = \bar{\boldsymbol{A}} \boldsymbol{x}_{k-1} + \bar{\boldsymbol{B}} u_k$$
$$y_k = \boldsymbol{C} \boldsymbol{x}_k$$

其中：
$$\bar{\boldsymbol{A}} = e^{\Delta \boldsymbol{A}}, \quad \bar{\boldsymbol{B}} = (\bar{\boldsymbol{A}} - \boldsymbol{I})\boldsymbol{A}^{-1}\boldsymbol{B}$$

### 2.2 卷积视角

**频域**：
$$\hat{y}(\omega) = \hat{\boldsymbol{K}}(\omega) \hat{u}(\omega)$$

其中卷积核：
$$\boldsymbol{K}(t) = \boldsymbol{C}e^{t\boldsymbol{A}}\boldsymbol{B}$$

**时域卷积**：
$$y_k = \sum_{j=0}^{k} \boldsymbol{K}_j u_{k-j}$$

---

## 3. S4高效计算

### 3.1 HiPPO矩阵的DPLR结构

**分解**：
$$\boldsymbol{A} = \boldsymbol{D} - \boldsymbol{P}\boldsymbol{L}\boldsymbol{P}^T$$

- $\boldsymbol{D}$：对角矩阵
- $\boldsymbol{P}, \boldsymbol{L}$：低秩

### 3.2 Cauchy核技巧

**核矩阵**：
$$\boldsymbol{K} = (\boldsymbol{\omega}_i - \lambda_j)^{-1}$$

**快速计算**：FFT + 多项式求值，复杂度 $O(N \log N)$

---

## 4. Mamba选择性SSM

### 4.1 核心创新

**输入依赖参数**：
$$\boldsymbol{B}_t = \text{Linear}_B(\boldsymbol{x}_t), \quad \boldsymbol{C}_t = \text{Linear}_C(\boldsymbol{x}_t)$$

**效果**：
- 重要token → 大$\boldsymbol{B}_t$ → 强影响
- 噪声token → 小$\boldsymbol{B}_t$ → 弱影响

### 4.2 硬件感知实现

**Scan操作融合**：
```cuda
// 传统：分离kernel
for (int i = 0; i < N; i++) {
    x[i] = A * x[i-1] + B * u[i];
}

// Mamba：融合kernel（减少HBM访问）
__global__ void selective_scan_fused(...) {
    // 单kernel完成所有操作
}
```

---

## 5. 为什么线性注意力需要Short Conv？

### 5.1 线性注意力局限

**标准Attention**：
$$\boldsymbol{O} = \text{softmax}(\boldsymbol{QK}^T)\boldsymbol{V}$$

**线性化**（核技巧）：
$$\boldsymbol{O} = \phi(\boldsymbol{Q})(\phi(\boldsymbol{K})^T\boldsymbol{V})$$

**问题**：局部性丧失（无位置偏好）

### 5.2 Short Conv补偿

**添加1D卷积**（窗口=3-7）：
$$\boldsymbol{x}'_t = \sum_{i=-w}^{w} \boldsymbol{W}_i \boldsymbol{x}_{t+i}$$

**作用**：
- 捕捉局部模式
- 引入归纳偏置
- 性能提升2-5%

---

## 6. 未来方向

**方向1：非线性SSM**
- 当前：线性系统限制表达力
- 探索：门控SSM、二次SSM
- 挑战：保持 $O(N)$ 复杂度

**方向2：长序列SSM**
- 目标：100万+ tokens
- 方法：分层SSM、记忆压缩

**方向3：多模态SSM**
- 图像、视频的SSM建模
- 2D/3D状态空间

---

**撰写日期**：2025-11-18
**版本**：v1.0
