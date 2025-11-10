# 📊 图表嵌入指南

本指南介绍如何在Markdown文章中嵌入交互式Plotly图表。

---

## 可用图表组件

### 1. 概率分布可视化
**文件**: `docs/assets/charts/distribution_plot.html`

**功能**:
- 7种概率分布：正态分布、t分布、卡方分布、指数分布、均匀分布、Beta分布、Gamma分布
- 可调参数（均值、方差、自由度等）
- 采样数量调节（100-10000）
- 采样直方图对比
- 实时统计信息（理论均值/方差、采样均值/方差）

**适用场景**:
- 解释随机变量的分布
- 对比不同参数下的分布形状
- 验证中心极限定理
- 展示最大值/最小值的分布

### 2. 矩阵可视化
**文件**: `docs/assets/charts/matrix_visualization.html`

**功能**:
- 热图显示矩阵元素
- SVD分解可视化（奇异值分布）
- 特征值分布和复平面图
- 随机矩阵谱范数验证
- 多种矩阵类型：随机、高斯、低秩、对称、相关矩阵
- 可调矩阵大小（3×3 到 50×50）

**适用场景**:
- 展示矩阵结构
- SVD分解演示
- 随机矩阵理论验证（Marchenko-Pastur定理）
- 特征值分析

### 3. 优化算法可视化
**文件**: `docs/assets/charts/optimization_viz.html`

**功能**:
- 6种优化器：GD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
- 4种损失函数：Bowl, Rosenbrock, Beale, Rastrigin
- 可调学习率和迭代次数
- 实时优化轨迹
- 损失收敛曲线
- 多优化器对比

**适用场景**:
- 对比不同优化算法
- 解释Momentum、Adam等算法原理
- 展示鞍点、局部最小值问题
- 学习率调优演示

---

## 嵌入方法

### 方法1：iframe嵌入（推荐）

在Markdown中使用HTML的iframe标签：

```markdown
## 概率分布演示

<iframe src="../assets/charts/distribution_plot.html"
        width="100%"
        height="800"
        frameborder="0"
        style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
</iframe>
```

**参数说明**:
- `src`: 相对路径指向图表HTML文件
- `width`: 建议使用100%以适应响应式布局
- `height`: 根据图表内容调整（推荐600-1000px）
- `frameborder`: 设为0去除边框
- `style`: 可选的CSS样式

### 方法2：直接链接

提供链接让用户在新标签页打开：

```markdown
[🔗 打开交互式概率分布可视化](../assets/charts/distribution_plot.html){:target="_blank"}
```

### 方法3：内联HTML（完整控制）

对于需要自定义样式的场景：

```markdown
<div class="interactive-chart">
    <h3>🎯 优化算法对比</h3>
    <p>下面的交互式图表展示了6种优化算法在不同损失函数上的表现。</p>

    <iframe src="../assets/charts/optimization_viz.html"
            width="100%"
            height="1000"
            frameborder="0"
            loading="lazy"
            style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: 20px 0;">
    </iframe>

    <div class="note-box">
        <strong>提示：</strong>
        点击上方的算法按钮可以选择要对比的优化器。尝试调整学习率观察不同的收敛行为。
    </div>
</div>
```

---

## 完整示例

### 示例1：在推导中嵌入分布图

```markdown
## 正态分布的性质

### 定义

正态分布的概率密度函数为：

$$
f(x; \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

其中 $\mu$ 是均值，$\sigma^2$ 是方差。

### 交互式演示

下面的可视化工具允许你调整参数并观察分布形状的变化：

<iframe src="../assets/charts/distribution_plot.html"
        width="100%"
        height="800"
        frameborder="0">
</iframe>

<div class="intuition-box">
🧠 <strong>直觉理解：</strong>

- 增大 $\sigma$ 会使分布更"扁平"，数据更分散
- 改变 $\mu$ 会左右平移整个分布
- 勾选"显示采样直方图"可以验证理论分布与实际采样的一致性
</div>
```

### 示例2：SVD分解演示

```markdown
## 奇异值分解 (SVD)

对于任意 $m \times n$ 矩阵 $\boldsymbol{W}$，存在分解：

$$
\boldsymbol{W} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T
$$

### 可视化工具

<iframe src="../assets/charts/matrix_visualization.html"
        width="100%"
        height="900"
        frameborder="0">
</iframe>

<div class="example-box">
💡 <strong>例子：</strong>

尝试以下操作：
1. 将矩阵类型切换为"低秩矩阵"
2. 选择"SVD分解"可视化
3. 观察奇异值图：前几个奇异值显著大于其余的

这正是低秩矩阵的特征！
</div>
```

### 示例3：优化算法对比

```markdown
## Adam优化器

### 算法描述

Adam结合了Momentum和RMSProp的优点：

<div class="step-by-step">
    <div class="step">
        <strong>一阶矩估计：</strong>
        $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
    </div>

    <div class="step">
        <strong>二阶矩估计：</strong>
        $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
    </div>

    <div class="step">
        <strong>偏差修正：</strong>
        $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
    </div>

    <div class="step">
        <strong>参数更新：</strong>
        $$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
    </div>
</div>

### 与其他优化器对比

<iframe src="../assets/charts/optimization_viz.html"
        width="100%"
        height="1000"
        frameborder="0">
</iframe>

<details>
    <summary><strong>💡 实验建议</strong></summary>
    <div markdown="1">

1. **尝试Rosenbrock函数**：
   - 这是一个经典的优化难题
   - 观察Adam和Momentum的收敛速度差异

2. **调整学习率**：
   - 学习率过大：轨迹震荡
   - 学习率过小：收敛缓慢
   - Adam对学习率不敏感

3. **对比收敛曲线**：
   - 查看下方的损失曲线图
   - 注意纵轴是对数刻度

    </div>
</details>
```

---

## 样式定制

### 容器样式

为图表容器添加自定义样式：

```html
<div class="interactive-chart" style="
    background: linear-gradient(to bottom, #f8f9fa, #ffffff);
    padding: 30px;
    border-radius: 12px;
    margin: 40px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
">
    <iframe src="../assets/charts/distribution_plot.html"
            width="100%"
            height="800"
            frameborder="0">
    </iframe>
</div>
```

### 配合现有样式框

结合项目的样式类：

```html
<div class="derivation-box">
    <h3>随机矩阵谱范数的数值验证</h3>

    <p>根据Marchenko-Pastur定理，当 $m, n \to \infty$ 时：</p>

    $$
    \|\boldsymbol{W}\|_2 \approx \sqrt{m} + \sqrt{n}
    $$

    <iframe src="../assets/charts/matrix_visualization.html"
            width="100%"
            height="700"
            frameborder="0"
            onload="this.contentWindow.document.getElementById('vizType').value='random_matrix'; this.contentWindow.updateVisualization();">
    </iframe>
</div>
```

---

## 最佳实践

### ✅ 推荐做法

1. **提供上下文**：在图表前解释背景和目的
   ```markdown
   为了直观理解参数对分布的影响，我们提供了以下交互式工具...
   ```

2. **引导交互**：告诉用户可以尝试什么
   ```markdown
   <div class="note-box">
   💡 <strong>试试看：</strong>
   - 将自由度设为1，观察t分布的"重尾"特性
   - 增大采样数量到10000，验证Law of Large Numbers
   </div>
   ```

3. **总结观察**：在图表后总结关键发现
   ```markdown
   从上面的可视化可以看出，随着自由度增大，t分布逐渐接近正态分布。
   ```

4. **设置合理高度**：
   - 简单图表：600-700px
   - 多子图表：800-1000px
   - 复杂交互：1000-1200px

5. **使用lazy loading**：
   ```html
   <iframe src="..." loading="lazy"></iframe>
   ```

### ❌ 避免的做法

1. ❌ 不要无说明直接嵌入图表
2. ❌ 不要设置过小的高度导致滚动条
3. ❌ 不要在同一页面嵌入过多图表（建议≤3个）
4. ❌ 不要忘记添加fallback提示：
   ```html
   <iframe src="...">
       您的浏览器不支持iframe。请<a href="...">点击这里</a>在新窗口打开图表。
   </iframe>
   ```

---

## 性能优化

### 延迟加载

对于文章中靠后的图表，使用延迟加载：

```html
<iframe src="../assets/charts/optimization_viz.html"
        width="100%"
        height="1000"
        frameborder="0"
        loading="lazy">
</iframe>
```

### 条件加载

对于可选的高级内容：

```html
<details>
    <summary><strong>🔍 查看交互式演示</strong></summary>
    <div>
        <iframe src="../assets/charts/matrix_visualization.html"
                width="100%"
                height="900"
                frameborder="0"
                loading="lazy">
        </iframe>
    </div>
</details>
```

---

## 故障排查

### 图表不显示

1. **检查路径**：确保相对路径正确
   - 从 `docs/posts/example.html` 引用图表：使用 `../assets/charts/`
   - 从 `blogs_raw/example.md` 编写时同样使用相对路径

2. **检查高度**：`height` 必须设置明确的像素值或百分比

3. **检查浏览器控制台**：按F12查看是否有加载错误

### 布局问题

1. **响应式**：始终使用 `width="100%"`
2. **移动端**：测试在小屏幕上的显示效果
3. **嵌套**：避免在iframe中再嵌套iframe

---

## 扩展开发

### 创建自定义图表

如果需要为特定文章创建定制图表：

1. 复制现有模板（如 `distribution_plot.html`）
2. 修改JavaScript逻辑和数学函数
3. 保存到 `docs/assets/charts/custom_chart_name.html`
4. 在文章中引用

### 参数预设

在URL中传递参数（未来功能）：

```html
<iframe src="../assets/charts/distribution_plot.html?dist=normal&mu=0&sigma=1" ...>
</iframe>
```

---

## 总结

**核心原则**：
- 📊 图表应增强理解，而非炫技
- 📝 始终提供文字解释作为补充
- 🎯 引导用户进行有意义的交互
- 🚀 注意性能，避免过度嵌入

**快速开始**：
```markdown
<iframe src="../assets/charts/distribution_plot.html"
        width="100%"
        height="800"
        frameborder="0">
</iframe>
```

现在你可以为你的234篇数学博客添加丰富的交互式可视化了！
