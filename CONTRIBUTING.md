# 📝 内容编辑指南 / Contributing Guide

本文档指导如何为博客文章添加详细的数学推导、可视化内容和注释。

---

## 目录

1. [工作流程](#工作流程)
2. [Markdown编写规范](#markdown编写规范)
3. [数学公式指南](#数学公式指南)
4. [样式框使用](#样式框使用)
5. [图表嵌入](#图表嵌入)
6. [标签规范](#标签规范)
7. [提交流程](#提交流程)

---

## 工作流程

### 编辑单篇文章

```bash
# 1. 选择要编辑的文章
cd blogs_raw/
ls *.md  # 查看所有文章

# 2. 编辑Markdown文件
vim n个正态随机数的最大值的渐近估计.md

# 3. 重新生成HTML（单个文件）
python3 ../scripts/generate_posts.py blogs_raw/n个正态随机数的最大值的渐近估计.md

# 4. 本地预览
cd ../docs
python3 -m http.server 8000
# 访问 http://localhost:8000
```

### 批量更新

```bash
# 使用增量模式只重新生成修改过的文件
python3 scripts/generate_posts.py --incremental

# 完整重新生成所有文章
python3 scripts/generate_posts.py
```

---

## Markdown编写规范

### 文件结构

每个Markdown文件应包含以下部分：

```markdown
---
title: 文章标题
slug: 文章-url-slug
date: 2025-11-06
source: https://spaces.ac.cn/archives/xxxxx
tags: 机器学习, 数学, 优化
status: pending  # 或 completed
tags_reviewed: false  # 标签是否已精心设计
---

# 文章标题

[原文内容保持不变...]

---

## 公式推导与注释

[在这里添加详细推导...]
```

### Frontmatter字段说明

- `title`: **必需**，文章标题
- `slug`: **必需**，URL友好的文件名
- `date`: **必需**，发布日期（YYYY-MM-DD）
- `source`: 原文链接
- `tags`: 逗号分隔的标签列表
- `status`: `pending`（待完成）或 `completed`（已完成）
- `tags_reviewed`: `true`（标签已精心设计）或 `false`（使用默认标签）

### 标题层级

```markdown
# H1 - 仅用于文章主标题（自动生成）

## H2 - 主要章节
### H3 - 次级章节
#### H4 - 小节

不要使用 H5、H6
```

---

## 数学公式指南

### 行内公式

使用单个美元符号 `$...$`：

```markdown
这里是行内公式 $f(x) = x^2 + 1$，它会与文字在同一行。
```

### 块级公式

使用双美元符号 `$$...$$`：

```markdown
$$
\mathbb{E}[\boldsymbol{W}] = \boldsymbol{\mu}
$$
```

### 多行公式

使用 `align` 环境：

```markdown
$$
\begin{align}
f(x) &= (x-1)^2 \\
     &= x^2 - 2x + 1 \\
     &= x^2 - 2x + 1
\end{align}
$$
```

### 常用符号

| 类型 | LaTeX | 渲染 |
|------|-------|------|
| 加粗向量 | `\boldsymbol{W}` | $\boldsymbol{W}$ |
| 期望 | `\mathbb{E}[X]` | $\mathbb{E}[X]$ |
| 范数 | `\|\boldsymbol{x}\|_2` | $\|\boldsymbol{x}\|_2$ |
| 约等于 | `\approx` | $\approx$ |
| 箭头 | `\rightarrow` | $\rightarrow$ |
| 分数 | `\frac{a}{b}` | $\frac{a}{b}$ |
| 求和 | `\sum_{i=1}^{n}` | $\sum_{i=1}^{n}$ |
| 积分 | `\int_{0}^{1}` | $\int_{0}^{1}$ |

### 公式编号

Markdown扩展会自动添加编号。如需引用：

```markdown
这是一个重要公式：

$$
E = mc^2
$$ {#eq:einstein}

后续可以引用公式 \eqref{eq:einstein}。
```

---

## 样式框使用

### 推导框（蓝色）

用于完整的数学推导过程：

```markdown
<div class="derivation-box">

### 推导：最大值的期望估计

从CDF开始推导：

$$
P(\max_i z_i \leq x) = P(z_1 \leq x, z_2 \leq x, \ldots, z_n \leq x)
$$

由于独立性：

$$
= \prod_{i=1}^{n} P(z_i \leq x) = [\Phi(x)]^n
$$

其中 $\Phi(x)$ 是标准正态分布的CDF。

</div>
```

### 定理框（绿色）

用于正式的数学定理：

```markdown
<div class="theorem-box">

### 定理：Marchenko-Pastur定理

设 $\boldsymbol{W}$ 为 $m \times n$ 高斯随机矩阵，当 $m, n \to \infty$ 且 $m/n \to \gamma$ 时，
最大奇异值几乎必然收敛到：

$$
\sigma_{\max}(\boldsymbol{W}) \to \sqrt{m} + \sqrt{n}
$$

</div>
```

### 证明框（紫色）

用于定理的证明：

```markdown
<div class="proof-box">

我们通过以下步骤证明：

**步骤1**：首先注意到...

$$
\boldsymbol{W}^T \boldsymbol{W} = \sum_{i=1}^{m} \boldsymbol{w}_i \boldsymbol{w}_i^T
$$

**步骤2**：应用随机矩阵理论...

**步骤3**：因此得证。 $\square$

</div>
```

### 例子框（橙色）

用于具体的计算示例：

```markdown
<div class="example-box">

### 例子：$n=1000$ 时的数值计算

当 $n = 1000$ 时，理论估计为：

$$
\mathbb{E}[z_{\max}] \approx \sqrt{2 \log 1000} \approx 3.72
$$

通过10000次蒙特卡洛模拟，得到经验平均值 3.71，误差仅为0.27%。

</div>
```

### 直觉理解框（青色）

用于非正式的直观解释：

```markdown
<div class="intuition-box">

### 🧠 直觉理解

为什么最大值的期望是 $\sqrt{2\log n}$ 量级？

**信息论视角**：
- 要使某个值成为最大值，它必须"击败"其他 $n-1$ 个值
- 这需要该值落在分布的极端尾部
- 尾部概率呈指数衰减，因此需要 $\log n$ 量级的偏离

</div>
```

### 注释框（黄色）

用于重要提示和注意事项：

```markdown
<div class="note-box">

**注意**：以上推导假设样本独立同分布。如果存在相关性，结论会有所不同。

</div>
```

### 公式逐行解释

用于详细解释复杂公式的每一步：

```markdown
<div class="formula-explanation">

<div class="formula-step">
<div class="step-label">步骤1：应用链式法则</div>

$$
\frac{\partial L}{\partial \boldsymbol{W}} = \frac{\partial L}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{W}}
$$

<div class="step-explanation">
这里 $\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x}$ 是线性变换。
</div>
</div>

<div class="formula-step">
<div class="step-label">步骤2：计算梯度</div>

$$
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{W}} = \boldsymbol{x}^T
$$

<div class="step-explanation">
对矩阵求导，得到输入的转置。
</div>
</div>

</div>
```

### 分步推导（自动编号）

```markdown
<div class="step-by-step">

<div class="step">
从定义出发，设 $z_{\max} = \max_{i=1}^n z_i$。
</div>

<div class="step">
计算CDF：$P(z_{\max} \leq x) = [\Phi(x)]^n$
</div>

<div class="step">
对CDF求导得到PDF：$f(x) = n [\Phi(x)]^{n-1} \phi(x)$
</div>

<div class="step">
计算期望：$\mathbb{E}[z_{\max}] = \int_{-\infty}^{\infty} x f(x) \, dx$
</div>

</div>
```

---

## 可折叠区域

### 使用原生 `<details>` 标签（推荐）

```markdown
<details>
<summary>点击展开详细推导过程</summary>
<div markdown="1">

这里是可以折叠的内容。默认是折叠状态。

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

可以包含公式、列表、代码等任何Markdown内容。

</div>
</details>
```

**重要**：
- 必须添加 `<div markdown="1">` 使内部Markdown生效
- 适用于长推导、补充材料、高级内容

### 使用自定义可折叠区域

```markdown
<div class="collapsible-section">
    <div class="collapsible-header">
        <h4>完整的积分计算</h4>
        <span class="collapsible-icon">▼</span>
    </div>
    <div class="collapsible-content">
        <div class="collapsible-inner">

从分部积分开始：

$$
\int x e^{-x^2} dx = -\frac{1}{2} e^{-x^2} + C
$$

        </div>
    </div>
</div>
```

---

## 图表嵌入

详细参考 [CHART_GUIDE.md](CHART_GUIDE.md)。

### 快速嵌入

```markdown
### 交互式可视化

<iframe src="../assets/charts/distribution_plot.html"
        width="100%"
        height="800"
        frameborder="0"
        style="border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
</iframe>
```

### 配合说明文字

```markdown
<div class="derivation-box">

### 数值验证

理论预测 $\|\boldsymbol{W}\|_2 \approx \sqrt{m} + \sqrt{n}$。下面通过蒙特卡洛模拟验证：

<iframe src="../assets/charts/matrix_visualization.html"
        width="100%"
        height="700"
        frameborder="0">
</iframe>

可以看到，随着矩阵维度增大，经验分布确实收敛到理论值。

</div>
```

---

## 标签规范

### 推荐标签分类

**主题标签**（每篇文章1-2个）:
- 机器学习
- 深度学习
- 自然语言处理
- 计算机视觉
- 数学
- 统计学
- 优化
- 概率论
- 线性代数
- 随机矩阵

**技术标签**（每篇文章0-3个）:
- Transformer
- 注意力机制
- 扩散模型
- VAE
- GAN
- BERT
- GPT
- SGD
- Adam
- Momentum

**理论标签**（每篇文章0-2个）:
- 渐近分析
- 大偏差理论
- 信息论
- 测度论
- 泛函分析
- 凸优化

### 标签添加流程

1. **阅读文章**：理解核心内容
2. **选择主题标签**：1-2个主要领域
3. **添加技术标签**：涉及的具体技术/算法
4. **添加理论标签**：使用的数学工具
5. **标记已审核**：设置 `tags_reviewed: true`

### 示例

```markdown
---
title: Adam优化器的收敛性分析
slug: adam-convergence-analysis
date: 2025-11-10
tags: 优化, 机器学习, Adam, 收敛性分析, 随机优化
tags_reviewed: true
---
```

---

## 提交流程

### 1. 检查清单

在提交前确认：

- [ ] Frontmatter完整（title, slug, date, tags）
- [ ] 公式语法正确（LaTeX）
- [ ] 样式框使用恰当
- [ ] 图表嵌入（如适用）
- [ ] 标签已精心设计（`tags_reviewed: true`）
- [ ] 本地预览无误

### 2. 生成HTML

```bash
# 增量生成（推荐）
python3 scripts/generate_posts.py --incremental

# 或生成单个文件
python3 scripts/generate_posts.py blogs_raw/你的文章.md
```

### 3. 本地测试

```bash
cd docs
python3 -m http.server 8000
```

访问 http://localhost:8000 检查：
- 公式渲染正确
- 样式框显示正常
- 图表加载无误
- 目录导航有效
- 上一篇/下一篇链接正确

### 4. 提交到Git

```bash
git add blogs_raw/你的文章.md docs/posts/你的文章.html docs/data/blog_list.json
git commit -m "添加详细推导：你的文章标题

- 添加完整数学推导
- 嵌入交互式图表
- 更新标签"
git push
```

---

## 写作建议

### ✅ 最佳实践

1. **从简到难**：先直观解释，再严格证明
2. **公式与文字结合**：每个公式后都有文字说明
3. **使用直觉框**：帮助读者建立几何/物理直觉
4. **提供例子**：具体的数值计算示例
5. **分步推导**：复杂推导分成小步骤
6. **可视化辅助**：适当使用图表增强理解
7. **可折叠长推导**：避免吓到读者

### ❌ 常见错误

1. ❌ 跳过推导步骤
2. ❌ 公式没有解释
3. ❌ 滥用专业术语
4. ❌ 缺少直观理解
5. ❌ 没有具体例子
6. ❌ 推导过于冗长（应使用折叠）

### 写作模板

```markdown
## 问题定义

[清晰描述问题...]

<div class="theorem-box">
[正式的数学陈述...]
</div>

<div class="intuition-box">
🧠 **直觉理解**
[非正式的解释...]
</div>

<div class="derivation-box">
### 完整推导

<div class="step-by-step">
<div class="step">步骤1...</div>
<div class="step">步骤2...</div>
...
</div>

</div>

<div class="example-box">
### 数值例子
[具体计算...]
</div>

<iframe src="../assets/charts/...">
</iframe>

<details>
<summary>高级话题：理论扩展</summary>
<div markdown="1">
[可选的高级内容...]
</div>
</details>
```

---

## 常见问题

### Q: 公式渲染失败？

A: 检查：
1. `$` 符号是否配对
2. 特殊字符（`_`, `^`, `{`, `}`）是否正确转义
3. 使用 `$$...$$` 而非 `\[...\]`

### Q: 样式框内的Markdown不生效？

A: 在HTML块内添加 `<div markdown="1">`：

```markdown
<div class="derivation-box">
<div markdown="1">

这里的**加粗**和*斜体*会生效。

</div>
</div>
```

### Q: 图表显示空白？

A: 检查：
1. 相对路径是否正确（`../assets/charts/`）
2. iframe高度是否足够
3. 浏览器控制台是否有错误

---

## 资源链接

- [MathJax文档](https://docs.mathjax.org/)
- [Markdown扩展](https://python-markdown.github.io/extensions/)
- [图表嵌入指南](CHART_GUIDE.md)
- [Plotly文档](https://plotly.com/javascript/)

---

## 联系方式

遇到问题？
- 查看项目README
- 参考已完成的示例文章
- 提交Issue到GitHub仓库

---

**Happy Writing! 🚀**
