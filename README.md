# 📚 机器学习与数学博客精选 | ML & Math Blog Posts

精选来自[科学空间](https://spaces.ac.cn)的234篇机器学习与数学文章，配备完整的技术基础设施，支持详细的数学推导、交互式可视化和高质量的内容创作。

🌐 **在线访问**: [GitHub Pages](https://your-username.github.io/ml_posts/) _(配置后替换链接)_

---

## ✨ 核心特性

### 📖 内容展示
- ✅ **234篇精选文章**，按日期排序并自动编号
- ✅ **实时搜索**：标题、标签、内容全文搜索
- ✅ **标签筛选**：多标签组合筛选，可点击标签快速跳转
- ✅ **分页浏览**：20篇/页，共12页，智能页码导航
- ✅ **响应式设计**：完美适配桌面、平板和手机

### 🎨 样式系统
- ✅ **6种专用内容框**：推导框、定理框、证明框、例子框、注释框、直觉理解框
- ✅ **公式逐行解释**：详细解释复杂公式的每一步
- ✅ **自动编号分步推导**：步骤清晰，层次分明
- ✅ **可折叠区域**：长推导可折叠，不影响阅读体验

### 📊 交互式可视化
- ✅ **概率分布可视化**：7种分布，可调参数，实时采样
- ✅ **矩阵可视化**：SVD分解、特征值分布、随机矩阵谱
- ✅ **优化算法对比**：6种优化器在不同损失函数上的表现

### 🔧 开发工具
- ✅ **增量构建**：只重新生成修改过的文件
- ✅ **干运行模式**：预览生成结果
- ✅ **自动TOC生成**：文章目录自动生成并侧边栏显示
- ✅ **文章导航**：上一篇/下一篇，面包屑导航
- ✅ **标签状态追踪**：记录哪些文章的标签已经精心设计

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

**依赖项**:
- `feedparser>=6.0.0` - RSS抓取
- `markdown>=3.4.0` - Markdown转HTML
- `jinja2>=3.1.0` - 模板引擎
- `beautifulsoup4>=4.9.0` - HTML解析

### 生成HTML页面

```bash
# 完整重新生成所有234篇文章
python3 scripts/generate_posts.py

# 增量模式（只处理修改过的文件）- 推荐！
python3 scripts/generate_posts.py --incremental

# 预览模式（不实际写入文件）
python3 scripts/generate_posts.py --dry-run

# 生成单个文件
python3 scripts/generate_posts.py blogs_raw/example.md
```

### 本地预览

```bash
cd docs
python3 -m http.server 8000
```

访问 http://localhost:8000

---

## 📂 项目结构

```
ml_posts/
├── docs/                      # GitHub Pages根目录
│   ├── index.html            # 搜索首页（带分页和标签筛选）
│   ├── posts/                # 234个HTML文章页面
│   ├── assets/
│   │   ├── css/
│   │   │   ├── main.css      # 主页样式
│   │   │   └── post.css      # 文章样式（含6种内容框）
│   │   ├── js/
│   │   │   └── collapsible.js # 可折叠功能
│   │   └── charts/           # 交互式图表组件
│   │       ├── distribution_plot.html     # 概率分布
│   │       ├── matrix_visualization.html  # 矩阵可视化
│   │       └── optimization_viz.html      # 优化算法
│   └── data/
│       └── blog_list.json    # 文章元数据（含序号、标签等）
│
├── blogs_raw/                # 234个Markdown源文件
├── scripts/
│   ├── fetch_and_filter.py   # RSS抓取脚本
│   ├── fetch_all_blogs.py    # 批量抓取脚本
│   └── generate_posts.py     # Markdown→HTML生成器（458行）
│
├── templates/
│   └── post_template.html    # Jinja2文章模板
│
├── CONTRIBUTING.md           # 内容编辑工作流指南
├── CHART_GUIDE.md            # 图表嵌入指南
├── PROGRESS_REPORT.md        # 项目进度报告
└── requirements.txt
```

---

## ✍️ 编辑文章

### 基础工作流

```bash
# 1. 编辑Markdown文件
vim blogs_raw/你的文章.md

# 2. 重新生成HTML
python3 scripts/generate_posts.py --incremental

# 3. 本地预览
cd docs && python3 -m http.server 8000
```

### Markdown文件格式

```markdown
---
title: 随机矩阵的谱范数的快速估计
slug: 随机矩阵的谱范数的快速估计
date: 2025-11-06
source: https://spaces.ac.cn/archives/11335
tags: 随机矩阵, 数学, 深度学习
status: completed
tags_reviewed: true
---

# 文章标题

[原文内容...]

---

## 公式推导与注释

### 定理：Marchenko-Pastur

<div class="theorem-box">

对于$m \times n$高斯随机矩阵$\boldsymbol{W}$，当$m, n \to \infty$时：

$$
\|\boldsymbol{W}\|_2 \approx \sqrt{m} + \sqrt{n}
$$

</div>

<div class="derivation-box">

### 推导过程

<div class="step-by-step">
<div class="step">
从定义出发，谱范数等于最大奇异值...
</div>
<div class="step">
应用随机矩阵理论...
</div>
</div>

</div>

### 交互式验证

<iframe src="../assets/charts/matrix_visualization.html"
        width="100%"
        height="700"
        frameborder="0">
</iframe>
```

---

## 🎨 可用样式组件

### 内容框

| 类名 | 颜色 | 用途 | 示例 |
|------|------|------|------|
| `.derivation-box` | 蓝色 | 数学推导 | 完整的公式推导过程 |
| `.theorem-box` | 绿色 | 定理陈述 | 📐 正式的数学定理 |
| `.proof-box` | 紫色 | 证明 | ✓ 定理的证明过程 |
| `.example-box` | 橙色 | 例子 | 💡 具体的数值计算 |
| `.intuition-box` | 青色 | 直觉解释 | 🧠 非正式的理解 |
| `.note-box` | 黄色 | 注释 | 重要提示和警告 |

### 使用示例

```html
<div class="derivation-box">
  <h3>推导标题</h3>
  推导内容...
</div>
```

**完整文档**: 参见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📊 交互式图表

### 概率分布可视化

**文件**: `docs/assets/charts/distribution_plot.html`

**功能**:
- 7种分布：正态、t、卡方、指数、均匀、Beta、Gamma
- 可调参数（均值、方差、自由度等）
- 实时采样和统计对比

### 矩阵可视化

**文件**: `docs/assets/charts/matrix_visualization.html`

**功能**:
- 矩阵热图
- SVD分解演示
- 特征值分布
- 随机矩阵谱范数验证

### 优化算法对比

**文件**: `docs/assets/charts/optimization_viz.html`

**功能**:
- 6种优化器：GD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
- 4种损失函数：Bowl, Rosenbrock, Beale, Rastrigin
- 实时轨迹和收敛曲线

### 嵌入方法

```html
<iframe src="../assets/charts/distribution_plot.html"
        width="100%"
        height="800"
        frameborder="0">
</iframe>
```

**完整指南**: 参见 [CHART_GUIDE.md](CHART_GUIDE.md)

---

## 📋 标签系统

### 推荐标签分类

**主题标签** (每篇1-2个):
- 机器学习、深度学习、数学、统计学、优化、概率论

**技术标签** (每篇0-3个):
- Transformer、注意力机制、扩散模型、Adam、SGD

**理论标签** (每篇0-2个):
- 渐近分析、大偏差理论、信息论、凸优化

### 标记已审核标签

```markdown
---
tags: 优化, 机器学习, Adam, 收敛性分析
tags_reviewed: true  # 表示标签已精心设计
---
```

---

## 🌐 部署到GitHub Pages

### 1. 推送代码

```bash
git add .
git commit -m "Update blog posts"
git push origin main
```

### 2. 配置GitHub Pages

1. 进入仓库 Settings → Pages
2. Source: Deploy from a branch
3. Branch: `main`
4. Folder: `/docs`
5. 点击 Save

### 3. 访问网站

几分钟后访问：`https://your-username.github.io/ml_posts/`

---

## 🔧 高级功能

### 增量构建

只重新生成修改过的文件，大幅提升速度：

```bash
python3 scripts/generate_posts.py --incremental
```

**工作原理**:
- 计算每个Markdown文件的MD5哈希
- 与缓存（`.build_cache.json`）对比
- 只处理变化的文件

### 文章序号

文章按发布日期排序，自动分配序号：
- 最旧的文章：#1
- 最新的文章：#234

序号显示在：
- 文章标题前的徽章
- 面包屑导航
- 上一篇/下一篇导航

### TOC侧边栏

自动从Markdown标题生成目录：
- Sticky定位，滚动时保持可见
- 支持3级标题
- 带permalink锚点

### 可折叠长推导

对于复杂的推导过程，使用可折叠区域：

```html
<details>
<summary>点击展开详细推导</summary>
<div markdown="1">
长推导内容...
</div>
</details>
```

或使用JavaScript驱动的自定义组件。

---

## 📊 项目统计

| 指标 | 数值 |
|------|------|
| 文章总数 | 234篇 |
| Markdown源文件 | 3.8MB |
| HTML页面 | 5.5MB |
| 分页数 | 12页 (20篇/页) |
| 详细推导完成 | 2篇 (示例) |
| 交互式图表 | 3个 |
| CSS样式 | >500行 |
| Python生成器 | 458行 |

---

## 📚 文档索引

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - 内容编辑完整指南
  - Markdown写作规范
  - 数学公式LaTeX指南
  - 样式框使用说明
  - 标签规范
  - 提交流程

- **[CHART_GUIDE.md](CHART_GUIDE.md)** - 交互式图表嵌入指南
  - 3种图表组件详细说明
  - 嵌入方法和最佳实践
  - 样式定制技巧
  - 性能优化建议

- **[PROGRESS_REPORT.md](PROGRESS_REPORT.md)** - 项目进度报告
  - 完成功能列表
  - 技术细节说明
  - 统计信息

---

## 🛠️ 技术栈

**前端**:
- HTML5, CSS3
- Bootstrap 5.1.3
- JavaScript (Vanilla)
- Font Awesome 5.15.4
- Plotly.js 2.27.0

**数学与代码**:
- MathJax 3 (LaTeX公式渲染)
- Highlight.js 11.7.0 (代码高亮)

**构建工具**:
- Python 3.8+
- Python-Markdown 3.4+ (含11个扩展)
- Jinja2 3.1+ (模板引擎)
- BeautifulSoup4 (HTML解析)

**部署**:
- GitHub Pages
- 纯静态，无需后端

---

## 🤝 贡献指南

欢迎贡献！可以：

1. **添加数学推导**：为现有文章补充详细推导
2. **创建可视化**：开发新的交互式图表
3. **改进样式**：优化CSS样式和布局
4. **报告问题**：提交Issue
5. **提交PR**：Pull Request

**详细流程**: 参见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📜 许可证

- **内容**: 来源于[科学空间](https://spaces.ac.cn)，版权归原作者所有
- **代码**: MIT License

---

## 🙏 致谢

- **博客作者**: [苏剑林](https://spaces.ac.cn) - 所有文章原作者
- **模板灵感**: [ICLR Blogposts 2025](https://github.com/iclr-blogposts/2025)
- **数学渲染**: [MathJax](https://www.mathjax.org/)
- **图表库**: [Plotly.js](https://plotly.com/javascript/)

---

## 📞 联系方式

- **Issues**: [GitHub Issues](https://github.com/your-username/ml_posts/issues)
- **讨论**: [GitHub Discussions](https://github.com/your-username/ml_posts/discussions)

---

**最后更新**: 2025-11-10
