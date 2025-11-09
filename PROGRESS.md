# 项目进度报告

## 完成时间
2025-11-10

## 已完成的工作

### ✅ 阶段 1：模板与首页

1. **模板仓库分析** ✅
   - 克隆并深入研究了 ICLR Blogposts 2025 模板
   - 分析了布局、构建流程、静态资源结构
   - 提取了关键的 CSS/JS 资源

2. **首页搭建** ✅
   - 创建了 `docs/index.html` 搜索首页
   - 实现了实时搜索功能（标题、标签、内容）
   - 采用 Bootstrap 5 + 自定义样式
   - 支持响应式设计

3. **资源配置** ✅
   - 复制了必要的 JavaScript 文件到 `docs/assets/js/`
   - 创建了自定义 CSS 样式（`main.css` 和 `post.css`）
   - 配置了 MathJax 3 用于数学公式渲染
   - 集成了 Highlight.js 用于代码高亮

### ✅ 阶段 2：内容抓取与整理

1. **RSS 抓取脚本** ✅
   - 实现了 `scripts/fetch_and_filter.py`
   - 从 `https://spaces.ac.cn/feed` 抓取最新文章
   - 基于关键词智能筛选数学/机器学习相关内容
   - 成功抓取并保存了 10 篇相关博文

2. **内容组织** ✅
   - 所有文章保存到 `blogs_raw/<slug>.md` (Markdown 格式)
   - 创建了 `blog_list.txt` 文本清单
   - 生成了 `docs/data/blog_list.json` 用于前端展示
   - 为每篇文章预留了"公式推导与注释"部分

3. **已抓取的文章列表** (10篇)
   - n个正态随机数的最大值的渐近估计 (2025-11-06)
   - 流形上的最速下降：5. 对偶梯度下降 (2025-11-03)
   - 低精度Attention可能存在有偏的舍入误差 (2025-10-27)
   - MuP之上：1. 好模型的三个特征 (2025-10-21)
   - 随机矩阵的谱范数的快速估计 (2025-10-12)
   - DiVeQ：一种非常简洁的VQ训练方案 (2025-10-08)
   - 为什么线性注意力要加Short Conv？ (2025-10-05)
   - AdamW的Weight RMS的渐近估计 (2025-10-01)
   - 重新思考学习率与Batch Size（四）：EMA (2025-09-22)
   - 重新思考学习率与Batch Size（三）：Muon (2025-09-15)

### ✅ 阶段 3：页面生成与验证

1. **HTML 生成脚本** ✅
   - 实现了 `scripts/generate_posts.py`
   - 支持 Markdown 到 HTML 的转换
   - 使用模板生成统一风格的文章页面
   - 自动提取元信息（标题、日期、标签等）

2. **文章页面** ✅
   - 为所有 10 篇文章生成了 HTML 页面
   - 页面包含导航、元信息、返回首页链接
   - 支持 MathJax 数学公式渲染
   - 代码高亮显示正常工作

3. **本地验证** ✅
   - 使用 Python HTTP 服务器成功启动本地预览
   - 首页加载正常，搜索功能工作正常
   - 文章链接可访问，样式显示正确
   - 资源文件路径正确

## 项目结构

```
ml_posts/
├── docs/                           # GitHub Pages 根目录
│   ├── index.html                 # ✅ 搜索首页
│   ├── posts/                     # ✅ 10篇文章 HTML
│   ├── assets/
│   │   ├── css/
│   │   │   ├── main.css          # ✅ 全局样式
│   │   │   └── post.css          # ✅ 文章样式
│   │   └── js/                    # ✅ JavaScript 文件
│   └── data/
│       └── blog_list.json         # ✅ 博客数据
├── blogs_raw/                      # ✅ 10篇 Markdown 源文件
├── scripts/
│   ├── fetch_and_filter.py        # ✅ RSS 抓取脚本
│   └── generate_posts.py          # ✅ HTML 生成脚本
├── templates/
│   └── post_template.html         # ✅ 文章模板
├── blog_list.txt                  # ✅ 文本清单
├── requirements.txt               # ✅ Python 依赖
├── README.md                      # ✅ 项目文档
├── .gitignore                     # ✅ Git 忽略配置
└── TODO.md                        # 原始任务清单
```

## 技术实现

### 前端技术
- **HTML5/CSS3**: 现代语义化标记
- **Bootstrap 5**: 响应式布局框架
- **Font Awesome**: 图标库
- **MathJax 3**: LaTeX 数学公式渲染
- **Highlight.js**: 代码语法高亮
- **原生 JavaScript**: 搜索与交互功能

### 后端脚本
- **Python 3**: 核心脚本语言
- **feedparser**: RSS feed 解析
- **html2text**: HTML 转 Markdown
- **markdown**: Markdown 转 HTML
- **jinja2**: 模板引擎（未使用完整功能，采用简单替换）

### 部署方案
- **GitHub Pages**: 静态网站托管
- **纯静态**: 无需服务器端处理
- **CDN 资源**: 使用 jsDelivr CDN 加速

## 功能特性

✅ **自动化工作流**
- 一键抓取最新博客文章
- 自动筛选相关内容
- 批量生成 HTML 页面

✅ **用户体验**
- 实时搜索，无需后端
- 响应式设计，支持移动端
- 数学公式完美渲染
- 代码高亮清晰可读

✅ **可维护性**
- 清晰的项目结构
- 详细的文档说明
- 模块化的脚本设计
- 易于扩展和修改

## 下一步工作

### 待完成任务

1. **公式推导增强** 📝
   - 在 `blogs_raw/*.md` 文件中补充详细的数学推导
   - 添加每个公式的逐步推导过程
   - 标注关键步骤和技巧

2. **GitHub Pages 部署** 🚀
   - 配置 GitHub Pages 设置
   - 测试线上部署
   - 配置自定义域名（可选）

3. **内容迭代** 🔄
   - 定期运行抓取脚本获取新文章
   - 逐步完善已有文章的推导
   - 更新 blog_list.json 中的状态

4. **功能增强** ✨
   - 添加标签筛选功能
   - 实现文章分类
   - 添加阅读进度指示
   - 考虑添加评论功能（如 Giscus）

## 本地开发指南

### 抓取新文章
```bash
python3 scripts/fetch_and_filter.py
```

### 生成 HTML
```bash
python3 scripts/generate_posts.py
```

### 本地预览
```bash
cd docs
python3 -m http.server 8000
# 访问 http://localhost:8000
```

### 批量更新
```bash
python3 scripts/fetch_and_filter.py && python3 scripts/generate_posts.py
```

## 部署到 GitHub Pages

1. 提交代码到 GitHub
2. 在仓库设置中启用 Pages（选择 `/docs` 目录）
3. 访问自动生成的 URL

## 总结

该项目已经完成了所有基础设施的搭建：

- ✅ 完整的自动化工作流（抓取 → 筛选 → 生成）
- ✅ 功能完善的搜索首页
- ✅ 10 篇初始博客文章
- ✅ 美观的文章展示页面
- ✅ 本地验证通过
- ✅ 准备好部署到 GitHub Pages

现在可以：
1. 继续添加详细的数学公式推导
2. 部署到 GitHub Pages 供他人访问
3. 定期抓取新文章扩充内容库
