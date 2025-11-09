# 机器学习与数学博客精选 | ML & Math Blog Posts

精选来自[科学空间](https://spaces.ac.cn)的机器学习与数学相关博客文章，添加详细的公式推导与注释。

## 项目结构

```
ml_posts/
├── docs/                    # GitHub Pages 根目录
│   ├── index.html          # 搜索首页
│   ├── posts/              # 生成的博客文章 HTML
│   ├── assets/             # CSS、JS、图片等资源
│   │   ├── css/
│   │   ├── js/
│   │   └── img/
│   └── data/
│       └── blog_list.json  # 博客列表数据
├── blogs_raw/              # 原始博客 Markdown 文件
├── scripts/                # 工具脚本
│   ├── fetch_and_filter.py    # RSS 抓取与筛选
│   └── generate_posts.py      # Markdown 转 HTML
├── templates/              # HTML 模板
│   └── post_template.html
├── blog_list.txt          # 博客清单（文本格式）
└── requirements.txt       # Python 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 抓取博客文章

从科学空间 RSS feed 抓取并筛选数学/机器学习相关文章：

```bash
python3 scripts/fetch_and_filter.py
```

这将：
- 抓取最新的博客文章
- 基于关键词筛选数学/机器学习相关内容
- 保存 Markdown 格式到 `blogs_raw/`
- 更新 `blog_list.txt` 和 `docs/data/blog_list.json`

### 3. 生成 HTML 页面

将 Markdown 文章转换为 HTML 页面：

```bash
python3 scripts/generate_posts.py
```

生成的 HTML 文件将保存到 `docs/posts/`。

### 4. 本地预览

使用 Python 内置 HTTP 服务器预览：

```bash
cd docs
python3 -m http.server 8000
```

然后访问 `http://localhost:8000`

## 工作流程

### 添加新文章

1. **自动抓取**（推荐）：
   ```bash
   python3 scripts/fetch_and_filter.py
   python3 scripts/generate_posts.py
   ```

2. **手动添加**：
   - 在 `blogs_raw/` 创建 Markdown 文件
   - 添加 frontmatter（见下方格式）
   - 运行 `python3 scripts/generate_posts.py`

### Markdown 文件格式

```markdown
---
title: 文章标题
slug: article-slug
date: 2025-11-10
source: https://spaces.ac.cn/archives/xxxx
tags: 机器学习, 数学, 优化
status: pending
---

# 文章标题

文章内容...

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释
```

### 添加公式推导

在 `blogs_raw/` 中的 Markdown 文件末尾的"公式推导与注释"部分添加详细推导：

```markdown
## 公式推导与注释

### 公式1：梯度下降更新规则

原文提到梯度下降的更新规则：

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

**推导过程**：

从泰勒展开可得...
```

使用 MathJax 语法编写数学公式（支持 `$inline$` 和 `$$display$$`）。

## 部署到 GitHub Pages

1. 将代码推送到 GitHub 仓库

2. 在仓库设置中启用 GitHub Pages：
   - Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` / `master`
   - Folder: `/docs`

3. 访问 `https://your-username.github.io/ml_posts/`

## 功能特性

- ✅ 自动从 RSS feed 抓取相关文章
- ✅ 基于关键词智能筛选数学/机器学习内容
- ✅ 实时搜索功能（标题、标签、内容）
- ✅ MathJax 数学公式渲染
- ✅ 代码高亮显示
- ✅ 响应式设计，支持移动端
- ✅ 无需后端，纯静态部署

## 进阶使用

### 自定义筛选关键词

编辑 `scripts/fetch_and_filter.py` 中的 `ML_KEYWORDS` 和 `MATH_KEYWORDS` 列表。

### 修改样式

- 全局样式：`docs/assets/css/main.css`
- 文章样式：`docs/assets/css/post.css`

### 修改模板

编辑 `templates/post_template.html` 来自定义文章页面布局。

## 维护任务

### 更新博客列表

```bash
# 抓取新文章
python3 scripts/fetch_and_filter.py

# 重新生成所有页面
python3 scripts/generate_posts.py
```

### 标记文章完成状态

编辑 `docs/data/blog_list.json`，将相应文章的 `status` 改为 `"completed"`。

## 技术栈

- **前端**: HTML5, CSS3, Bootstrap 5, JavaScript
- **数学公式**: MathJax 3
- **代码高亮**: Highlight.js
- **后端脚本**: Python 3
- **Markdown**: Python-Markdown
- **部署**: GitHub Pages

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目内容来源于[科学空间](https://spaces.ac.cn)，版权归原作者所有。
本项目代码采用 MIT 许可证。

## 致谢

- 博客来源：[科学空间](https://spaces.ac.cn) by 苏剑林
- 页面模板灵感：[ICLR Blogposts 2025](https://github.com/iclr-blogposts/2025)
