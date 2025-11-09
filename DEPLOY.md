# GitHub Pages 部署指南

## 自动部署（推荐）

本项目已配置 GitHub Actions 自动部署。每次推送到 master 分支时，会自动：
1. 抓取最新博客文章
2. 生成 HTML 页面
3. 部署到 GitHub Pages

### 启用 GitHub Pages

1. 访问仓库的 **Settings** 页面
2. 在左侧菜单找到 **Pages**
3. 在 **Source** 部分选择：
   - Source: `GitHub Actions`
4. 保存设置

### 查看部署状态

- 访问仓库的 **Actions** 标签页
- 查看最新的 workflow 运行状态
- 部署成功后，网站将在几分钟内可访问

### 访问网站

部署完成后，网站将可通过以下地址访问：

```
https://[username].github.io/ml_blogs/
```

将 `[username]` 替换为你的 GitHub 用户名。

## 手动部署

如果不使用 GitHub Actions，可以手动部署：

### 方法 1：使用 /docs 目录（简单）

1. 访问仓库的 **Settings** → **Pages**
2. Source 选择：`Deploy from a branch`
3. Branch 选择：`master`
4. Folder 选择：`/docs`
5. 保存设置

网站将自动部署到 `https://[username].github.io/ml_blogs/`

### 方法 2：自定义域名

1. 在 `docs/` 目录下创建 `CNAME` 文件
2. 在文件中写入你的域名，例如：`blog.example.com`
3. 在域名提供商处配置 DNS：
   - 添加 CNAME 记录指向 `[username].github.io`
4. 在 GitHub Pages 设置中输入自定义域名

## 本地预览

在部署前可以本地预览：

```bash
cd docs
python3 -m http.server 8000
```

然后访问 `http://localhost:8000`

## 更新内容

### 抓取新文章

```bash
python3 scripts/fetch_and_filter.py
python3 scripts/generate_posts.py
git add -A
git commit -m "Update blog posts"
git push origin master
```

### 添加数学推导

1. 编辑 `blogs_raw/文章名.md` 文件
2. 在"公式推导与注释"部分添加内容
3. 重新生成 HTML：
   ```bash
   python3 scripts/generate_posts.py
   ```
4. 提交并推送

## 故障排查

### 部署失败

1. 检查 Actions 标签页的错误日志
2. 确认 requirements.txt 中的依赖都能正常安装
3. 检查 docs/ 目录是否有 index.html

### 网站无法访问

1. 确认 GitHub Pages 已启用
2. 检查 Source 设置是否正确
3. 等待几分钟让 DNS 生效
4. 清除浏览器缓存

### 数学公式不显示

1. 确认 HTML 中包含 MathJax 脚本
2. 检查公式语法是否正确（使用 `$$` 包裹）
3. 查看浏览器控制台是否有错误

## 维护建议

### 定期更新

建议每周运行一次抓取脚本：

```bash
python3 scripts/fetch_and_filter.py
python3 scripts/generate_posts.py
```

### 备份

定期备份 `blogs_raw/` 目录中的 Markdown 文件。

### 监控

- 定期检查 GitHub Actions 的运行状态
- 确认网站可访问性
- 检查是否有新的文章需要添加推导

## 进阶配置

### 自定义样式

编辑以下文件来自定义样式：
- `docs/assets/css/main.css` - 全局样式
- `docs/assets/css/post.css` - 文章页面样式

### 修改模板

编辑 `templates/post_template.html` 来修改文章页面布局。

### 添加功能

可以在 `docs/index.html` 中添加：
- 标签筛选
- 分类功能
- 分页
- 评论系统（如 Giscus）

## 技术支持

遇到问题？查看：
- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- 项目 README.md
- 项目 Issues
