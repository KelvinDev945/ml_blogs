# TODO
在每一步开始前将步骤重点步骤写入PROGRESS_REPORT.md并再完成后更新进度。最新进度放在PROGRESS_REPORT.md的最前面，每一步或阶段结束后进行commit并push到master

## 阶段 1：模板与首页 ✅
- [x] 克隆并研究 `iclr-blogposts/2025` 模板仓库，梳理布局、构建流程与静态资源位置
- [x] 在 `docs/index.html` 实现搜索首页骨架，保持模板风格并适配本项目信息架构
- [x] 本地验证首页搜索与资源加载，整理 GitHub Pages 部署要点

## 阶段 2：内容抓取与整理 🔄
- [x] 编写 `scripts/fetch_and_filter.py` 抓取 `https://spaces.ac.cn/feed`，筛选数学或机器学习相关博文
- [x] 将抓取内容保存到 `blogs_raw/<slug>.md` 并在 `blog_list.txt` 记录标题、路径及完成状态
- [ ] **当前任务**：为选定博文中的数学公式撰写逐步推导与简要注释

## 阶段 3：页面生成与发布 ✅
- [x] 基于模板生成 `docs/posts/<slug>.html`，补充导航、元信息与公式推导说明
- [x] 验证整站链接、样式与搜索功能，准备 GitHub Pages 发布方案

## 下一步：内容增强与部署 📝
- [ ] 选择 1-2 篇文章添加详细的数学公式推导
- [ ] 提交代码并部署到 GitHub Pages
- [ ] 持续迭代优化

