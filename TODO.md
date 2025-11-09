# TODO
在每一步开始前将步骤重点步骤写入PROGRESS_REPORT.md并再完成后更新进度。最新进度放在PROGRESS_REPORT.md的最前面，每一步或阶段结束后进行commit并push到master
在完成当前blog后，抓取全部blog并完成blog_list.txt中所有页面
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

## 下一步：内容增强与部署 ✅
- [x] 选择 1-2 篇文章添加详细的数学公式推导（已完成2篇）
- [x] 提交代码并部署到 GitHub Pages
- [x] 设置 GitHub Actions 自动部署

## 已完成：自动化与部署 ✅
- [x] 抓取 RSS feed 中的博客文章（10篇）
- [x] 将所有 blog 内容转化为网页
- [x] 补充 GitHub Action 的 workflow
- [x] 创建部署文档（DEPLOY.md）

## 后续迭代计划 📝

### 短期目标（1-2周）
- [x] 程序化抓取所有blog，并存储于blog_list.txt（已完成：245篇）
- [ ] 为剩余所有篇文章添加详细数学推导（2/245篇已完成）
- [ ] 优化首页搜索功能（添加标签筛选）
- [ ] 添加文章分类功能
- [ ] 设置定期自动抓取（每周）

### 中期目标（1个月）
- [ ] 实现深色模式切换
- [ ] 添加文章目录（TOC）

### 长期目标（持续）
- [ ] 添加交互式可视化

## 部署说明

请参阅 DEPLOY.md 了解如何：
- 启用 GitHub Pages
- 配置自动部署
- 本地预览和测试
- 更新内容
- 故障排查


