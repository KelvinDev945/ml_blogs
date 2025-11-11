#!/usr/bin/env python3
"""
删除不属于ML和数学主题的博客文章
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"
POSTS_DIR = PROJECT_ROOT / "docs" / "posts"

# 要删除的博客列表（不包括.md扩展名）
NON_ML_MATH_BLOGS = [
    # 生活杂记类
    "观测iss",
    "个性邮箱",
    "利用熄火保护-通断器实现燃气灶智能关火",
    "基于树莓派zero2w搭建一个随身旁路由",
    "智能家居之手搓一套能接入米家的零冷水装置",
    "智能家居之热水器零冷水技术原理浅析",
    "智能家居之小爱同学控制极米投影仪的简单方案",
    "生活杂记炒锅的尽头是铁锅",
    "生活杂记用电饭锅来煮米汤",

    # Cool-papers工具网站相关
    "写了个刷论文的辅助网站cool-papers",
    "cool-papers更新简单搭建了一个站内检索系统",
    "cool-papers-站内搜索的一些新尝试",
    "cool-papers浏览器扩展升级至v020",
    "cool-papers更新简单适配zotero-connector",
    "更便捷的cool-papers打开方式chrome重定向扩展",
    "新年快乐记录一下-cool-papers-的开发体验",

    # 前端技术/MathJax实现细节（不是数学内容）
    "让mathjax更好地兼容谷歌翻译和延时加载",
    "让mathjax的数学公式随窗口大小自动缩放",
    "近乎完美地解决mathjax与marked的冲突",

    # 编程技巧（非ML/数学）
    "旁门左道之如何让python的重试代码更加优雅",

    # 个人回复/公告
    "苏剑林-我的pretrain的小模型暂时没有链接",
    "苏剑林-就是反向构造出来的",

    # 吐槽文章/非技术
    "开局一段扯数据全靠编真被一篇神论文气到了",
    "关于whiteningbert原创性的疑问和沟通",
]

def remove_blog_files():
    """删除博客的markdown和HTML文件"""
    removed_count = 0
    skipped_count = 0

    for blog_name in NON_ML_MATH_BLOGS:
        # 删除markdown文件
        md_file = BLOGS_RAW_DIR / f"{blog_name}.md"
        if md_file.exists():
            md_file.unlink()
            print(f"✓ 删除markdown: {md_file.name}")
            removed_count += 1
        else:
            print(f"⊘ 未找到markdown: {md_file.name}")
            skipped_count += 1

        # 删除HTML文件
        html_file = POSTS_DIR / f"{blog_name}.html"
        if html_file.exists():
            html_file.unlink()
            print(f"✓ 删除HTML: {html_file.name}")

    print(f"\n总结:")
    print(f"  已删除: {removed_count} 个markdown文件")
    print(f"  未找到: {skipped_count} 个文件")
    print(f"\n注意: 请运行 generate_posts.py 重新生成 blog_list.json")

if __name__ == "__main__":
    print("开始删除非ML/数学主题的博客...\n")
    remove_blog_files()
