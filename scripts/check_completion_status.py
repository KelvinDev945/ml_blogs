#!/usr/bin/env python3
"""检查文章完成状态和行数统计."""
import json
import os
from pathlib import Path

# Load blog list
with open('/home/kelvin/dev/ml_posts/docs/data/blog_list.json', 'r', encoding='utf-8') as f:
    blogs = json.load(f)

print("=" * 80)
print("第八批文章状态检查（子批次A-B：优先级最高的6篇）")
print("=" * 80)

# 子批次A和B的文章
priority_articles = [
    (207, "低精度Attention可能存在有偏的舍入误差"),
    (202, "为什么线性注意力要加Short Conv？"),
    (206, "MuP之上：1. 好模型的三个特征"),
    (200, "AdamW的Weight RMS的渐近估计"),
    (199, "重新思考学习率与Batch Size（四）：EMA"),
    (197, "重新思考学习率与Batch Size（三）：Muon"),
]

for post_num, title in priority_articles:
    # 从blog_list.json中查找
    blog = next((b for b in blogs if b.get('post_number') == post_num), None)

    if blog:
        slug = blog.get('slug', '')
        status = blog.get('status', 'unknown')

        # 查找对应的文件
        blogs_raw = Path('/home/kelvin/dev/ml_posts/blogs_raw')
        matching_files = list(blogs_raw.glob(f"{slug}*.md"))

        print(f"\n#{post_num} - {title}")
        print(f"  JSON状态: {status}")

        if matching_files:
            # 检查主文件（不含backup的）
            main_files = [f for f in matching_files if 'backup' not in f.name]
            if main_files:
                file_path = main_files[0]
                # 读取行数
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())

                # 读取frontmatter中的status
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'status: completed' in content:
                        file_status = 'completed'
                    elif 'status: pending' in content:
                        file_status = 'pending'
                    else:
                        file_status = 'unknown'

                print(f"  文件: {file_path.name}")
                print(f"  文件状态: {file_status}")
                print(f"  行数: {lines} 行 {'✅' if lines >= 1200 else '⚠️ 需扩充'}")

                if status != file_status:
                    print(f"  ⚠️ 状态不一致！JSON={status}, 文件={file_status}")
            else:
                print(f"  ❌ 未找到有效文件")
        else:
            print(f"  ❌ 未找到对应文件")

print("\n" + "=" * 80)
print("建议：")
print("=" * 80)
print("1. 同步blog_list.json和文件frontmatter的status字段")
print("2. 对于行数<1200的文章，继续扩充数学推导")
print("3. 优先处理子批次A和B的6篇文章")
