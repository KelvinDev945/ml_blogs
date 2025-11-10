#!/usr/bin/env python3
"""
使用archive ID对博客排序并分配序号
archive ID越小 = 越早的文章 = 序号越小
"""

import json
import re
from pathlib import Path

BLOG_LIST_JSON = Path("docs/data/blog_list.json")

def extract_archive_id(url):
    """从URL提取archive ID"""
    match = re.search(r'/archives/(\d+)', url)
    return int(match.group(1)) if match else 999999

def main():
    print("=" * 60)
    print("基于archive ID排序博客并分配序号")
    print("=" * 60)

    # 加载
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blogs = json.load(f)

    print(f"\n总共 {len(blogs)} 篇博客")

    # 提取archive ID
    for blog in blogs:
        source = blog.get('source', '')
        archive_id = extract_archive_id(source)
        blog['archive_id'] = archive_id

    # 按archive ID排序（越小越早）
    blogs.sort(key=lambda b: b.get('archive_id', 999999))

    # 分配序号
    for i, blog in enumerate(blogs, 1):
        blog['post_number'] = i

    # 保存
    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(blogs, f, ensure_ascii=False, indent=2)

    print(f"✓ 已按archive ID排序")
    print(f"✓ 已分配序号 1-{len(blogs)}")

    # 显示结果
    print(f"\n最早的5篇 (archive ID最小):")
    for b in blogs[:5]:
        print(f"  #{b['post_number']}: {b['title'][:45]}")
        print(f"      archive: {b['archive_id']}, date: {b.get('date', 'N/A')}")

    print(f"\n最新的5篇 (archive ID最大):")
    for b in blogs[-5:]:
        print(f"  #{b['post_number']}: {b['title'][:45]}")
        print(f"      archive: {b['archive_id']}, date: {b.get('date', 'N/A')}")

    print("\n" + "=" * 60)
    print("✓ 完成！")
    print(f"✓ 已保存: {BLOG_LIST_JSON}")
    print("\n下一步: python3 scripts/generate_posts.py 重新生成HTML")
    print("=" * 60)

if __name__ == '__main__':
    main()
