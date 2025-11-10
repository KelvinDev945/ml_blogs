#!/usr/bin/env python3
"""
从HTML页面中提取日期信息
使用regex查找 year-month-day 格式的日期
"""

import json
import re
from pathlib import Path
from datetime import datetime

DOCS_POSTS_DIR = Path("docs/posts")
BLOG_LIST_JSON = Path("docs/data/blog_list.json")
BLOGS_RAW_DIR = Path("blogs_raw")

def extract_date_from_html(html_path):
    """从HTML文件中提取日期"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找 YYYY-MM-DD 格式的日期
        # 匹配2000-2099年的日期
        date_patterns = [
            r'(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])',  # 标准格式
            r'发布日期[：:\s]*(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])',
            r'date[：:\s]*(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])',
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # 如果pattern包含分组，提取完整的日期
                if isinstance(matches[0], tuple):
                    year, month, day = matches[0]
                    date_str = f"{year}-{month}-{day}"
                else:
                    date_str = matches[0]

                # 验证日期有效性
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    return date_str
                except:
                    continue

        return None

    except Exception as e:
        print(f"  错误读取 {html_path}: {e}")
        return None

def update_markdown_date(md_path, date_str):
    """更新Markdown文件的日期字段"""
    if not md_path.exists():
        return

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更新front matter中的date字段
    def replace_date(match):
        front_matter = match.group(1)
        # 替换date字段
        front_matter = re.sub(
            r'^date:\s*.*$',
            f'date: {date_str}',
            front_matter,
            flags=re.MULTILINE
        )
        return f'---\n{front_matter}\n---'

    content = re.sub(r'^---\s*\n(.*?)\n---', replace_date, content, count=1, flags=re.DOTALL)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)

def extract_archive_id(url):
    """从URL提取archive ID"""
    match = re.search(r'/archives/(\d+)', url)
    return int(match.group(1)) if match else 999999

def main():
    print("=" * 60)
    print("从HTML页面提取日期")
    print("=" * 60)

    # 加载blog_list.json
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blogs = json.load(f)

    print(f"\n总共 {len(blogs)} 篇博客")

    # 提取日期
    print("\n开始从HTML提取日期...")
    success_count = 0
    failed_count = 0

    for blog in blogs:
        slug = blog['slug']
        html_path = DOCS_POSTS_DIR / f"{slug}.html"

        if not html_path.exists():
            failed_count += 1
            continue

        date_str = extract_date_from_html(html_path)

        if date_str:
            # 更新blog数据
            blog['date'] = date_str

            # 更新markdown文件
            md_path = BLOGS_RAW_DIR / f"{slug}.md"
            update_markdown_date(md_path, date_str)

            success_count += 1
        else:
            failed_count += 1

    print(f"  成功提取: {success_count}")
    print(f"  未找到日期: {failed_count}")

    # 按日期排序
    print("\n按日期排序并分配序号...")

    def sort_key(blog):
        date_str = blog.get('date', '')
        archive_id = extract_archive_id(blog.get('source', ''))

        if date_str and 'http' not in date_str:
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return (0, dt.timestamp(), archive_id)
            except:
                return (1, 0, archive_id)
        else:
            # 没有有效日期，用archive ID排序
            return (2, 0, archive_id)

    blogs.sort(key=sort_key)

    # 分配序号
    for i, blog in enumerate(blogs, 1):
        blog['post_number'] = i

    # 保存
    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(blogs, f, ensure_ascii=False, indent=2)

    # 统计
    with_date = sum(1 for b in blogs if b.get('date') and 'http' not in b.get('date', ''))

    print(f"  ✓ 已分配序号 1-{len(blogs)}")
    print(f"  有效日期: {with_date}/{len(blogs)}")

    # 显示结果
    print(f"\n最早的5篇:")
    for b in blogs[:5]:
        date = b.get('date', 'N/A')
        archive = extract_archive_id(b.get('source', ''))
        print(f"  #{b['post_number']}: {b['title'][:45]}")
        print(f"      日期: {date}, archive: {archive}")

    print(f"\n最新的5篇:")
    for b in blogs[-5:]:
        date = b.get('date', 'N/A')
        archive = extract_archive_id(b.get('source', ''))
        print(f"  #{b['post_number']}: {b['title'][:45]}")
        print(f"      日期: {date}, archive: {archive}")

    print("\n" + "=" * 60)
    print("✓ 完成！")
    print(f"✓ 已保存: {BLOG_LIST_JSON}")
    print("\n下一步: python3 scripts/generate_posts.py 重新生成HTML")
    print("=" * 60)

if __name__ == '__main__':
    main()
