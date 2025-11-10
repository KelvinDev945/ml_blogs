#!/usr/bin/env python3
"""
从原网页抓取完整日期 - 最终版本
查找<span>YYYY-MM-DD</span>格式的日期
"""

import json
import re
import time
import requests
from datetime import datetime
from pathlib import Path

BLOG_LIST_JSON = Path("docs/data/blog_list.json")
BLOGS_RAW_DIR = Path("blogs_raw")
DELAY = 0.3  # 秒

def fetch_date_from_url(url):
    """从原网页抓取完整日期"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        html = response.text

        # 查找<span>YYYY-MM-DD</span>格式
        date_match = re.search(r'<span>(20\d{2})-(0\d|1[0-2])-([0-2]\d|3[01])</span>', html)
        if date_match:
            date_str = date_match.group(0).replace('<span>', '').replace('</span>', '')
            # 验证日期有效性
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except:
                pass

        # 备用：查找任何YYYY-MM-DD格式
        date_match = re.search(r'(20\d{2})-(0\d|1[0-2])-([0-2]\d|3[01])', html)
        if date_match:
            date_str = date_match.group(0)
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except:
                pass

        return None

    except Exception as e:
        return None

def extract_archive_id(url):
    """从URL提取archive ID"""
    match = re.search(r'/archives/(\d+)', url)
    return int(match.group(1)) if match else 999999

def update_markdown_date(md_path, date_str):
    """更新Markdown文件的日期字段"""
    if not md_path.exists():
        return

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    def replace_date(match):
        front_matter = match.group(1)
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

def main():
    print("=" * 70)
    print("抓取所有博客的完整日期（最终版本）")
    print("=" * 70)

    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blogs = json.load(f)

    print(f"\n总共 {len(blogs)} 篇博客")

    # 清理错误的日期
    for blog in blogs:
        date = blog.get('date', '')
        if 'http' in date or 'source:' in date or not date:
            blog['date'] = ''

    need_fetch = sum(1 for b in blogs if not b.get('date'))
    print(f"需要抓取日期: {need_fetch} 篇")

    if need_fetch > 0:
        estimated_time = need_fetch * DELAY / 60
        print(f"\n开始抓取（预计需要 {estimated_time:.1f} 分钟）...")
        print(f"延迟设置: {DELAY}秒/篇\n")

        success_count = 0
        failed_count = 0
        progress_marker = 0

        for i, blog in enumerate(blogs, 1):
            if blog.get('date'):
                continue

            source_url = blog.get('source', '')
            if not source_url:
                failed_count += 1
                continue

            # 进度显示
            if i % 10 == 0 or i == len(blogs):
                progress_marker = i
                percent = (i / len(blogs)) * 100
                print(f"[{percent:.0f}%] 进度: {i}/{len(blogs)}")

            date_str = fetch_date_from_url(source_url)

            if date_str:
                blog['date'] = date_str
                md_path = BLOGS_RAW_DIR / f"{blog['slug']}.md"
                update_markdown_date(md_path, date_str)
                success_count += 1
            else:
                failed_count += 1

            time.sleep(DELAY)

        print(f"\n抓取完成:")
        print(f"  ✓ 成功: {success_count}")
        print(f"  ✗ 失败: {failed_count}")

    # 按日期排序
    print("\n按日期排序...")

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
            return (2, 0, archive_id)

    blogs.sort(key=sort_key)

    # 分配序号
    for i, blog in enumerate(blogs, 1):
        blog['post_number'] = i

    # 保存
    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(blogs, f, ensure_ascii=False, indent=2)

    with_date = sum(1 for b in blogs if b.get('date') and 'http' not in b.get('date', ''))

    print(f"  ✓ 序号分配: 1-{len(blogs)}")
    print(f"  ✓ 有日期: {with_date}/{len(blogs)}")

    print(f"\n最早5篇:")
    for b in blogs[:5]:
        print(f"  #{b['post_number']}: {b['title'][:35]} ({b.get('date', 'N/A')})")

    print(f"\n最新5篇:")
    for b in blogs[-5:]:
        print(f"  #{b['post_number']}: {b['title'][:35]} ({b.get('date', 'N/A')})")

    print("\n" + "=" * 70)
    print("✓ 完成！")
    print(f"✓ 已保存: {BLOG_LIST_JSON}")
    print("\n下一步: python3 scripts/generate_posts.py 重新生成HTML")
    print("=" * 70)

if __name__ == '__main__':
    main()
