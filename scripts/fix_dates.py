#!/usr/bin/env python3
"""
修复博客日期并从原网页抓取
"""

import json
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

BLOG_LIST_JSON = Path("docs/data/blog_list.json")
BLOGS_RAW_DIR = Path("blogs_raw")
DELAY = 0.5  # 秒

def fetch_date_from_url(url):
    """从原网页抓取发布日期"""
    try:
        print(f"    抓取: {url[:60]}...")
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # 方法1: 查找发布时间的span标签
        # spaces.ac.cn的博客页面通常有这样的结构
        time_spans = soup.find_all('span', class_='time')
        for span in time_spans:
            text = span.get_text().strip()
            # 匹配日期格式，例如 "2024-12-10"
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
            if date_match:
                return date_match.group(0)

        # 方法2: 查找<time>标签
        time_tag = soup.find('time')
        if time_tag:
            # 尝试datetime属性
            if time_tag.get('datetime'):
                dt_str = time_tag['datetime']
                try:
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%d')
                except:
                    pass

            # 尝试文本内容
            text = time_tag.get_text().strip()
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
            if date_match:
                return date_match.group(0)

        # 方法3: 查找meta标签
        meta_date = soup.find('meta', {'property': 'article:published_time'})
        if meta_date and meta_date.get('content'):
            try:
                dt = datetime.fromisoformat(meta_date['content'].replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d')
            except:
                pass

        # 方法4: 在页面中搜索日期模式
        # 找到类似"发布于 2024-12-10"的文本
        page_text = soup.get_text()
        date_patterns = [
            r'发布于[：:\s]*(\d{4})-(\d{2})-(\d{2})',
            r'发布时间[：:\s]*(\d{4})-(\d{2})-(\d{2})',
            r'时间[：:\s]*(\d{4})-(\d{2})-(\d{2})',
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, page_text)
            if match:
                if len(match.groups()) == 3:
                    year, month, day = match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        return None

    except Exception as e:
        print(f"      错误: {e}")
        return None

def extract_archive_id(url):
    """从URL提取archive ID"""
    match = re.search(r'/archives/(\d+)', url)
    return int(match.group(1)) if match else 99999

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

def main():
    print("=" * 70)
    print("修复博客日期")
    print("=" * 70)

    # 加载
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blogs = json.load(f)

    print(f"\n总共 {len(blogs)} 篇博客")

    # 清理错误的日期
    for blog in blogs:
        date = blog.get('date', '')
        if 'http' in date or 'source:' in date or not date:
            blog['date'] = ''

    # 统计需要抓取的数量
    need_fetch = sum(1 for b in blogs if not b.get('date'))
    print(f"需要从网页抓取日期: {need_fetch} 篇")

    if need_fetch == 0:
        print("\n所有博客都有日期，无需抓取")
    else:
        print(f"\n开始抓取（预计需要 {need_fetch * DELAY / 60:.1f} 分钟）...\n")

        success_count = 0
        failed_blogs = []

        for i, blog in enumerate(blogs, 1):
            if blog.get('date'):
                continue  # 已有日期，跳过

            source_url = blog.get('source', '')
            if not source_url:
                continue

            print(f"[{i}/{len(blogs)}] {blog['title'][:50]}...")

            date_str = fetch_date_from_url(source_url)

            if date_str:
                blog['date'] = date_str
                # 更新markdown
                md_path = BLOGS_RAW_DIR / f"{blog['slug']}.md"
                update_markdown_date(md_path, date_str)
                print(f"    ✓ {date_str}")
                success_count += 1
            else:
                print(f"    ✗ 未获取到日期")
                failed_blogs.append(blog)

            time.sleep(DELAY)

        print(f"\n抓取完成:")
        print(f"  成功: {success_count}")
        print(f"  失败: {len(failed_blogs)}")

        if failed_blogs:
            print(f"\n失败的博客 (将使用archive ID排序):")
            for blog in failed_blogs[:10]:
                archive_id = extract_archive_id(blog.get('source', ''))
                print(f"  - {blog['title'][:50]} (archive: {archive_id})")

    # 按日期和archive ID排序
    print("\n按日期排序并分配序号...")

    def sort_key(blog):
        date_str = blog.get('date', '')
        archive_id = extract_archive_id(blog.get('source', ''))

        if date_str:
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return (0, dt.timestamp(), archive_id)
            except:
                # 日期格式错误，放到后面
                return (1, 0, archive_id)
        else:
            # 没有日期，用archive ID排序（放到最后）
            return (2, 0, archive_id)

    blogs.sort(key=sort_key)

    # 分配序号
    for i, blog in enumerate(blogs, 1):
        blog['post_number'] = i

    # 保存
    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(blogs, f, ensure_ascii=False, indent=2)

    # 统计
    with_date = sum(1 for b in blogs if b.get('date'))

    print(f"  ✓ 已分配序号 1-{len(blogs)}")
    print(f"  有日期: {with_date}/{len(blogs)}")

    # 显示结果
    print(f"\n最早的3篇:")
    for b in blogs[:3]:
        date = b.get('date', 'N/A')
        archive = extract_archive_id(b.get('source', ''))
        print(f"  #{b['post_number']}: {b['title'][:40]}")
        print(f"      日期: {date}, archive: {archive}")

    print(f"\n最新的3篇:")
    for b in blogs[-3:]:
        date = b.get('date', 'N/A')
        archive = extract_archive_id(b.get('source', ''))
        print(f"  #{b['post_number']}: {b['title'][:40]}")
        print(f"      日期: {date}, archive: {archive}")

    print("\n" + "=" * 70)
    print("✓ 完成！")
    print(f"✓ 已保存: {BLOG_LIST_JSON}")
    print("\n下一步: python3 scripts/generate_posts.py 重新生成HTML")
    print("=" * 70)

if __name__ == '__main__':
    main()
