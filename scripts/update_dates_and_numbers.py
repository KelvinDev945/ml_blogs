#!/usr/bin/env python3
"""
为每篇博客添加日期和序号
- 日期根据原markdown文件中日期为准
- 如果没有找到，需要查询原网页
- 序号是越早的博客越小（按时间升序编号）
"""

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# 配置
BLOGS_RAW_DIR = Path("blogs_raw")
BLOG_LIST_JSON = Path("docs/data/blog_list.json")
DELAY_BETWEEN_REQUESTS = 1.0  # 秒

def extract_date_from_markdown(md_path):
    """从Markdown文件的front matter中提取日期"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取front matter
    match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return None

    front_matter = match.group(1)

    # 提取date字段
    date_match = re.search(r'^date:\s*(.+)$', front_matter, re.MULTILINE)
    if date_match:
        date_str = date_match.group(1).strip()
        if date_str and date_str != '':
            return date_str

    return None

def fetch_date_from_url(url):
    """从原网页抓取发布日期"""
    try:
        print(f"  正在抓取: {url}")
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # 尝试多种方式提取日期

        # 方法1: 查找<time>标签
        time_tag = soup.find('time')
        if time_tag and time_tag.get('datetime'):
            date_str = time_tag['datetime']
            # 解析ISO格式日期
            try:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d')
            except:
                pass

        # 方法2: 查找meta标签
        meta_tags = [
            soup.find('meta', {'property': 'article:published_time'}),
            soup.find('meta', {'name': 'date'}),
            soup.find('meta', {'name': 'publishdate'}),
        ]

        for meta in meta_tags:
            if meta and meta.get('content'):
                date_str = meta['content']
                try:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%d')
                except:
                    pass

        # 方法3: 从URL中提取archive编号，然后估算日期
        # spaces.ac.cn/archives/数字 - 数字越大越新
        match = re.search(r'/archives/(\d+)', url)
        if match:
            archive_id = int(match.group(1))
            # 这里我们先返回None，后面可以基于archive_id排序
            return None

        print(f"  ⚠️ 未能从网页提取日期: {url}")
        return None

    except Exception as e:
        print(f"  ❌ 抓取失败: {url}, 错误: {e}")
        return None

def update_markdown_date(md_path, date_str):
    """更新Markdown文件的日期字段"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更新front matter中的date字段
    def replace_date(match):
        front_matter = match.group(1)
        # 替换或添加date字段
        if re.search(r'^date:', front_matter, re.MULTILINE):
            front_matter = re.sub(
                r'^date:\s*.*$',
                f'date: {date_str}',
                front_matter,
                flags=re.MULTILINE
            )
        else:
            # 在slug后面添加date
            front_matter = re.sub(
                r'^(slug:.*)$',
                r'\1\ndate: ' + date_str,
                front_matter,
                flags=re.MULTILINE
            )
        return f'---\n{front_matter}\n---'

    content = re.sub(r'^---\s*\n(.*?)\n---', replace_date, content, count=1, flags=re.DOTALL)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)

def extract_archive_id(url):
    """从URL中提取archive ID"""
    match = re.search(r'/archives/(\d+)', url)
    if match:
        return int(match.group(1))
    return 0

def main():
    print("=" * 60)
    print("为博客添加日期和序号")
    print("=" * 60)

    # 1. 加载blog_list.json
    print("\n[1/5] 加载blog_list.json...")
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blogs = json.load(f)

    print(f"  共 {len(blogs)} 篇博客")

    # 2. 提取和更新日期
    print("\n[2/5] 检查和更新日期...")

    blogs_with_dates = []
    blogs_without_dates = []

    for blog in blogs:
        slug = blog['slug']
        md_path = BLOGS_RAW_DIR / f"{slug}.md"

        if not md_path.exists():
            print(f"  ⚠️ 文件不存在: {md_path}")
            continue

        # 从markdown提取日期
        date_str = extract_date_from_markdown(md_path)

        if date_str:
            blog['date'] = date_str
            blogs_with_dates.append(blog)
        else:
            blogs_without_dates.append(blog)

    print(f"  已有日期: {len(blogs_with_dates)} 篇")
    print(f"  缺少日期: {len(blogs_without_dates)} 篇")

    # 3. 从网页抓取缺失的日期
    if blogs_without_dates:
        print(f"\n[3/5] 从网页抓取缺失的日期 (共{len(blogs_without_dates)}篇)...")
        print("  注意: 这可能需要几分钟时间...")

        for i, blog in enumerate(blogs_without_dates, 1):
            source_url = blog.get('source', '')
            if not source_url:
                print(f"  [{i}/{len(blogs_without_dates)}] ⚠️ {blog['title']}: 无source URL")
                continue

            print(f"  [{i}/{len(blogs_without_dates)}] {blog['title']}")

            date_str = fetch_date_from_url(source_url)

            if date_str:
                blog['date'] = date_str
                # 更新markdown文件
                md_path = BLOGS_RAW_DIR / f"{blog['slug']}.md"
                update_markdown_date(md_path, date_str)
                print(f"    ✓ 日期: {date_str}")
            else:
                # 如果无法获取日期，使用archive ID作为排序依据
                archive_id = extract_archive_id(source_url)
                blog['archive_id'] = archive_id
                print(f"    ⚠️ 未获取到日期，archive ID: {archive_id}")

            # 延迟，避免请求过快
            if i < len(blogs_without_dates):
                time.sleep(DELAY_BETWEEN_REQUESTS)
    else:
        print("\n[3/5] 跳过抓取 (所有博客都有日期)")

    # 4. 按日期排序并分配序号
    print("\n[4/5] 按日期排序并分配序号...")

    # 合并所有博客
    all_blogs = blogs_with_dates + blogs_without_dates

    # 排序函数：先按日期，再按archive ID
    def sort_key(blog):
        date_str = blog.get('date', '')
        archive_id = blog.get('archive_id', extract_archive_id(blog.get('source', '')))

        if date_str:
            try:
                # 转换为日期对象
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return (dt, archive_id)
            except:
                # 如果日期格式不对，使用archive_id
                return (datetime(1970, 1, 1), archive_id)
        else:
            # 没有日期，只用archive_id
            return (datetime(1970, 1, 1), archive_id)

    # 排序：越早的越靠前
    all_blogs.sort(key=sort_key)

    # 分配序号
    for i, blog in enumerate(all_blogs, 1):
        blog['post_number'] = i

    print(f"  已分配序号: 1 到 {len(all_blogs)}")

    # 显示一些统计信息
    with_dates = sum(1 for b in all_blogs if b.get('date'))
    print(f"  有日期的博客: {with_dates}/{len(all_blogs)}")

    # 显示最早和最晚的博客
    if all_blogs:
        earliest = all_blogs[0]
        latest = all_blogs[-1]
        print(f"\n  最早 (#1): {earliest['title']}")
        print(f"    日期: {earliest.get('date', 'N/A')}, archive: {extract_archive_id(earliest.get('source', ''))}")
        print(f"  最晚 (#{len(all_blogs)}): {latest['title']}")
        print(f"    日期: {latest.get('date', 'N/A')}, archive: {extract_archive_id(latest.get('source', ''))}")

    # 5. 保存更新后的blog_list.json
    print("\n[5/5] 保存更新后的blog_list.json...")

    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_blogs, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 已保存到: {BLOG_LIST_JSON}")

    # 统计信息
    print("\n" + "=" * 60)
    print("完成统计:")
    print("=" * 60)
    print(f"总博客数: {len(all_blogs)}")
    print(f"有日期: {sum(1 for b in all_blogs if b.get('date'))}")
    print(f"序号范围: 1 到 {len(all_blogs)}")
    print("\n下一步: 运行 python3 scripts/generate_posts.py 重新生成HTML页面")
    print("=" * 60)

if __name__ == '__main__':
    main()
