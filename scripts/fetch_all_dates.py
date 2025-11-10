#!/usr/bin/env python3
"""
从原网页抓取完整日期（包括年份）
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
DELAY = 0.3  # 秒

# 月份映射
MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

def fetch_date_from_url(url):
    """从原网页抓取完整日期"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        # 方法1: 从date-wrap div中提取日期和月份
        date_wrap = soup.find('div', class_='date-wrap')
        if date_wrap:
            day_span = date_wrap.find('span', class_='date-day')
            month_span = date_wrap.find('span', class_='date-month')

            if day_span and month_span:
                day = day_span.get_text().strip()
                month_text = month_span.get_text().strip()
                month = MONTH_MAP.get(month_text, None)

                if month:
                    # 从图片路径中提取年份
                    # 查找类似 /usr/uploads/YYYY/MM/ 的路径
                    img_match = re.search(r'/usr/uploads/(\d{4})/(\d{2})/', html)
                    if img_match:
                        year = img_match.group(1)
                        # 验证月份是否匹配
                        img_month = img_match.group(2)
                        if img_month == month:
                            date_str = f"{year}-{month}-{day.zfill(2)}"
                            # 验证日期有效性
                            try:
                                datetime.strptime(date_str, '%Y-%m-%d')
                                return date_str
                            except:
                                pass

        # 方法2: 从文章内容中查找完整日期
        # 查找2019-2025年的日期
        date_match = re.search(r'(201\d|202\d)年(\d{1,2})月(\d{1,2})日', html)
        if date_match:
            year, month, day = date_match.groups()
            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except:
                pass

        # 方法3: 查找标准格式日期
        date_match = re.search(r'(201\d|202\d)-(0\d|1[0-2])-([0-2]\d|3[01])', html)
        if date_match:
            return date_match.group(0)

        # 方法4: 从URL猜测年份，结合HTML中的月份
        if date_wrap and day_span and month_span:
            day = day_span.get_text().strip()
            month_text = month_span.get_text().strip()
            month = MONTH_MAP.get(month_text, None)

            if month:
                # 从archive ID推断大致年份
                archive_match = re.search(r'/archives/(\d+)', url)
                if archive_match:
                    archive_id = int(archive_match.group(1))
                    # 简单映射：archive ID越大越新
                    if archive_id < 1000:
                        year = '2020'
                    elif archive_id < 5000:
                        year = '2021'
                    elif archive_id < 8000:
                        year = '2022'
                    elif archive_id < 9500:
                        year = '2023'
                    elif archive_id < 10500:
                        year = '2024'
                    else:
                        year = '2025'

                    date_str = f"{year}-{month}-{day.zfill(2)}"
                    try:
                        datetime.strptime(date_str, '%Y-%m-%d')
                        return date_str
                    except:
                        pass

        return None

    except Exception as e:
        print(f"      错误: {e}")
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

    # 更新front matter中的date字段
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
    print("抓取所有博客的完整日期")
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

    need_fetch = sum(1 for b in blogs if not b.get('date'))
    print(f"需要抓取日期: {need_fetch} 篇")

    if need_fetch > 0:
        print(f"\n开始抓取（预计需要 {need_fetch * DELAY / 60:.1f} 分钟）...\n")

        success_count = 0
        failed_blogs = []

        for i, blog in enumerate(blogs, 1):
            if blog.get('date'):
                continue  # 已有日期，跳过

            source_url = blog.get('source', '')
            if not source_url:
                continue

            title_short = blog['title'][:45]
            print(f"[{i}/{len(blogs)}] {title_short}...")

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

        print(f"\n抓取结果:")
        print(f"  成功: {success_count}")
        print(f"  失败: {len(failed_blogs)}")

    # 按日期和archive ID排序
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
            # 没有日期，用archive ID排序
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
    print(f"  有日期: {with_date}/{len(blogs)}")

    # 显示结果
    print(f"\n最早的5篇:")
    for b in blogs[:5]:
        date = b.get('date', 'N/A')
        archive = extract_archive_id(b.get('source', ''))
        print(f"  #{b['post_number']}: {b['title'][:40]}")
        print(f"      日期: {date}, archive: {archive}")

    print(f"\n最新的5篇:")
    for b in blogs[-5:]:
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
