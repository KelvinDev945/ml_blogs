#!/usr/bin/env python3
"""
更新existing状态文章的日期和完整标题
"""

import json
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"

def extract_metadata_from_file(md_file):
    """从markdown文件中提取元数据"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取YAML front matter
    match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return None

    yaml_content = match.group(1)
    metadata = {}

    # 解析YAML字段
    for line in yaml_content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()

    return metadata

def fetch_article_date(url):
    """从网站获取文章发布日期"""
    try:
        print(f"  正在获取日期: {url}")
        response = requests.get(url, timeout=15)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        # 方法1: 查找time标签
        time_tag = soup.find('time')
        if time_tag and time_tag.get('datetime'):
            datetime_str = time_tag.get('datetime')
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d")

        # 方法2: 查找date-wrap格式 (新版网站)
        date_wrap = soup.find('div', class_='date-wrap')
        if date_wrap:
            day_span = date_wrap.find('span', class_='date-day')
            month_span = date_wrap.find('span', class_='date-month')

            if day_span and month_span:
                day = day_span.get_text().strip()
                month_abbr = month_span.get_text().strip()

                # 月份缩写映射
                month_map = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                }

                month = month_map.get(month_abbr, '01')
                # 假设是当前年份或2025年（根据文章编号推断）
                year = '2025'  # 可以根据archives编号进一步优化

                return f"{year}-{month}-{day.zfill(2)}"

        return None
    except Exception as e:
        print(f"  ⚠️ 获取失败: {e}")
        return None

def update_existing_articles():
    """更新existing状态的文章"""

    # 读取blog_list.json
    print("读取blog_list.json...")
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blog_list = json.load(f)

    # 找到所有existing状态的文章
    existing_articles = [entry for entry in blog_list if entry['status'] == 'existing']
    print(f"找到 {len(existing_articles)} 篇existing状态的文章\n")

    updated_count = 0

    for entry in existing_articles:
        slug = entry['slug']
        md_file = BLOGS_RAW_DIR / f"{slug}.md"

        if not md_file.exists():
            print(f"⚠️ 文件不存在: {slug}.md")
            continue

        print(f"\n处理: {entry['title']}")

        # 从文件中提取元数据
        metadata = extract_metadata_from_file(md_file)
        if not metadata:
            print(f"  ⚠️ 无法提取元数据")
            continue

        # 更新完整标题
        full_title = metadata.get('title', entry['title'])
        if full_title != entry['title']:
            print(f"  ✓ 更新标题: {full_title}")
            entry['title'] = full_title
            updated_count += 1

        # 获取并更新日期
        if not entry.get('date'):
            source_url = metadata.get('source')
            if source_url:
                date = fetch_article_date(source_url)
                if date:
                    print(f"  ✓ 设置日期: {date}")
                    entry['date'] = date
                    updated_count += 1
                time.sleep(1)  # 避免请求过快

    # 保存更新
    print(f"\n保存更新到blog_list.json...")
    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(blog_list, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"更新完成！共更新 {updated_count} 个字段")
    print(f"{'='*60}")

if __name__ == "__main__":
    update_existing_articles()
