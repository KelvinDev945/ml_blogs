#!/usr/bin/env python3
"""
Download incomplete blog posts
"""

import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import html2text
from datetime import datetime

BLOGS_RAW_DIR = Path("blogs_raw")

# Incomplete files with their URLs
INCOMPLETE_FILES = [
    {
        "slug": "mup之上1-好模型的三个特征",
        "url": "https://spaces.ac.cn/archives/11340",
        "title": "MuP之上：1. 好模型的三个特征"
    },
    {
        "slug": "重新思考学习率与batch-size三muon",
        "url": "https://spaces.ac.cn/archives/11285",
        "title": "重新思考学习率与Batch Size（三）：Muon"
    },
    {
        "slug": "重新思考学习率与batch-size四ema",
        "url": "https://spaces.ac.cn/archives/11301",
        "title": "重新思考学习率与Batch Size（四）：EMA"
    }
]

def fetch_article_content(url):
    """Fetch article content from URL"""
    try:
        print(f"  Fetching: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find article content - try multiple selectors
        article_div = (soup.find('div', class_='PostContent') or
                      soup.find('div', class_='Post') or
                      soup.find('div', class_='post') or
                      soup.find('article') or
                      soup.find('div', id='content'))

        if not article_div:
            print(f"  ✗ Could not find article content")
            return None, None, None

        # Extract date
        date_tag = soup.find('time')
        date_str = ""
        if date_tag:
            datetime_attr = date_tag.get('datetime', '')
            if datetime_attr:
                try:
                    dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d")
                except:
                    pass

        # Extract tags
        tags = []
        tag_links = soup.find_all('a', href=lambda x: x and '/tag/' in x)
        for tag_link in tag_links[:5]:
            tag_text = tag_link.get_text(strip=True)
            if tag_text:
                tags.append(tag_text)

        if not tags:
            tags = ["优化"]

        # Convert to markdown
        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_links = False
        h.ignore_images = False
        h.bypass_tables = False

        content_md = h.handle(str(article_div))

        return content_md, date_str, tags

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None, None

def download_incomplete_files():
    """Download all incomplete files"""

    print(f"Downloading {len(INCOMPLETE_FILES)} incomplete files\n")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for item in INCOMPLETE_FILES:
        slug = item['slug']
        url = item['url']
        title = item['title']

        print(f"\nProcessing: {title}")

        # Fetch content
        content, date_str, tags = fetch_article_content(url)

        if not content:
            print(f"  ✗ Failed to download")
            fail_count += 1
            continue

        # Create backup of old file
        md_file = BLOGS_RAW_DIR / f"{slug}.md"
        if md_file.exists():
            backup_file = BLOGS_RAW_DIR / f"{slug}.md.backup"
            md_file.rename(backup_file)
            print(f"  ✓ Backed up old file to: {backup_file.name}")

        # Save new content
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"---\n")
            f.write(f"title: {title}\n")
            f.write(f"slug: {slug}\n")
            f.write(f"date: {date_str}\n")
            f.write(f"source: {url}\n")
            f.write(f"tags: {', '.join(tags)}\n")
            f.write(f"status: pending\n")
            f.write(f"---\n\n")
            f.write(f"# {title}\n\n")
            f.write(f"**原文链接**: [{url}]({url})\n\n")
            f.write(f"**发布日期**: {date_str}\n\n")
            f.write(f"---\n\n")
            f.write(content)
            f.write(f"\n\n---\n\n")
            f.write(f"## 公式推导与注释\n\n")
            f.write(f"TODO: 添加详细的数学公式推导和注释\n\n")

        # Check file size
        file_size = md_file.stat().st_size
        line_count = len(md_file.read_text(encoding='utf-8').split('\n'))

        print(f"  ✓ Downloaded: {file_size} bytes, {line_count} lines")
        success_count += 1

        # Rate limiting
        time.sleep(2)

    print(f"\n{'=' * 80}")
    print(f"Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    download_incomplete_files()
