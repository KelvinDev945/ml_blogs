#!/usr/bin/env python3
"""
Fetch all blog posts from spaces.ac.cn
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import re
from pathlib import Path
import html2text
from datetime import datetime

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"
BLOG_LIST_FILE = PROJECT_ROOT / "blog_list.txt"
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"

# URLs
ARCHIVE_URL = "https://spaces.ac.cn/content.html"
BASE_URL = "https://spaces.ac.cn"

# Keywords for filtering math/ML related posts
ML_KEYWORDS = [
    "机器学习", "深度学习", "神经网络", "transformer", "attention", "bert", "gpt",
    "卷积", "循环神经", "强化学习", "生成模型", "对抗", "优化", "梯度",
    "反向传播", "embedding", "预训练", "微调", "模型", "算法",
    "machine learning", "deep learning", "neural network", "ai", "artificial intelligence"
]

MATH_KEYWORDS = [
    "数学", "线性代数", "微积分", "概率", "统计", "矩阵", "向量", "张量",
    "导数", "积分", "优化", "凸优化", "梯度下降", "牛顿法", "拉格朗日",
    "公式", "定理", "证明", "推导", "泰勒", "傅里叶", "范数", "特征值",
    "奇异值", "正态分布", "高斯", "随机", "期望", "方差",
    "mathematics", "linear algebra", "calculus", "probability", "statistics",
    "matrix", "vector", "tensor", "derivative", "integral", "optimization"
]

ALL_KEYWORDS = ML_KEYWORDS + MATH_KEYWORDS


def slugify(text):
    """Convert title to slug for filename"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except Chinese, alphanumeric, spaces, and hyphens
    text = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', text)
    # Replace spaces and underscores with hyphens
    text = re.sub(r'[\s_]+', '-', text)
    # Remove consecutive hyphens
    text = re.sub(r'-+', '-', text)
    # Trim hyphens from start and end
    text = text.strip('-')
    # Limit length
    if len(text) > 100:
        text = text[:100].rsplit('-', 1)[0]
    return text


def is_relevant(title, content):
    """Check if post is relevant to math/ML based on keywords"""
    text = (title + " " + content).lower()
    return any(keyword.lower() in text for keyword in ALL_KEYWORDS)


def fetch_archive_page():
    """Fetch the archive page and extract all article links"""
    print(f"Fetching archive page from {ARCHIVE_URL}...")

    try:
        response = requests.get(ARCHIVE_URL, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except Exception as e:
        print(f"Error fetching archive page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all article links
    articles = []
    links = soup.find_all('a', href=re.compile(r'/archives/\d+'))

    for link in links:
        # Skip comment links
        if '#PostComment' in link.get('href', ''):
            continue

        href = link.get('href')
        title = link.get_text(strip=True)

        # Extract article ID
        match = re.search(r'/archives/(\d+)', href)
        if match:
            article_id = match.group(1)
            url = f"{BASE_URL}/archives/{article_id}"

            if title and len(title) > 0:
                articles.append({
                    'id': article_id,
                    'title': title,
                    'url': url
                })

    # Remove duplicates based on ID
    seen = set()
    unique_articles = []
    for article in articles:
        if article['id'] not in seen:
            seen.add(article['id'])
            unique_articles.append(article)

    print(f"Found {len(unique_articles)} unique articles")
    return unique_articles


def fetch_article_content(url, article_id):
    """Fetch article content"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find article content - try multiple selectors
        article_div = (soup.find('div', class_='PostContent') or
                      soup.find('div', class_='Post') or
                      soup.find('div', class_='post') or
                      soup.find('div', id='content'))

        if not article_div:
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
        tag_links = soup.find_all('a', href=re.compile(r'/tag/'))
        for tag_link in tag_links[:5]:  # Max 5 tags
            tag_text = tag_link.get_text(strip=True)
            if tag_text:
                tags.append(tag_text)

        # If no tags found, try to infer from content
        if not tags:
            content_text = article_div.get_text()
            if any(kw in content_text for kw in ["机器学习", "深度学习", "神经网络"]):
                tags.append("机器学习")
            if any(kw in content_text for kw in ["数学", "矩阵", "概率", "统计"]):
                tags.append("数学")
            if any(kw in content_text for kw in ["优化", "梯度", "学习率"]):
                tags.append("优化")

        if not tags:
            tags = ["机器学习"]

        # Convert to markdown
        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_links = False
        h.ignore_images = False

        content_md = h.handle(str(article_div))

        return content_md, date_str, tags

    except Exception as e:
        print(f"  Error fetching article {article_id}: {e}")
        return None, None, None


def fetch_all_and_save(max_articles=None, start_from=0):
    """Fetch all articles and save to disk"""

    # Fetch all article links from archive
    articles = fetch_archive_page()

    if not articles:
        print("No articles found!")
        return

    print(f"\nTotal articles to process: {len(articles)}")

    if max_articles:
        articles = articles[start_from:start_from + max_articles]
        print(f"Processing {len(articles)} articles (from {start_from} to {start_from + len(articles)})")

    # Create directories
    BLOGS_RAW_DIR.mkdir(exist_ok=True)

    # Track saved articles
    saved_articles = []
    skipped_articles = []

    for idx, article in enumerate(articles, 1):
        article_id = article['id']
        title = article['title']
        url = article['url']

        print(f"\n[{idx}/{len(articles)}] Processing: {title}")

        # Check if already exists
        slug = slugify(title)
        if not slug:
            slug = f"article-{article_id}"

        md_file = BLOGS_RAW_DIR / f"{slug}.md"

        if md_file.exists():
            print(f"  ✓ Already exists: {md_file.name}")
            # Still add to the list
            saved_articles.append({
                "title": title,
                "slug": slug,
                "id": article_id,
                "url": url,
                "status": "existing"
            })
            continue

        # Fetch article content
        content, date_str, tags = fetch_article_content(url, article_id)

        if not content:
            print(f"  ✗ Failed to fetch content")
            skipped_articles.append(article)
            continue

        # Check relevance
        if not is_relevant(title, content):
            print(f"  ⊘ Not relevant (skipped)")
            skipped_articles.append(article)
            continue

        print(f"  ✓ Relevant to ML/Math")

        # Save to markdown
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

        print(f"  ✓ Saved: {md_file.name}")

        saved_articles.append({
            "title": title,
            "slug": slug,
            "id": article_id,
            "url": url,
            "date": date_str,
            "tags": tags,
            "status": "pending",
            "description": content[:200].replace('\n', ' ').strip() + "..."
        })

        # Rate limiting
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total processed: {len(articles)}")
    print(f"  Saved: {len([a for a in saved_articles if a.get('status') != 'existing'])}")
    print(f"  Already existed: {len([a for a in saved_articles if a.get('status') == 'existing'])}")
    print(f"  Skipped: {len(skipped_articles)}")
    print(f"{'='*60}")

    # Update blog_list.txt
    with open(BLOG_LIST_FILE, 'a', encoding='utf-8') as f:
        for article in saved_articles:
            if article.get('status') != 'existing':
                tags_str = ",".join(article.get('tags', ['机器学习']))
                date_str = article.get('date', '')
                f.write(f"{article['title']} | {article['slug']} | "
                       f"blogs_raw/{article['slug']}.md | pending | "
                       f"{date_str} | {tags_str}\n")

    print(f"✓ Updated {BLOG_LIST_FILE}")

    # Update blog_list.json
    try:
        with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_slugs = {entry.get('slug') for entry in existing_data}
    for article in saved_articles:
        if article['slug'] not in existing_slugs:
            existing_data.append({
                "title": article['title'],
                "slug": article['slug'],
                "path": f"blogs_raw/{article['slug']}.md",
                "status": article.get('status', 'pending'),
                "date": article.get('date', ''),
                "tags": article.get('tags', ['机器学习']),
                "description": article.get('description', '')
            })

    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Updated {BLOG_LIST_JSON}")
    print(f"\nTotal blogs in database: {len(existing_data)}")


if __name__ == "__main__":
    import sys

    # Parse arguments
    max_articles = None
    start_from = 0

    if len(sys.argv) > 1:
        try:
            max_articles = int(sys.argv[1])
        except ValueError:
            print("Usage: python fetch_all_blogs.py [max_articles] [start_from]")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            start_from = int(sys.argv[2])
        except ValueError:
            print("Usage: python fetch_all_blogs.py [max_articles] [start_from]")
            sys.exit(1)

    fetch_all_and_save(max_articles=max_articles, start_from=start_from)
