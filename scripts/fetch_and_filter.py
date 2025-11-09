#!/usr/bin/env python3
"""
Fetch and filter blog posts from spaces.ac.cn
Filters for math and machine learning related content
"""

import feedparser
import re
import json
import os
from pathlib import Path
from datetime import datetime
import html2text
import unicodedata

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"
BLOG_LIST_FILE = PROJECT_ROOT / "blog_list.txt"
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"

# RSS feed URL
FEED_URL = "https://spaces.ac.cn/feed"

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
    "公式", "定理", "证明", "推导", "泰勒", "傅里叶",
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


def html_to_markdown(html_content):
    """Convert HTML content to Markdown"""
    h = html2text.HTML2Text()
    h.body_width = 0  # Don't wrap lines
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    return h.handle(html_content)


def fetch_and_filter():
    """Main function to fetch, filter, and save blog posts"""
    print(f"Fetching RSS feed from {FEED_URL}...")

    try:
        feed = feedparser.parse(FEED_URL)
    except Exception as e:
        print(f"Error fetching feed: {e}")
        return

    if not feed.entries:
        print("No entries found in feed")
        return

    print(f"Found {len(feed.entries)} total entries")

    # Create directories if they don't exist
    BLOGS_RAW_DIR.mkdir(exist_ok=True)

    # Track blog entries
    blog_entries = []
    saved_count = 0

    for entry in feed.entries:
        title = entry.title
        link = entry.link
        published = entry.get('published', '')

        # Get content
        content = ""
        if hasattr(entry, 'content'):
            content = entry.content[0].value
        elif hasattr(entry, 'summary'):
            content = entry.summary

        # Check relevance
        if not is_relevant(title, content):
            print(f"  Skipping (not relevant): {title}")
            continue

        print(f"  ✓ Relevant: {title}")

        # Generate slug
        slug = slugify(title)
        if not slug:
            slug = f"post-{saved_count + 1}"

        # Avoid duplicate slugs
        base_slug = slug
        counter = 1
        while (BLOGS_RAW_DIR / f"{slug}.md").exists():
            slug = f"{base_slug}-{counter}"
            counter += 1

        # Convert HTML to Markdown
        markdown_content = html_to_markdown(content)

        # Parse date
        date_str = ""
        if published:
            try:
                dt = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
                date_str = dt.strftime("%Y-%m-%d")
            except:
                try:
                    dt = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
                    date_str = dt.strftime("%Y-%m-%d")
                except:
                    date_str = published[:10] if len(published) >= 10 else ""

        # Extract potential tags from content
        tags = []
        for keyword in ["机器学习", "深度学习", "数学", "优化", "神经网络"]:
            if keyword in title or keyword in content:
                tags.append(keyword)
        if not tags:
            tags = ["机器学习"]  # Default tag

        # Save markdown file
        md_file = BLOGS_RAW_DIR / f"{slug}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"---\n")
            f.write(f"title: {title}\n")
            f.write(f"slug: {slug}\n")
            f.write(f"date: {date_str}\n")
            f.write(f"source: {link}\n")
            f.write(f"tags: {', '.join(tags)}\n")
            f.write(f"status: pending\n")
            f.write(f"---\n\n")
            f.write(f"# {title}\n\n")
            f.write(f"**原文链接**: [{link}]({link})\n\n")
            f.write(f"**发布日期**: {date_str}\n\n")
            f.write(f"---\n\n")
            f.write(markdown_content)
            f.write(f"\n\n---\n\n")
            f.write(f"## 公式推导与注释\n\n")
            f.write(f"TODO: 添加详细的数学公式推导和注释\n\n")

        # Add to blog entries
        blog_entries.append({
            "title": title,
            "slug": slug,
            "path": f"blogs_raw/{slug}.md",
            "status": "pending",
            "date": date_str,
            "tags": tags,
            "description": markdown_content[:200].replace('\n', ' ').strip() + "..."
        })

        saved_count += 1

    print(f"\n✓ Saved {saved_count} relevant blog posts to {BLOGS_RAW_DIR}")

    # Update blog_list.txt
    with open(BLOG_LIST_FILE, 'a', encoding='utf-8') as f:
        for entry in blog_entries:
            tags_str = ",".join(entry['tags'])
            f.write(f"{entry['title']} | {entry['slug']} | {entry['path']} | "
                   f"{entry['status']} | {entry['date']} | {tags_str}\n")

    print(f"✓ Updated {BLOG_LIST_FILE}")

    # Update blog_list.json
    try:
        with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Add new entries (avoid duplicates by slug)
    existing_slugs = {entry.get('slug') for entry in existing_data}
    for entry in blog_entries:
        if entry['slug'] not in existing_slugs:
            existing_data.append(entry)

    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Updated {BLOG_LIST_JSON}")
    print(f"\nTotal blogs in database: {len(existing_data)}")


if __name__ == "__main__":
    fetch_and_filter()
