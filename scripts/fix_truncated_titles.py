#!/usr/bin/env python3
"""修复被截断的文章标题"""

import json
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"

def fetch_full_title(url):
    """从网站获取完整标题"""
    try:
        response = requests.get(url, timeout=15)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        # 方法1: 查找entry-title
        title_tag = soup.find('h1', class_='entry-title')
        if title_tag:
            return title_tag.get_text().strip()

        # 方法2: 从title标签提取
        title_tag = soup.find('title')
        if title_tag:
            full_title = title_tag.get_text().strip()
            # 移除网站名称后缀
            full_title = re.sub(r'\s*[|\-]\s*(苏剑林的博客|科学空间|Scientific Spaces).*$', '', full_title)
            return full_title.strip()

        return None
    except Exception as e:
        print(f"    ⚠️  获取失败: {e}")
        return None

def extract_source_url(md_file):
    """从markdown文件提取source URL"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        match = re.search(r'^source:\s*(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except:
        pass
    return None

def main():
    # 读取blog_list.json
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blog_list = json.load(f)

    # 找到所有被截断的标题
    truncated_entries = [e for e in blog_list if '...' in e['title']]

    print(f"找到 {len(truncated_entries)} 个被截断的标题\n")

    updated = 0

    for entry in truncated_entries:
        print(f"处理: {entry['title']}")

        # 获取source URL
        md_file = BLOGS_RAW_DIR / f"{entry['slug']}.md"
        source_url = entry.get('source')

        if not source_url and md_file.exists():
            source_url = extract_source_url(md_file)

        if not source_url:
            print(f"  ⚠️  没有source URL")
            continue

        # 获取完整标题
        full_title = fetch_full_title(source_url)

        if full_title and full_title != entry['title']:
            print(f"  ✓ 更新: {full_title}")
            entry['title'] = full_title
            updated += 1

            # 同时更新markdown文件
            if md_file.exists():
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 更新YAML中的title
                    content = re.sub(
                        r'^title:.*$',
                        f'title: {full_title}',
                        content,
                        count=1,
                        flags=re.MULTILINE
                    )

                    # 更新第一个标题
                    content = re.sub(
                        r'^# .*$',
                        f'# {full_title}',
                        content,
                        count=1,
                        flags=re.MULTILINE
                    )

                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    print(f"  ✓ 已更新markdown文件")
                except Exception as e:
                    print(f"  ⚠️  更新markdown文件失败: {e}")

        time.sleep(1)  # 避免请求过快

    # 保存更新
    if updated > 0:
        with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
            json.dump(blog_list, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ 已更新 {updated} 个标题")
        print(f"{'='*60}")
    else:
        print("\n没有需要更新的标题")

if __name__ == "__main__":
    main()
