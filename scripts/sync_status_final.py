#!/usr/bin/env python3
"""最终版本：同步blog_list.json的完成状态"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TODO_FILE = PROJECT_ROOT / "TODO.md"
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"

def normalize_title(title):
    """标准化标题用于匹配 - 移除所有特殊字符和格式差异"""
    normalized = title.strip()

    # 统一冒号和括号
    normalized = normalized.replace('：', ':')

    # 处理Transformer系列格式
    normalized = re.sub(r'Transformer升级之路:(\d+)、', r'Transformer升级之路\1:', normalized)

    # 移除所有括号（包括中英文）
    normalized = normalized.replace('（', '').replace('）', '').replace('(', '').replace(')', '')

    # 移除英文括号内容
    normalized = re.sub(r'\([A-Za-z\s]+\)', '', normalized)

    # 统一连接符
    normalized = normalized.replace('=', '-').replace('+', '-')

    # 移除所有标点符号
    normalized = normalized.replace('、', '')
    normalized = normalized.replace('？', '').replace('?', '')
    normalized = normalized.replace('！', '').replace('!', '')

    # 移除所有引号（包括中英文）
    normalized = normalized.replace('"', '').replace('"', '').replace('"', '')
    normalized = normalized.replace("'", '').replace("'", '').replace("'", '')
    normalized = normalized.replace('『', '').replace('』', '')
    normalized = normalized.replace('「', '').replace('」', '')

    # 移除书名号
    normalized = normalized.replace('《', '').replace('》', '')

    # 移除所有空格
    normalized = normalized.replace(' ', '')

    return normalized

def extract_completed_from_todo():
    """从TODO.md提取已完成文章"""
    with open(TODO_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配 [x] 标记的文章
    pattern = r'\* \[x\] (?:#\d+ - )?(.*?) ✅'
    matches = re.findall(pattern, content)

    return [title.strip() for title in matches]

def main():
    # 1. 提取已完成标题
    print("提取TODO.md中的已完成文章...")
    completed_titles = extract_completed_from_todo()
    print(f"找到 {len(completed_titles)} 篇")

    # 2. 读取blog_list.json
    print("\n读取blog_list.json...")
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blog_list = json.load(f)

    # 3. 创建标准化映射
    completed_normalized = {normalize_title(t): t for t in completed_titles}

    # 4. 更新状态
    print("\n更新状态...")
    updated = 0

    for entry in blog_list:
        norm_title = normalize_title(entry['title'])
        if norm_title in completed_normalized and entry['status'] != 'completed':
            entry['status'] = 'completed'
            updated += 1
            print(f"  ✓ {entry['title']}")

    # 5. 保存
    print(f"\n保存blog_list.json...")
    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(blog_list, f, ensure_ascii=False, indent=2)

    # 6. 统计
    status_count = {}
    for entry in blog_list:
        status_count[entry['status']] = status_count.get(entry['status'], 0) + 1

    print("\n" + "="*60)
    print("同步完成！")
    print("="*60)
    print(f"已完成文章（TODO）: {len(completed_titles)}")
    print(f"本次更新: {updated} 篇")
    print(f"\n状态分布:")
    for status, count in sorted(status_count.items()):
        pct = count / len(blog_list) * 100
        print(f"  {status}: {count} 篇 ({pct:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
