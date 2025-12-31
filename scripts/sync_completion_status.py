#!/usr/bin/env python3
"""
同步blog_list.json中的完成状态
从TODO.md中提取已完成的文章，更新blog_list.json的状态
"""

import json
import re
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
TODO_FILE = PROJECT_ROOT / "TODO.md"
BLOG_LIST_JSON = PROJECT_ROOT / "docs" / "data" / "blog_list.json"

def extract_completed_titles_from_todo():
    """从TODO.md中提取所有已完成的文章标题"""
    completed_titles = set()

    with open(TODO_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配格式: * [x] #xxx - 标题 ✅ 或 * [x] 标题 ✅
    pattern = r'\* \[x\] (?:#\d+ - )?(.*?) ✅'
    matches = re.findall(pattern, content)

    for title in matches:
        # 清理标题
        title = title.strip()
        # 移除可能的前缀编号
        title = re.sub(r'^#\d+ - ', '', title)
        completed_titles.add(title)

    return completed_titles

def normalize_title(title):
    """标准化标题用于匹配"""
    # 移除特殊字符和空格差异
    normalized = title.strip()

    # 统一中文冒号
    normalized = normalized.replace('：', ':')
    # 统一括号
    normalized = normalized.replace('（', '(').replace('）', ')')

    # 处理Transformer升级之路系列的格式差异
    # "Transformer升级之路：10、" -> "Transformer升级之路10:"
    normalized = re.sub(r'Transformer升级之路:(\d+)、', r'Transformer升级之路\1:', normalized)

    # 移除英文括号中的内容（如英文注释）
    normalized = re.sub(r'\([A-Za-z\s]+\)', '', normalized)

    # 统一数字格式 - 移除括号
    normalized = normalized.replace('（', '').replace('）', '').replace('(', '').replace(')', '')

    # 统一连接符
    normalized = normalized.replace('=', '-').replace('+', '-')

    # 移除顿号
    normalized = normalized.replace('、', '')
    # 统一问号 - 都移除
    normalized = normalized.replace('？', '').replace('?', '')
    # 统一感叹号
    normalized = normalized.replace('！', '').replace('!', '')
    # 统一引号
    normalized = normalized.replace('"', '').replace('"', '').replace('"', '').replace("'", '').replace("'", '').replace("'", '')
    # 移除书名号
    normalized = normalized.replace('《', '').replace('》', '')
    # 移除空格
    normalized = normalized.replace(' ', '')
    # 移除冒号前后的差异
    normalized = normalized.replace(':：', ':').replace('：:', ':')

    return normalized

def sync_blog_list_status():
    """同步blog_list.json的完成状态"""

    # 1. 提取已完成的标题
    print("正在从TODO.md提取已完成的文章...")
    completed_titles = extract_completed_titles_from_todo()
    print(f"找到 {len(completed_titles)} 篇已完成的文章")

    # 创建标准化标题映射
    normalized_completed = {normalize_title(t): t for t in completed_titles}

    # 2. 读取blog_list.json
    print(f"\n正在读取 {BLOG_LIST_JSON}...")
    with open(BLOG_LIST_JSON, 'r', encoding='utf-8') as f:
        blog_list = json.load(f)

    # 3. 更新状态
    print("\n正在更新状态...")
    updated_count = 0
    already_completed = 0
    not_found = []

    for entry in blog_list:
        title = entry['title']
        normalized_title = normalize_title(title)

        # 检查是否在已完成列表中
        if normalized_title in normalized_completed:
            if entry['status'] != 'completed':
                entry['status'] = 'completed'
                updated_count += 1
                print(f"  ✓ 更新: {title}")
            else:
                already_completed += 1

    # 检查哪些TODO中的文章在blog_list中找不到
    blog_list_titles = {normalize_title(e['title']): e['title'] for e in blog_list}
    for norm_title in normalized_completed:
        if norm_title not in blog_list_titles:
            not_found.append(normalized_completed[norm_title])
            # 调试：打印未匹配的标题
            original = normalized_completed[norm_title]
            if '对角' in original or 'FSQ' in original or ('16' in original and 'Transformer' in original):
                print(f"\n调试 - 未匹配: {original}")
                print(f"  标准化: {norm_title}")
                # 查找可能的匹配
                for blog_norm, blog_orig in blog_list_titles.items():
                    if '对角' in blog_orig and '对角' in original:
                        print(f"  blog中的对角文章标准化: {blog_norm}")
                        print(f"  blog原标题: {blog_orig}")
                    if 'FSQ' in blog_orig and 'FSQ' in original:
                        print(f"  blog中的FSQ文章标准化: {blog_norm}")
                        print(f"  blog原标题: {blog_orig}")
                    if '16' in blog_orig and 'Transformer' in blog_orig and '16' in original:
                        print(f"  blog中的T16文章标准化: {blog_norm}")
                        print(f"  blog原标题: {blog_orig}")

    # 4. 保存更新后的文件
    print(f"\n正在保存到 {BLOG_LIST_JSON}...")
    with open(BLOG_LIST_JSON, 'w', encoding='utf-8') as f:
        json.dump(blog_list, f, ensure_ascii=False, indent=2)

    # 5. 输出统计
    print("\n" + "="*60)
    print("同步完成！统计信息：")
    print("="*60)
    print(f"已完成文章总数（TODO.md）: {len(completed_titles)}")
    print(f"blog_list.json中的记录: {len(blog_list)}")
    print(f"本次更新的文章数: {updated_count}")
    print(f"已经是completed状态: {already_completed}")
    print(f"在blog_list中找不到的: {len(not_found)}")

    if not_found:
        print("\n⚠️  以下文章在TODO中标记为完成，但在blog_list.json中找不到：")
        for title in sorted(not_found)[:10]:  # 只显示前10个
            print(f"  - {title}")
        if len(not_found) > 10:
            print(f"  ... 还有 {len(not_found) - 10} 篇")

    # 统计最终状态分布
    status_count = {}
    for entry in blog_list:
        status = entry['status']
        status_count[status] = status_count.get(status, 0) + 1

    print("\n最终状态分布：")
    for status, count in sorted(status_count.items()):
        percentage = count / len(blog_list) * 100
        print(f"  {status}: {count} 篇 ({percentage:.1f}%)")

    print("="*60)

if __name__ == "__main__":
    sync_blog_list_status()
