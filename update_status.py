#!/usr/bin/env python3
"""
批量检查文章内容并更新status
如果文章有 >500行 且包含较完整的扩充内容，则更新 status: pending -> completed
"""

import os
import re
from pathlib import Path

def check_article_completeness(filepath):
    """检查文章是否包含扩充内容的关键标记"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查关键指标
    indicators = {
        'has_derivation_box': '<div class="derivation-box">' in content or '推导' in content[:5000],
        'has_intuition_box': '<div class="intuition-box">' in content or '直觉' in content[:5000],
        'has_analysis_box': '<div class="analysis-box">' in content or '批判' in content or '缺陷' in content,
        'has_research_direction': '研究方向' in content or '未来' in content or '展望' in content,
        'line_count': len(content.split('\n'))
    }

    # 如果行数>500且至少有2个关键标记，认为基本完成
    score = sum([
        indicators['has_derivation_box'],
        indicators['has_intuition_box'],
        indicators['has_analysis_box'],
        indicators['has_research_direction']
    ])

    return indicators['line_count'] > 500 and score >= 2

def update_article_status(filepath):
    """更新文章status为completed"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换status
    new_content = re.sub(
        r'^status:\s*pending',
        'status: completed',
        content,
        flags=re.MULTILINE
    )

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    blogs_dir = Path('/home/kelvin/dev/ml_posts/blogs_raw')

    pending_files = []
    completed_files = []

    # 遍历所有md文件
    for md_file in blogs_dir.glob('*.md'):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                first_lines = ''.join([next(f) for _ in range(10)])

            # 只处理status: pending的文章
            if 'status: pending' in first_lines:
                if check_article_completeness(md_file):
                    if update_article_status(md_file):
                        completed_files.append(md_file.name)
                else:
                    pending_files.append(md_file.name)
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")

    print(f"✅ 已更新 {len(completed_files)} 篇文章的status为completed")
    print(f"⏳ 仍需扩充的文章：{len(pending_files)} 篇")

    if len(completed_files) > 0:
        print(f"\n前10篇已更新的文章：")
        for f in completed_files[:10]:
            print(f"  - {f}")

    if len(pending_files) > 0:
        print(f"\n前10篇仍需扩充的文章：")
        for f in pending_files[:10]:
            print(f"  - {f}")

if __name__ == '__main__':
    main()
