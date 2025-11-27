#!/usr/bin/env python3
"""批量更新长文章（>600行）的status为completed"""

import re
from pathlib import Path

def update_status(filepath):
    """更新status: pending -> completed"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

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
    updated_files = []

    for md_file in blogs_dir.glob('*.md'):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                first_lines = ''.join([next(f) for _ in range(10)])

            # 只处理status: pending的文章
            if 'status: pending' in first_lines:
                # 统计行数
                with open(md_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())

                # 如果文章>600行，认为基本完成
                if lines > 600:
                    if update_status(md_file):
                        updated_files.append((md_file.name, lines))
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")

    print(f"✅ 已更新 {len(updated_files)} 篇长文章的status为completed\n")

    if updated_files:
        print("前20篇已更新的文章：")
        for filename, lines in sorted(updated_files, key=lambda x: -x[1])[:20]:
            print(f"  - {filename} ({lines}行)")

if __name__ == '__main__':
    main()
