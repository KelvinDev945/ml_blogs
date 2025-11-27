#!/usr/bin/env python3
"""检查哪些文章已增强，哪些未增强"""
import os
import subprocess
from pathlib import Path

# 获取所有已修改的文件
result = subprocess.run(
    ['git', 'diff', '--name-only'],
    capture_output=True,
    text=True,
    cwd='/home/kelvin/dev/ml_posts'
)

# 解析修改的文件（已增强）
enhanced_files = set()
for line in result.stdout.strip().split('\n'):
    if line.startswith('blogs_raw/') and line.endswith('.md'):
        enhanced_files.add(line.replace('blogs_raw/', ''))

# 获取所有文章
all_files = []
blogs_raw_dir = Path('/home/kelvin/dev/ml_posts/blogs_raw')
for md_file in sorted(blogs_raw_dir.glob('*.md')):
    all_files.append(md_file.name)

# 未增强的文章
not_enhanced = [f for f in all_files if f not in enhanced_files]

# 输出报告
print("=" * 80)
print("📊 文章增强状态报告")
print("=" * 80)
print()

print(f"📚 总文章数: {len(all_files)}")
print(f"✅ 已增强: {len(enhanced_files)}")
print(f"⏳ 未增强: {len(not_enhanced)}")
print(f"📈 完成率: {len(enhanced_files)/len(all_files)*100:.1f}%")
print()

print("=" * 80)
print("✅ 已增强的文章 (8篇)")
print("=" * 80)
for i, f in enumerate(sorted(enhanced_files), 1):
    print(f"{i}. {f}")

print()
print("=" * 80)
print(f"⏳ 未增强的文章 ({len(not_enhanced)}篇)")
print("=" * 80)
for i, f in enumerate(not_enhanced, 1):
    print(f"{i}. {f}")
