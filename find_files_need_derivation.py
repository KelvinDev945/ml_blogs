#!/usr/bin/env python3
"""
Find blog posts that need detailed derivations
"""

import re
from pathlib import Path

BLOGS_RAW_DIR = Path("blogs_raw")

def check_derivation_status(file_path):
    """Check if file has detailed derivation"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = {
        'filename': file_path.name,
        'has_derivation_section': False,
        'derivation_lines': 0,
        'is_detailed': False,
        'has_formulas': False
    }

    # Check for derivation section
    derivation_match = re.search(r'## 公式推导与注释', content)
    if derivation_match:
        result['has_derivation_section'] = True
        derivation_content = content[derivation_match.start():]

        # Count lines
        derivation_lines = [l for l in derivation_content.split('\n') if l.strip()]
        result['derivation_lines'] = len(derivation_lines)

        # Check for formulas (LaTeX math)
        formula_count = len(re.findall(r'\$\$|\\\[|\\\(', derivation_content))
        result['has_formulas'] = formula_count > 5

        # Consider detailed if > 100 lines and has formulas
        result['is_detailed'] = result['derivation_lines'] > 100 and result['has_formulas']

    return result

def main():
    """Find files needing derivations"""

    md_files = sorted(BLOGS_RAW_DIR.glob("*.md"))

    print(f"检查 {len(md_files)} 个博客文件...\n")
    print("=" * 100)

    needs_derivation = []
    has_simple_derivation = []
    has_detailed_derivation = []

    for md_file in md_files:
        result = check_derivation_status(md_file)

        if not result['has_derivation_section']:
            needs_derivation.append(result)
        elif not result['is_detailed']:
            has_simple_derivation.append(result)
        else:
            has_detailed_derivation.append(result)

    print(f"\n总文件数: {len(md_files)}")
    print(f"  ✅ 已有详细推导: {len(has_detailed_derivation)} ({len(has_detailed_derivation)*100//len(md_files)}%)")
    print(f"  ⚠️  简单推导/TODO: {len(has_simple_derivation)} ({len(has_simple_derivation)*100//len(md_files)}%)")
    print(f"  ❌ 无推导部分: {len(needs_derivation)} ({len(needs_derivation)*100//len(md_files)}%)")

    print(f"\n{'=' * 100}")
    print(f"需要添加推导的文件 ({len(needs_derivation)} 个)")
    print(f"{'=' * 100}\n")

    for result in needs_derivation[:20]:
        print(f"  {result['filename']}")

    if len(needs_derivation) > 20:
        print(f"\n  ... 还有 {len(needs_derivation) - 20} 个文件")

    print(f"\n{'=' * 100}")
    print(f"简单推导需要增强的文件 ({len(has_simple_derivation)} 个)")
    print(f"{'=' * 100}\n")

    for result in has_simple_derivation[:20]:
        print(f"  {result['filename']} - {result['derivation_lines']} 行")

    if len(has_simple_derivation) > 20:
        print(f"\n  ... 还有 {len(has_simple_derivation) - 20} 个文件")

    # Save list
    with open("files_need_derivation.txt", 'w', encoding='utf-8') as f:
        f.write("需要添加详细推导的文件列表\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"无推导部分 ({len(needs_derivation)} 个):\n")
        f.write("-" * 100 + "\n")
        for result in needs_derivation:
            f.write(f"{result['filename']}\n")

        f.write(f"\n\n简单推导需要增强 ({len(has_simple_derivation)} 个):\n")
        f.write("-" * 100 + "\n")
        for result in has_simple_derivation:
            f.write(f"{result['filename']} - {result['derivation_lines']} 行\n")

    print(f"\n{'=' * 100}")
    print(f"文件列表已保存至: files_need_derivation.txt")
    print(f"{'=' * 100}")

if __name__ == "__main__":
    main()
