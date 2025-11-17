#!/usr/bin/env python3
"""
Check if modified blog posts still contain original blog content
"""

import re
from pathlib import Path

BLOGS_RAW_DIR = Path("blogs_raw")

def analyze_blog_structure(file_path):
    """Analyze the structure of a blog file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = {
        'filename': file_path.name,
        'has_frontmatter': False,
        'has_original_link': False,
        'has_original_content': False,
        'has_derivation': False,
        'original_content_lines': 0,
        'derivation_lines': 0,
        'total_lines': len(content.split('\n')),
        'issues': []
    }

    # Check frontmatter
    if content.startswith('---'):
        result['has_frontmatter'] = True

    # Check for original link
    if '**原文链接**:' in content or 'source:' in content:
        result['has_original_link'] = True

    # Split content into sections
    # Original content is between the header and "## 公式推导与注释"
    # or between header and end if no derivation section

    # Find the derivation section
    derivation_match = re.search(r'## 公式推导与注释', content)

    if derivation_match:
        result['has_derivation'] = True
        derivation_start = derivation_match.start()

        # Original content is from after frontmatter to before derivation
        if '---' in content:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                original_content = parts[2][:derivation_start - content.find(parts[2])]
            else:
                original_content = content[:derivation_start]
        else:
            original_content = content[:derivation_start]

        # Derivation content
        derivation_content = content[derivation_start:]
        result['derivation_lines'] = len([l for l in derivation_content.split('\n') if l.strip()])
    else:
        # No derivation section
        if '---' in content:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                original_content = parts[2]
            else:
                original_content = content
        else:
            original_content = content
        derivation_content = ""

    # Check if original content exists
    # Remove title, link, and metadata lines
    original_clean = re.sub(r'#.*?\n', '', original_content)
    original_clean = re.sub(r'\*\*原文链接\*\*:.*?\n', '', original_clean)
    original_clean = re.sub(r'\*\*发布日期\*\*:.*?\n', '', original_clean)
    original_clean = re.sub(r'---+', '', original_clean)

    # Count substantial lines (non-empty, non-header)
    substantial_lines = [
        line for line in original_clean.split('\n')
        if line.strip() and not line.strip().startswith('#')
        and not line.strip().startswith('**')
        and not line.strip() == '---'
    ]

    result['original_content_lines'] = len(substantial_lines)

    # Check for issues
    if not result['has_original_link']:
        result['issues'].append('缺少原文链接')

    if result['has_derivation']:
        # If has derivation, should have original content
        if result['original_content_lines'] < 20:
            result['issues'].append(f'原始内容过少 ({result["original_content_lines"]} 行)')
            result['has_original_content'] = False
        else:
            result['has_original_content'] = True
    else:
        # No derivation section
        if result['original_content_lines'] < 10:
            result['issues'].append(f'内容过少 ({result["original_content_lines"]} 行)')
            result['has_original_content'] = False
        else:
            result['has_original_content'] = True

    # Check for truncation markers
    if '[[[...]]]' in content or '[[...]]' in content:
        result['issues'].append('包含截断标记 [[[...]]]')
        result['has_original_content'] = False

    # Check if it's mostly TODO
    if 'TODO:' in content and result['original_content_lines'] < 10:
        result['issues'].append('主要是TODO内容')

    return result

def main():
    """Check all blog files"""

    if not BLOGS_RAW_DIR.exists():
        print(f"Error: {BLOGS_RAW_DIR} does not exist")
        return

    md_files = sorted(BLOGS_RAW_DIR.glob("*.md"))

    print(f"检查 {len(md_files)} 个markdown文件的原始内容...\n")
    print("=" * 100)

    # Categorize files
    complete_with_derivation = []
    complete_without_derivation = []
    incomplete = []

    for md_file in md_files:
        result = analyze_blog_structure(md_file)

        if result['issues']:
            incomplete.append(result)
        elif result['has_derivation']:
            complete_with_derivation.append(result)
        else:
            complete_without_derivation.append(result)

    # Print summary
    print(f"\n{'=' * 100}")
    print(f"总结报告")
    print(f"{'=' * 100}\n")

    print(f"总文件数: {len(md_files)}")
    print(f"  ✅ 包含原文+详细推导: {len(complete_with_derivation)} ({len(complete_with_derivation)*100//len(md_files)}%)")
    print(f"  ✅ 只包含原文(未添加推导): {len(complete_without_derivation)} ({len(complete_without_derivation)*100//len(md_files)}%)")
    print(f"  ⚠️  有问题的文件: {len(incomplete)} ({len(incomplete)*100//len(md_files)}%)")

    # Show files with issues
    if incomplete:
        print(f"\n{'=' * 100}")
        print(f"⚠️  有问题的文件 ({len(incomplete)} 个)")
        print(f"{'=' * 100}\n")

        for result in incomplete[:30]:  # Show first 30
            print(f"📄 {result['filename']}")
            print(f"   总行数: {result['total_lines']}")
            print(f"   原始内容行数: {result['original_content_lines']}")
            if result['has_derivation']:
                print(f"   推导内容行数: {result['derivation_lines']}")
            print(f"   问题: {', '.join(result['issues'])}")
            print()

        if len(incomplete) > 30:
            print(f"... 还有 {len(incomplete) - 30} 个文件有问题\n")

    # Show some examples of complete files with derivation
    if complete_with_derivation:
        print(f"\n{'=' * 100}")
        print(f"✅ 包含原文+详细推导的文件示例 (共 {len(complete_with_derivation)} 个)")
        print(f"{'=' * 100}\n")

        for result in complete_with_derivation[:10]:
            print(f"📄 {result['filename']}")
            print(f"   总行数: {result['total_lines']}")
            print(f"   原始内容: {result['original_content_lines']} 行")
            print(f"   推导内容: {result['derivation_lines']} 行")
            print()

    # Save detailed report
    output_file = Path("original_content_check.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("原始博客内容检查报告\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"总文件数: {len(md_files)}\n")
        f.write(f"包含原文+详细推导: {len(complete_with_derivation)}\n")
        f.write(f"只包含原文: {len(complete_without_derivation)}\n")
        f.write(f"有问题的文件: {len(incomplete)}\n\n")

        if incomplete:
            f.write("=" * 100 + "\n")
            f.write("有问题的文件详情\n")
            f.write("=" * 100 + "\n\n")

            for result in incomplete:
                f.write(f"{result['filename']}\n")
                f.write(f"  总行数: {result['total_lines']}\n")
                f.write(f"  原始内容: {result['original_content_lines']} 行\n")
                if result['has_derivation']:
                    f.write(f"  推导内容: {result['derivation_lines']} 行\n")
                f.write(f"  问题: {', '.join(result['issues'])}\n\n")

    print(f"\n{'=' * 100}")
    print(f"详细报告已保存至: {output_file}")
    print(f"{'=' * 100}")

    # Final verdict
    print(f"\n{'=' * 100}")
    if len(incomplete) == 0:
        print("✅ 结论: 所有文件都包含原始博客内容！")
    elif len(incomplete) <= 5:
        print(f"⚠️  结论: 大部分文件正常，仅 {len(incomplete)} 个文件有问题")
    else:
        print(f"❌ 结论: 发现 {len(incomplete)} 个文件有问题，需要修复")
    print(f"{'=' * 100}\n")

if __name__ == "__main__":
    main()
