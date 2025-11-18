#!/usr/bin/env python3
"""
Check completeness of blog content in blogs_raw directory
"""

import os
import re
from pathlib import Path

BLOGS_RAW_DIR = Path("blogs_raw")

def check_file_completeness(file_path):
    """Check if a markdown file has complete content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    total_lines = len(lines)

    # Extract metadata
    has_frontmatter = content.startswith('---')
    has_original_link = '**原文链接**:' in content or 'source:' in content
    has_todo = 'TODO:' in content

    # Check if content is substantial (more than just metadata + TODO)
    # Remove frontmatter
    if has_frontmatter:
        try:
            parts = content.split('---', 2)
            if len(parts) >= 3:
                main_content = parts[2]
            else:
                main_content = content
        except:
            main_content = content
    else:
        main_content = content

    # Remove TODO section
    if has_todo:
        main_content = re.sub(r'## 公式推导与注释.*$', '', main_content, flags=re.DOTALL)

    # Count substantial content lines (non-empty, non-header)
    substantial_lines = [
        line for line in main_content.split('\n')
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('**')
    ]

    substantial_line_count = len(substantial_lines)

    # Determine completeness
    # Consider complete if:
    # 1. Has original link
    # 2. Has substantial content (> 10 lines)
    # Even if it has TODO marker
    is_complete = has_original_link and substantial_line_count > 10

    return {
        'total_lines': total_lines,
        'substantial_lines': substantial_line_count,
        'has_frontmatter': has_frontmatter,
        'has_original_link': has_original_link,
        'has_todo': has_todo,
        'is_complete': is_complete
    }

def main():
    """Check all files in blogs_raw directory"""

    if not BLOGS_RAW_DIR.exists():
        print(f"Error: {BLOGS_RAW_DIR} does not exist")
        return

    md_files = sorted(BLOGS_RAW_DIR.glob("*.md"))

    print(f"Checking {len(md_files)} markdown files in {BLOGS_RAW_DIR}/\n")
    print("=" * 80)

    complete_files = []
    incomplete_files = []

    for md_file in md_files:
        result = check_file_completeness(md_file)

        if result['is_complete']:
            complete_files.append((md_file.name, result))
        else:
            incomplete_files.append((md_file.name, result))

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files: {len(md_files)}")
    print(f"Complete files: {len(complete_files)} ({len(complete_files)*100//len(md_files)}%)")
    print(f"Incomplete files: {len(incomplete_files)} ({len(incomplete_files)*100//len(md_files)}%)")

    # Show incomplete files details
    if incomplete_files:
        print(f"\n{'=' * 80}")
        print(f"INCOMPLETE FILES ({len(incomplete_files)} files)")
        print(f"{'=' * 80}")

        for filename, result in incomplete_files[:20]:  # Show first 20
            print(f"\n{filename}")
            print(f"  Total lines: {result['total_lines']}")
            print(f"  Substantial lines: {result['substantial_lines']}")
            print(f"  Has original link: {result['has_original_link']}")
            print(f"  Has TODO: {result['has_todo']}")

        if len(incomplete_files) > 20:
            print(f"\n... and {len(incomplete_files) - 20} more incomplete files")

    # Show some complete files as examples
    if complete_files:
        print(f"\n{'=' * 80}")
        print(f"COMPLETE FILES (showing 10 examples)")
        print(f"{'=' * 80}")

        for filename, result in complete_files[:10]:
            print(f"\n{filename}")
            print(f"  Total lines: {result['total_lines']}")
            print(f"  Substantial lines: {result['substantial_lines']}")

    # Save incomplete files list
    if incomplete_files:
        output_file = Path("incomplete_files.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("INCOMPLETE FILES\n")
            f.write("=" * 80 + "\n\n")
            for filename, result in incomplete_files:
                f.write(f"{filename}\n")
                f.write(f"  Lines: {result['total_lines']}, Substantial: {result['substantial_lines']}\n")
                f.write(f"  Has link: {result['has_original_link']}, Has TODO: {result['has_todo']}\n\n")

        print(f"\n{'=' * 80}")
        print(f"Incomplete files list saved to: {output_file}")
        print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
