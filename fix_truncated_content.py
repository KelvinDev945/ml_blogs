#!/usr/bin/env python3
"""
Fix files with truncated original content
"""

import re
from pathlib import Path
import shutil

BLOGS_RAW_DIR = Path("blogs_raw")

# Files with issues and their potential source files
TRUNCATED_FILES = [
    {
        "problem_file": "adamw的weight-rms的渐近估计.md",
        "source_file": "adamw的weight-rms的.md",
        "url": "https://spaces.ac.cn/archives/11307"
    },
]

def extract_original_content(source_file):
    """Extract original blog content from source file"""
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the derivation section
    derivation_match = re.search(r'## 公式推导与注释', content)

    if derivation_match:
        # Extract content before derivation section
        original_end = derivation_match.start()
    else:
        # No derivation section, use entire content after frontmatter
        original_end = len(content)

    # Remove frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            body_start = content.find(parts[2])
            original_content = content[body_start:original_end]
        else:
            original_content = content[:original_end]
    else:
        original_content = content[:original_end]

    return original_content.strip()

def extract_derivation_content(problem_file):
    """Extract derivation content from problem file"""
    with open(problem_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the derivation section
    derivation_match = re.search(r'## 公式推导与注释', content)

    if derivation_match:
        # Extract derivation content
        derivation_content = content[derivation_match.start():]
        return derivation_content.strip()
    else:
        return ""

def extract_frontmatter(file_path):
    """Extract frontmatter from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            return f"---{parts[1]}---"

    return ""

def fix_file(problem_file, source_file):
    """Fix truncated file by combining original content and derivation"""

    problem_path = BLOGS_RAW_DIR / problem_file
    source_path = BLOGS_RAW_DIR / source_file

    if not problem_path.exists():
        print(f"  ✗ Problem file not found: {problem_file}")
        return False

    if not source_path.exists():
        print(f"  ✗ Source file not found: {source_file}")
        return False

    # Create backup
    backup_path = problem_path.with_suffix('.md.backup2')
    shutil.copy(problem_path, backup_path)
    print(f"  ✓ Created backup: {backup_path.name}")

    # Extract components
    frontmatter = extract_frontmatter(problem_path)
    original_content = extract_original_content(source_path)
    derivation_content = extract_derivation_content(problem_path)

    # Check for truncation marker
    if '[[[...]]]' in original_content or '[[...]]' in original_content:
        print(f"  ⚠️  Warning: Source file also contains truncation marker")
        return False

    # Combine
    new_content = f"{frontmatter}\n\n{original_content}\n\n---\n\n{derivation_content}\n"

    # Remove truncation markers if any
    new_content = re.sub(r'\[\[\[?\.\.\.\]?\]\].*?\n', '', new_content)

    # Write fixed file
    with open(problem_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # Verify
    original_lines = len([l for l in original_content.split('\n') if l.strip()])
    derivation_lines = len([l for l in derivation_content.split('\n') if l.strip()])

    print(f"  ✓ Fixed: {problem_file}")
    print(f"    Original content: {original_lines} lines")
    print(f"    Derivation content: {derivation_lines} lines")

    return True

def main():
    """Fix all truncated files"""

    print("修复截断的博客内容...\n")
    print("=" * 100)

    success_count = 0
    fail_count = 0

    for item in TRUNCATED_FILES:
        problem_file = item['problem_file']
        source_file = item['source_file']

        print(f"\n处理: {problem_file}")
        print(f"  源文件: {source_file}")

        if fix_file(problem_file, source_file):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 100}")
    print(f"总结:")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"{'=' * 100}")

if __name__ == "__main__":
    main()
