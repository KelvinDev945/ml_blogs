#!/usr/bin/env python3
"""
Fix all files with truncated original content
"""

import re
from pathlib import Path
import shutil

BLOGS_RAW_DIR = Path("blogs_raw")

# Map problem files to their source files (if they exist)
FILE_MAPPINGS = [
    # Files with complete source versions
    ("adamw的weight-rms的渐近估计.md", "adamw的weight-rms的.md"),
    ("为什么线性注意力要加short-conv.md", "为什么线性注意力要加short-c.md"),
    ("低精度attention可能存在有偏的舍入误差.md", "低精度attention可能存在有.md"),
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
            body_start = content.find(parts[2])
            original_content = content[body_start:original_end]
        else:
            original_content = content[:original_end]
    else:
        original_content = content[:original_end]

    # Remove truncation markers
    original_content = re.sub(r'\[\[\.{3,}\]\]\([^)]+\)', '', original_content)
    original_content = re.sub(r'\[\[…\]\]\([^)]+\)', '', original_content)

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

    # Check if source also has truncation
    if '[[...' in original_content or '[[…' in original_content:
        print(f"  ⚠️  Warning: Source file also contains truncation marker")
        # Try to clean it up anyway
        original_content = re.sub(r'\[\[\.{3,}\]\]\([^)]+\)', '', original_content)
        original_content = re.sub(r'\[\[…\]\]\([^)]+\)', '', original_content)

    # Combine
    new_content = f"{frontmatter}\n\n{original_content}\n\n---\n\n{derivation_content}\n"

    # Final cleanup
    new_content = re.sub(r'\n\n\n+', '\n\n', new_content)

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

    print("修复所有截断的博客内容...\n")
    print("=" * 100)

    success_count = 0
    fail_count = 0

    for problem_file, source_file in FILE_MAPPINGS:
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
    print(f"{'=' * 100}\n")

    # Re-run content check
    print("\n重新检查内容完整性...")
    import subprocess
    subprocess.run(["python3", "check_original_content.py"])

if __name__ == "__main__":
    main()
