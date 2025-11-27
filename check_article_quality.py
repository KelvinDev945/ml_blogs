#!/usr/bin/env python3
"""检查Summary覆盖的文章质量"""
import os
from pathlib import Path
import re

blogs_raw_dir = Path('/home/kelvin/dev/ml_posts/blogs_raw')

# 定义已有Summary覆盖的文章模式（基于文件名）
summary_patterns = {
    '扩散模型': ['生成扩散模型漫谈'],
    '矩阵理论': ['msign', 'svd', '低秩近似', '矩阵', 'monarch', 'hippo'],
    'RNN/SSM': ['重温ssm', 's4', 'mamba', 'rnn'],
    '优化器': ['adamw', 'adam', 'lion', 'tiger', 'muon', '优化器', '学习率', '梯度'],
    'Transformer/Attention': ['transformer升级之路', 'attention', 'flash', 'gau', 'rope', '位置编码'],
    '概率统计': ['viterbi', '概率', '贝叶斯', 'softmax', '熵'],
    '损失函数': ['cosent', 'globalpointer', 'emo', 'can', '损失'],
    'BERT/预训练': ['bert', 'roformer', '预训练'],
}

def get_article_info(filepath):
    """获取文章信息"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        # 获取基本信息
        info = {
            'filename': filepath.name,
            'size_kb': filepath.stat().st_size / 1024,
            'line_count': len(lines),
            'has_yaml': content.startswith('---'),
            'has_theorem_box': '<div class="theorem-box">' in content or 'theorem-box' in content,
            'has_derivation_box': '<div class="derivation-box">' in content or 'derivation-box' in content,
            'has_formula': '$$' in content or '$\\' in content,
            'formula_count': content.count('$$') // 2,
            'status': 'unknown',
        }

        # 提取status
        status_match = re.search(r'status:\s*(\w+)', content)
        if status_match:
            info['status'] = status_match.group(1)

        # 提取title
        title_match = re.search(r'title:\s*(.+)', content)
        if title_match:
            info['title'] = title_match.group(1)
        else:
            info['title'] = filepath.stem

        return info
    except Exception as e:
        return None

# 收集所有文章并分类
all_articles = {}
categorized = {topic: [] for topic in summary_patterns.keys()}
uncategorized = []

for md_file in blogs_raw_dir.glob('*.md'):
    info = get_article_info(md_file)
    if info:
        all_articles[md_file.name] = info

        # 分类
        filename_lower = md_file.name.lower()
        matched = False
        for topic, patterns in summary_patterns.items():
            for pattern in patterns:
                if pattern.lower() in filename_lower:
                    categorized[topic].append(info)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            uncategorized.append(info)

print("="*80)
print("📊 Summary覆盖文章的质量检查报告")
print("="*80)
print()

# 统计各主题的文章质量
for topic, articles in categorized.items():
    if not articles:
        continue

    print(f"\n## {topic}主题 ({len(articles)}篇)")
    print("-" * 80)

    # 统计质量指标
    avg_size = sum(a['size_kb'] for a in articles) / len(articles)
    avg_lines = sum(a['line_count'] for a in articles) / len(articles)
    avg_formulas = sum(a['formula_count'] for a in articles) / len(articles)

    has_structure = sum(1 for a in articles if a['has_theorem_box'] or a['has_derivation_box'])
    has_formulas = sum(1 for a in articles if a['formula_count'] > 0)

    print(f"\n平均指标:")
    print(f"  - 平均大小: {avg_size:.1f} KB")
    print(f"  - 平均行数: {avg_lines:.0f} 行")
    print(f"  - 平均公式数: {avg_formulas:.1f} 个")
    print(f"  - 有结构化内容: {has_structure}/{len(articles)} ({has_structure/len(articles)*100:.1f}%)")
    print(f"  - 有公式推导: {has_formulas}/{len(articles)} ({has_formulas/len(articles)*100:.1f}%)")

    # 按大小排序，显示前5篇和后5篇
    sorted_articles = sorted(articles, key=lambda x: x['size_kb'], reverse=True)

    print(f"\n📈 最详细的5篇:")
    for i, article in enumerate(sorted_articles[:5], 1):
        quality_indicator = ""
        if article['has_theorem_box']:
            quality_indicator += "📦"
        if article['formula_count'] > 10:
            quality_indicator += "🔢"
        if article['size_kb'] > 50:
            quality_indicator += "📚"

        print(f"  {i}. {article['filename'][:60]}")
        print(f"     {article['size_kb']:.1f}KB, {article['line_count']}行, {article['formula_count']}公式 {quality_indicator}")

    print(f"\n📉 最简略的5篇:")
    for i, article in enumerate(sorted_articles[-5:], 1):
        print(f"  {i}. {article['filename'][:60]}")
        print(f"     {article['size_kb']:.1f}KB, {article['line_count']}行, {article['formula_count']}公式")

print("\n\n" + "="*80)
print("📊 整体质量统计")
print("="*80)

total_covered = sum(len(articles) for articles in categorized.values())
print(f"\nSummary覆盖的文章总数: {total_covered}篇")
print(f"未覆盖的文章数: {len(uncategorized)}篇")
print(f"总文章数: {len(all_articles)}篇")

# 计算整体质量分布
all_covered = [a for articles in categorized.values() for a in articles]

size_ranges = [
    (0, 10, "极小 (<10KB)"),
    (10, 30, "较小 (10-30KB)"),
    (30, 50, "中等 (30-50KB)"),
    (50, 100, "较大 (50-100KB)"),
    (100, float('inf'), "超大 (>100KB)")
]

print(f"\n文件大小分布:")
for min_size, max_size, label in size_ranges:
    count = sum(1 for a in all_covered if min_size <= a['size_kb'] < max_size)
    pct = count / len(all_covered) * 100 if all_covered else 0
    bar = "█" * int(pct / 5)
    print(f"  {label:20s}: {count:3d}篇 ({pct:5.1f}%) {bar}")

print(f"\n结构化程度:")
structured = sum(1 for a in all_covered if a['has_theorem_box'] or a['has_derivation_box'])
print(f"  有结构化标记: {structured}/{len(all_covered)} ({structured/len(all_covered)*100:.1f}%)")

has_formulas = sum(1 for a in all_covered if a['formula_count'] > 0)
print(f"  有公式推导: {has_formulas}/{len(all_covered)} ({has_formulas/len(all_covered)*100:.1f}%)")

print(f"\n公式数量分布:")
formula_ranges = [
    (0, 0, "无公式"),
    (1, 5, "少量公式 (1-5)"),
    (6, 20, "中等公式 (6-20)"),
    (21, 50, "较多公式 (21-50)"),
    (51, float('inf'), "大量公式 (>50)")
]

for min_f, max_f, label in formula_ranges:
    if max_f == 0:
        count = sum(1 for a in all_covered if a['formula_count'] == 0)
    else:
        count = sum(1 for a in all_covered if min_f <= a['formula_count'] <= max_f)
    pct = count / len(all_covered) * 100 if all_covered else 0
    bar = "█" * int(pct / 5)
    print(f"  {label:20s}: {count:3d}篇 ({pct:5.1f}%) {bar}")

# 识别质量问题
print(f"\n\n" + "="*80)
print("⚠️  质量问题识别")
print("="*80)

print(f"\n可能需要扩展的文章（<15KB且公式<5个）:")
needs_expansion = [a for a in all_covered
                   if a['size_kb'] < 15 and a['formula_count'] < 5]

if needs_expansion:
    needs_expansion.sort(key=lambda x: x['size_kb'])
    for i, article in enumerate(needs_expansion[:20], 1):
        print(f"  {i}. {article['filename'][:65]}")
        print(f"     {article['size_kb']:.1f}KB, {article['formula_count']}公式")
    if len(needs_expansion) > 20:
        print(f"  ... 还有{len(needs_expansion)-20}篇")
else:
    print("  ✅ 所有文章质量良好！")

print(f"\n\n高质量文章（>50KB且公式>20个）:")
high_quality = [a for a in all_covered
                if a['size_kb'] > 50 and a['formula_count'] > 20]

if high_quality:
    high_quality.sort(key=lambda x: x['size_kb'], reverse=True)
    for i, article in enumerate(high_quality[:10], 1):
        print(f"  {i}. {article['filename'][:65]}")
        print(f"     {article['size_kb']:.1f}KB, {article['formula_count']}公式")
else:
    print("  ⚠️  没有特别详细的高质量文章")
