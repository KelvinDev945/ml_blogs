#!/usr/bin/env python3
"""分析未被Summary覆盖的文章，识别可归类的主题"""
import os
from pathlib import Path
from collections import defaultdict

# 获取所有文章
blogs_raw_dir = Path('/home/kelvin/dev/ml_posts/blogs_raw')
all_articles = sorted([f.name for f in blogs_raw_dir.glob('*.md')])

print(f"总文章数: {len(all_articles)}\n")

# 已有的8个主题Summary覆盖的文章（根据Summary总览.md）
covered_by_summary = {
    '扩散模型': 24,  # 生成扩散模型漫谈系列
    '矩阵理论': 13,  # SVD、msign、HiPPO等
    '优化器': 16,    # Adam、Lion、Muon、Tiger等
    'Transformer/Attention': 14,  # Attention、位置编码等
    '概率统计': 10,  # 贝叶斯、Viterbi等
    '损失函数': 9,   # CoSENT、GlobalPointer等
    '语言模型': 8,   # BERT、Decoder-only等
    'RNN/SSM': 6,    # S4、Mamba等
}

total_covered = sum(covered_by_summary.values())
print(f"已有Summary覆盖: {total_covered}篇\n")
print("各主题覆盖情况:")
for topic, count in covered_by_summary.items():
    print(f"  - {topic}: {count}篇")

# 手动识别已覆盖的文章（基于文件名模式）
covered_patterns = {
    '生成扩散模型漫谈': '扩散模型',
    'transformer升级之路': 'Transformer/Attention',
    'msign': '矩阵理论',
    'svd': '矩阵理论',
    '低秩近似': '矩阵理论',
    '矩阵': '矩阵理论',
    'monarch': '矩阵理论',
    'hippo': 'RNN/SSM',
    'ssm': 'RNN/SSM',
    's4': 'RNN/SSM',
    'mamba': 'RNN/SSM',
    '重温ssm': 'RNN/SSM',
    'adamw': '优化器',
    'adam': '优化器',
    'lion': '优化器',
    'tiger': '优化器',
    'muon': '优化器',
    '优化器': '优化器',
    '学习率': '优化器',
    '梯度': '优化器',
    'cosent': '损失函数',
    'globalpointer': '损失函数',
    'emo': '损失函数',
    'can': '损失函数',
    '损失': '损失函数',
    'bert': '语言模型',
    'decoder-only': '语言模型',
    'llm': '语言模型',
    '语言模型': '语言模型',
    'embedding': '语言模型',
    'viterbi': '概率统计',
    '概率': '概率统计',
    '贝叶斯': '概率统计',
    'softmax': '概率统计',
    '熵': '概率统计',
    'attention': 'Transformer/Attention',
    'flash': 'Transformer/Attention',
    'gau': 'Transformer/Attention',
    'rope': 'Transformer/Attention',
    '位置编码': 'Transformer/Attention',
}

# 分类所有文章
categorized = defaultdict(list)
uncategorized = []

for article in all_articles:
    article_lower = article.lower()
    matched = False

    for pattern, topic in covered_patterns.items():
        if pattern.lower() in article_lower:
            categorized[topic].append(article)
            matched = True
            break

    if not matched:
        uncategorized.append(article)

print(f"\n\n{'='*80}")
print("已有Summary主题的文章分布（基于文件名识别）:")
print(f"{'='*80}\n")

for topic in covered_by_summary.keys():
    if topic in categorized:
        print(f"\n## {topic} ({len(categorized[topic])}篇)")
        for i, article in enumerate(categorized[topic][:5], 1):
            print(f"  {i}. {article}")
        if len(categorized[topic]) > 5:
            print(f"  ... 还有{len(categorized[topic])-5}篇")

print(f"\n\n{'='*80}")
print(f"未被现有Summary覆盖的文章 ({len(uncategorized)}篇)")
print(f"{'='*80}\n")

# 分析未覆盖的文章，识别可归类的主题
theme_keywords = {
    'MoE': ['moe'],
    'VQ/量化': ['vq', 'fsq', 'diveq', '量化'],
    'LoRA/微调': ['lora', 'childtuning', 'ladder-side-tuning', '微调', '对齐'],
    '归一化': ['norm', 'normalization', '归一化', 'whitening'],
    '激活函数': ['relu', 'gelu', 'swish', 'squareplus', '激活'],
    '多任务学习': ['多任务学习'],
    '多模态': ['多模态'],
    '初始化': ['初始化'],
    '残差/网络结构': ['残差', 'deepnet', '1000层'],
    '数学理论': ['不等式', '概率', '定理', '证明', '推导', '恒等式', '估计', '球', '数集'],
    'Tokenization': ['tokenizer', 'bytepiece', '分词'],
    '信息抽取': ['gplinker', 'kgclue', 'seq2seq', '抽取'],
    'VAE/生成模型': ['vae', 'gan', 'wgan', 'ign', 'flow'],
    'Scaling Law/μP': ['scaling-law', 'mup', '尺度定律'],
    '训练技巧': ['batch-size', 'warmup', 'ema', '混合精度', 'xla', '训练'],
    '相似度/检索': ['相似度', 'cur分解', '检索', 'hubness'],
    '正则化/稳定性': ['dropout', '权重衰减', '梯度裁剪', '谱范数'],
    '其他NLP': ['roformerv2', 'nbce', 'naive-bayes'],
}

theme_groups = defaultdict(list)
truly_uncategorized = []

for article in uncategorized:
    article_lower = article.lower()
    matched = False

    for theme, keywords in theme_keywords.items():
        for keyword in keywords:
            if keyword.lower() in article_lower:
                theme_groups[theme].append(article)
                matched = True
                break
        if matched:
            break

    if not matched:
        truly_uncategorized.append(article)

# 按文章数排序
sorted_themes = sorted(theme_groups.items(), key=lambda x: len(x[1]), reverse=True)

print("可归类为新主题的文章群:\n")

for theme, articles in sorted_themes:
    if len(articles) >= 3:  # 至少3篇才值得成为主题
        print(f"### {theme} ({len(articles)}篇)")
        for article in articles[:10]:
            print(f"  - {article}")
        if len(articles) > 10:
            print(f"  ... 还有{len(articles)-10}篇")
        print()

print(f"\n{'='*80}")
print(f"完全零散的文章 ({len(truly_uncategorized)}篇)")
print(f"{'='*80}\n")
for i, article in enumerate(truly_uncategorized[:20], 1):
    print(f"{i}. {article}")
if len(truly_uncategorized) > 20:
    print(f"... 还有{len(truly_uncategorized)-20}篇")

# 统计建议
print(f"\n\n{'='*80}")
print("主题归类建议统计")
print(f"{'='*80}\n")

print(f"总文章数: {len(all_articles)}")
print(f"已有Summary覆盖（估算）: ~{total_covered}篇")
print(f"未覆盖文章: ~{len(uncategorized)}篇")
print()

print("可组成新主题Summary的文章群（≥3篇）:")
new_summary_articles = sum(len(articles) for theme, articles in sorted_themes if len(articles) >= 3)
print(f"  文章数: {new_summary_articles}篇")
print(f"  主题数: {sum(1 for theme, articles in sorted_themes if len(articles) >= 3)}个")
print()

print(f"零散文章（需要单独处理或归入'其他技术'）: {len(truly_uncategorized)}篇")
