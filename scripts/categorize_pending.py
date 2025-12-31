#!/usr/bin/env python3
"""Categorize pending blogs by theme."""
import json
from collections import defaultdict

# Load blog list
with open('/home/kelvin/dev/ml_posts/docs/data/blog_list.json', 'r', encoding='utf-8') as f:
    blogs = json.load(f)

# Get pending blogs with post numbers
pending = [b for b in blogs if b.get('status') == 'pending' and b.get('post_number') is not None]
pending.sort(key=lambda x: x['post_number'], reverse=True)

# Define theme keywords and categories
themes = {
    '扩散模型': ['扩散', 'diffusion', 'ddpm', 'ddim', 'ode', 'sde', 'flow', 'score'],
    'Transformer/Attention': ['transformer', 'attention', 'rope', 'mla', 'mha', 'gqa', 'mqa', '位置编码', '长度外推'],
    '优化器/学习率': ['优化器', 'adam', 'sgd', 'muon', '学习率', 'batch size', 'optimizer', 'mup'],
    'MoE': ['moe', '专家', 'expert', '路由'],
    '矩阵理论': ['矩阵', 'matrix', 'svd', 'msign', '特征值', '奇异值', '低秩'],
    '概率/统计': ['概率', '随机', '正态', '分布', 'probability', '统计', '期望'],
    'SSM/RNN': ['ssm', 'rnn', 'hippo', 'state space', '线性系统'],
    'VQ/量化': ['vq', '量化', 'quantization', 'tokenizer'],
    'LoRA/微调': ['lora', '微调', 'fine-tune', 'peft'],
    '损失函数': ['损失', 'loss', 'softmax', '交叉熵', 'margin'],
    '多模态': ['多模态', 'multimodal', '视觉', 'vision'],
    '数学基础': ['不等式', '恒等式', '定理', '证明', '推导'],
    '其他': []  # Default category
}

# Categorize blogs
categorized = defaultdict(list)
for blog in pending:
    title = blog.get('title', '').lower()
    tags = [t.lower() for t in blog.get('tags', [])]
    text = title + ' ' + ' '.join(tags)

    matched = False
    for theme, keywords in themes.items():
        if theme == '其他':
            continue
        if any(kw in text for kw in keywords):
            categorized[theme].append(blog)
            matched = True
            break

    if not matched:
        categorized['其他'].append(blog)

# Print categorization
print("=" * 80)
print("待完成博客主题分类（共{}篇）".format(len(pending)))
print("=" * 80)
print()

for theme in themes.keys():
    if theme not in categorized or len(categorized[theme]) == 0:
        continue

    blogs_in_theme = categorized[theme]
    print(f"\n### {theme}：{len(blogs_in_theme)}篇")
    print("-" * 80)
    for b in blogs_in_theme[:20]:  # Show max 20 per category
        num = b.get('post_number', 'N/A')
        title = b.get('title', 'Untitled')
        date = b.get('date', 'N/A')
        print(f"  * #{num:3} - {title} ({date})")
    if len(blogs_in_theme) > 20:
        print(f"  ... 以及其他 {len(blogs_in_theme) - 20} 篇")

# Summary
print("\n" + "=" * 80)
print("主题统计汇总")
print("=" * 80)
for theme, blogs_list in sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True):
    if len(blogs_list) > 0:
        print(f"{theme:20s}: {len(blogs_list):3d} 篇")
print("-" * 80)
print(f"{'总计':20s}: {len(pending):3d} 篇")
