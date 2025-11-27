#!/usr/bin/env python3
"""分析待扩充文章的主题分布"""

import re
from pathlib import Path
from collections import defaultdict

def categorize_article(filename):
    """根据文件名判断主题"""
    name_lower = filename.lower()

    if any(x in name_lower for x in ['优化器', 'adam', 'lion', 'muon', 'tiger', 'optimizer']):
        return '优化器'
    elif any(x in name_lower for x in ['transformer', 'attention', 'gau', '位置编码', 'rope', 'decoder']):
        return 'Transformer/Attention'
    elif any(x in name_lower for x in ['扩散', 'ddpm', 'ddim', 'diffusion']):
        return '扩散模型'
    elif any(x in name_lower for x in ['低秩', 'svd', 'cr', 'id', '矩阵']):
        return '矩阵/低秩近似'
    elif any(x in name_lower for x in ['损失', 'loss', 'emo', 'can']):
        return '损失函数'
    elif any(x in name_lower for x in ['msign', 'relu', 'gelu', 'swish', 'squareplus']):
        return '激活函数/数学'
    elif any(x in name_lower for x in ['多任务', 'multi-task', '梯度']):
        return '多任务学习'
    elif any(x in name_lower for x in ['gplinker', 'globalpointer', 'clue', 'bert4keras']):
        return '实用工具/NLP任务'
    elif any(x in name_lower for x in ['bytepiece', 'tokenizer']):
        return 'Tokenizer'
    elif any(x in name_lower for x in ['batch-size', '学习率', 'learning-rate', 'pre-norm', 'post-norm', '残差']):
        return '训练技巧'
    else:
        return '其他'

def main():
    blogs_dir = Path('/home/kelvin/dev/ml_posts/blogs_raw')

    pending_files = []

    for md_file in blogs_dir.glob('*.md'):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                first_lines = ''.join([next(f) for _ in range(10)])

            if 'status: pending' in first_lines:
                pending_files.append(md_file.name)
        except:
            pass

    # 按主题分类
    categories = defaultdict(list)
    for filename in pending_files:
        category = categorize_article(filename)
        categories[category].append(filename)

    # 输出统计
    print("=== 待扩充文章主题分布 ===\n")
    for category, files in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"【{category}】({len(files)}篇)")
        for f in sorted(files)[:5]:  # 只显示前5篇
            print(f"  - {f}")
        if len(files) > 5:
            print(f"  ... 还有{len(files)-5}篇")
        print()

    print(f"\n总计：{len(pending_files)}篇待扩充文章")

if __name__ == '__main__':
    main()
