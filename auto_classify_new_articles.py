#!/usr/bin/env python3
"""
自动为新文章分类
根据文件名和文章内容中的关键词，将未分类的文章归类到合适的主题
"""

import json
from pathlib import Path
import re

# 主题关键词（按优先级排序）
TOPIC_KEYWORDS = {
    "扩散模型": {
        "keywords": ["扩散模型", "ddpm", "ddim", "扩散", "diffusion", "得分匹配", "score matching",
                    "ode", "sde", "reflow", "一致性模型", "consistency", "fokker-planck"],
        "priority": 10
    },
    "Transformer": {
        "keywords": ["transformer", "attention", "注意力", "rope", "位置编码", "positional",
                    "外推", "mla", "gqa", "mha", "mqa", "flash", "softmax", "gau"],
        "priority": 9
    },
    "优化理论": {
        "keywords": ["优化器", "optimizer", "adam", "sgd", "muon", "lion", "tiger", "adafactor",
                    "学习率", "learning rate", "batch size", "梯度下降", "momentum", "amos"],
        "priority": 8
    },
    "矩阵理论": {
        "keywords": ["矩阵", "matrix", "svd", "低秩", "low-rank", "msign", "特征值", "eigenvalue",
                    "奇异值", "singular", "伪逆", "pseudoinverse", "正交", "orthogonal", "monarch",
                    "newton-schulz", "hippo"],
        "priority": 7
    },
    "随机矩阵/概率": {
        "keywords": ["概率", "probability", "随机", "random", "正态", "gaussian", "分布", "distribution",
                    "期望", "expectation", "方差", "variance", "贝叶斯", "bayes", "统计"],
        "priority": 6
    },
    "损失函数": {
        "keywords": ["损失", "loss", "交叉熵", "cross entropy", "kl", "emo", "对比学习",
                    "contrastive", "cosent", "circle loss"],
        "priority": 5
    },
    "梯度分析": {
        "keywords": ["梯度", "gradient", "反向传播", "backprop", "梯度裁剪", "gradient clip",
                    "梯度惩罚", "gradient penalty", "lora", "微调", "fine-tuning"],
        "priority": 4
    },
    "RNN/SSM": {
        "keywords": ["rnn", "lstm", "gru", "ssm", "state space", "s4", "线性注意力",
                    "linear attention", "hippo", "mamba"],
        "priority": 3
    },
    "VQ/量化": {
        "keywords": ["vq", "vector quantization", "量化", "quantization", "编码", "codebook",
                    "fsq", "离散", "discrete"],
        "priority": 2
    },
    "语言模型": {
        "keywords": ["语言模型", "language model", "llm", "bert", "gpt", "t5", "pegasus",
                    "预训练", "pretrain", "tokenizer", "embedding", "decoder"],
        "priority": 1
    },
}

def normalize_text(text):
    """规范化文本：转小写，去除特殊字符"""
    text = text.lower()
    text = re.sub(r'[_\-]', ' ', text)
    return text

def classify_article(slug, content_sample=""):
    """
    根据slug和内容样本对文章进行分类
    返回: (主题, 置信度分数)
    """
    text = normalize_text(slug + " " + content_sample)

    # 统计每个主题的匹配分数
    topic_scores = {}

    for topic, config in TOPIC_KEYWORDS.items():
        score = 0
        matched_keywords = []

        for keyword in config["keywords"]:
            keyword_normalized = normalize_text(keyword)
            if keyword_normalized in text:
                # 关键词匹配得分
                keyword_score = len(keyword.split())  # 多词关键词得分更高
                score += keyword_score
                matched_keywords.append(keyword)

        # 优先级加权
        if score > 0:
            score *= config["priority"]
            topic_scores[topic] = {
                "score": score,
                "matched_keywords": matched_keywords
            }

    # 返回得分最高的主题
    if topic_scores:
        best_topic = max(topic_scores.items(), key=lambda x: x[1]["score"])
        return best_topic[0], best_topic[1]

    return "其他", {"score": 0, "matched_keywords": []}

def read_article_preview(filepath, lines=50):
    """读取文章前几行作为预览"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            preview = ''.join(f.readlines()[:lines])
        return preview
    except Exception as e:
        print(f"⚠️  无法读取文件 {filepath}: {e}")
        return ""

def auto_classify_new_articles():
    """自动分类所有未分类的文章"""

    # 读取现有分类
    with open('topic_classification.json', 'r', encoding='utf-8') as f:
        classified = json.load(f)

    # 获取已分类文章的slug
    classified_slugs = set()
    max_number = 0
    for topic_articles in classified.values():
        for article in topic_articles:
            classified_slugs.add(article['slug'])
            max_number = max(max_number, article.get('number', 0))

    # 获取所有文章
    all_files = list(Path('blogs_raw').glob('*.md'))

    # 找出未分类的文章
    unclassified_files = [f for f in all_files if f.stem not in classified_slugs]

    print(f"发现 {len(unclassified_files)} 篇未分类文章")
    print("=" * 100)

    # 准备新分类
    new_classifications = {topic: [] for topic in TOPIC_KEYWORDS.keys()}
    new_classifications["其他"] = []

    # 分类每篇文章
    for i, filepath in enumerate(unclassified_files, 1):
        slug = filepath.stem

        # 读取文章预览以提高分类准确性
        preview = read_article_preview(filepath)

        # 分类
        topic, info = classify_article(slug, preview)

        # 创建文章条目
        article_entry = {
            "title": slug.replace('-', ' ').replace('_', ' '),  # 简单处理标题
            "slug": slug,
            "number": max_number + i,  # 分配新的编号
            "matched_keywords": info["matched_keywords"][:5],  # 保留前5个匹配的关键词
            "confidence": info["score"]
        }

        new_classifications[topic].append(article_entry)

        # 打印进度
        if i % 10 == 0:
            print(f"进度: {i}/{len(unclassified_files)} ({i*100//len(unclassified_files)}%)")

    # 打印分类结果摘要
    print("\n" + "=" * 100)
    print("分类结果摘要:")
    print("=" * 100)

    for topic, articles in sorted(new_classifications.items(), key=lambda x: -len(x[1])):
        if articles:
            print(f"\n📚 {topic}: {len(articles)} 篇")
            # 显示前3篇作为示例
            for article in articles[:3]:
                keywords_str = ", ".join(article["matched_keywords"][:3])
                print(f"  • {article['slug'][:60]}... (关键词: {keywords_str})")
            if len(articles) > 3:
                print(f"  ... 还有 {len(articles) - 3} 篇")

    # 保存详细分类报告
    with open('new_articles_classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(new_classifications, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("✅ 详细分类报告已保存到: new_articles_classification_report.json")

    # 询问是否合并到主分类文件
    print("\n是否要将新分类合并到 topic_classification.json?")
    print("这将自动执行，或者你可以手动审核 new_articles_classification_report.json 后再合并。")

    return new_classifications, classified

def merge_classifications(new_classifications, existing_classifications):
    """合并新分类到现有分类"""

    merged = dict(existing_classifications)

    for topic, articles in new_classifications.items():
        if topic not in merged:
            merged[topic] = []

        # 移除临时字段
        clean_articles = []
        for article in articles:
            clean_article = {
                "title": article["title"],
                "slug": article["slug"],
                "number": article["number"]
            }
            clean_articles.append(clean_article)

        merged[topic].extend(clean_articles)

    # 按number排序每个主题的文章
    for topic in merged:
        merged[topic] = sorted(merged[topic], key=lambda x: x.get('number', 0))

    return merged

if __name__ == "__main__":
    new_classifications, existing = auto_classify_new_articles()

    print("\n" + "=" * 100)
    response = input("是否合并到 topic_classification.json? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        merged = merge_classifications(new_classifications, existing)

        # 备份原文件
        import shutil
        shutil.copy('topic_classification.json', 'topic_classification.json.backup')
        print("✅ 已备份原文件到: topic_classification.json.backup")

        # 保存合并后的分类
        with open('topic_classification.json', 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print("✅ 新分类已合并到 topic_classification.json")

        # 打印统计
        print("\n" + "=" * 100)
        print("更新后的统计:")
        print("=" * 100)
        total = 0
        for topic, articles in merged.items():
            count = len(articles)
            total += count
            print(f"{topic}: {count} 篇")
        print(f"\n总计: {total} 篇文章")
    else:
        print("❌ 未合并。请手动审核 new_articles_classification_report.json")
