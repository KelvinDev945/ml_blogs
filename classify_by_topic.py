#!/usr/bin/env python3
"""
Classify files by topic for adding derivations
"""

import re
from pathlib import Path

BLOGS_RAW_DIR = Path("blogs_raw")

# Topic keywords for classification
TOPICS = {
    "扩散模型": ["扩散", "ddpm", "ddim", "reflow", "一致性", "得分匹配", "ode", "sde"],
    "Transformer/Attention": ["transformer", "attention", "rope", "位置编码", "外推", "mla", "gqa", "mha"],
    "优化器": ["adam", "sgd", "muon", "lion", "优化器", "学习率", "batch size", "梯度"],
    "矩阵理论": ["矩阵", "svd", "低秩", "msign", "特征值", "奇异值", "伪逆"],
    "概率统计": ["概率", "正态", "分布", "期望", "方差", "贝叶斯", "随机"],
    "损失函数": ["损失", "交叉熵", "kl", "softmax", "emo"],
    "语言模型": ["llm", "bert", "gpt", "decoder", "embedding", "语言模型"],
    "VQ/编码": ["vq", "编码", "离散", "quantization"],
    "RNN/SSM": ["rnn", "ssm", "hippo", "线性注意力"],
    "其他": []  # 默认分类
}

def get_file_topic(filename):
    """Determine file topic based on filename"""
    filename_lower = filename.lower()

    for topic, keywords in TOPICS.items():
        if topic == "其他":
            continue
        for keyword in keywords:
            if keyword in filename_lower:
                return topic

    return "其他"

def classify_files():
    """Classify all files needing derivations by topic"""

    # Read files needing derivations
    with open("files_need_derivation.txt", 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract filenames
    lines = content.split('\n')
    files_no_derivation = []
    files_simple_derivation = []

    in_no_section = False
    in_simple_section = False

    for line in lines:
        if "无推导部分" in line:
            in_no_section = True
            in_simple_section = False
            continue
        elif "简单推导需要增强" in line:
            in_no_section = False
            in_simple_section = True
            continue
        elif line.startswith("---"):
            continue
        elif line.strip() == "":
            continue

        if in_no_section and line.strip().endswith(".md"):
            files_no_derivation.append(line.strip())
        elif in_simple_section and line.strip():
            # Extract filename from "filename.md - N 行" format
            parts = line.strip().split(" - ")
            if parts and parts[0].endswith(".md"):
                files_simple_derivation.append(parts[0])

    # Classify by topic
    topic_classification = {topic: {"no_derivation": [], "simple_derivation": []}
                           for topic in TOPICS.keys()}

    for filename in files_no_derivation:
        topic = get_file_topic(filename)
        topic_classification[topic]["no_derivation"].append(filename)

    for filename in files_simple_derivation:
        topic = get_file_topic(filename)
        topic_classification[topic]["simple_derivation"].append(filename)

    return topic_classification

def main():
    """Main classification function"""

    classification = classify_files()

    print("=" * 100)
    print("按主题分类的文件列表")
    print("=" * 100)

    total_no = 0
    total_simple = 0

    for topic, files in classification.items():
        no_count = len(files["no_derivation"])
        simple_count = len(files["simple_derivation"])
        total = no_count + simple_count

        if total == 0:
            continue

        total_no += no_count
        total_simple += simple_count

        print(f"\n{'━' * 100}")
        print(f"📚 主题: {topic}")
        print(f"{'━' * 100}")
        print(f"总计: {total} 个文件 (无推导: {no_count}, 简单推导: {simple_count})")

        if files["no_derivation"]:
            print(f"\n❌ 无推导部分 ({no_count} 个):")
            for i, filename in enumerate(files["no_derivation"], 1):
                print(f"  {i}. {filename}")

        if files["simple_derivation"]:
            print(f"\n⚠️  简单推导需增强 ({simple_count} 个):")
            # Show first 10 files
            for i, filename in enumerate(files["simple_derivation"][:10], 1):
                print(f"  {i}. {filename}")
            if simple_count > 10:
                print(f"  ... 还有 {simple_count - 10} 个文件")

    print(f"\n{'=' * 100}")
    print(f"总计统计")
    print(f"{'=' * 100}")
    print(f"无推导部分: {total_no} 个文件")
    print(f"简单推导需增强: {total_simple} 个文件")
    print(f"总计: {total_no + total_simple} 个文件")
    print(f"{'=' * 100}")

    # Save detailed classification
    with open("files_by_topic.txt", 'w', encoding='utf-8') as f:
        f.write("按主题分类的详细文件列表\n")
        f.write("=" * 100 + "\n\n")

        for topic, files in classification.items():
            total = len(files["no_derivation"]) + len(files["simple_derivation"])
            if total == 0:
                continue

            f.write(f"\n{'━' * 100}\n")
            f.write(f"主题: {topic}\n")
            f.write(f"{'━' * 100}\n")
            f.write(f"总计: {total} 个文件\n\n")

            if files["no_derivation"]:
                f.write(f"无推导部分 ({len(files['no_derivation'])} 个):\n")
                for filename in files["no_derivation"]:
                    f.write(f"  - {filename}\n")
                f.write("\n")

            if files["simple_derivation"]:
                f.write(f"简单推导需增强 ({len(files['simple_derivation'])} 个):\n")
                for filename in files["simple_derivation"]:
                    f.write(f"  - {filename}\n")
                f.write("\n")

    print(f"\n详细分类已保存至: files_by_topic.txt")

if __name__ == "__main__":
    main()
