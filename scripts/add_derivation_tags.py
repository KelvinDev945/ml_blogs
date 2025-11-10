#!/usr/bin/env python3
"""
为已完成数学推导的文章添加"详细推导"标签

此脚本会：
1. 检测文章中是否包含"## 公式推导与注释"部分
2. 如果包含，自动添加"详细推导"标签到frontmatter
3. 保留原有标签
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BLOGS_RAW_DIR = PROJECT_ROOT / "blogs_raw"

# 已完成数学推导的文章列表（按批次）
COMPLETED_ARTICLES = [
    # 第一批：概率/随机矩阵（7篇）
    "n个正态随机数的最大值的渐近估计.md",
    "随机矩阵的谱范数的快速估计.md",
    "概率分布的熵归一化.md",
    "从重参数的角度看离散概率分布的构建.md",
    "圆内随机n点在同一个圆心角为θ的扇形的概率.md",
    "通向概率分布之路盘点softmax及其替代品.md",
    "用傅里叶级数拟合一维概率密度函数.md",

    # 第二批：矩阵理论（12篇）
    "矩阵r次方根和逆r次方根的高效计算.md",
    "矩阵平方根和逆平方根的高效计算.md",
    "对角低秩三角阵的高效求逆方法.md",
    "通过msign来计算奇异值裁剪mclip上.md",
    "通过msign来计算奇异值裁剪mclip下.md",
    "矩阵符号函数mcsgn能计算什么.md",
    "msign的导数.md",
    "svd的导数.md",
    "矩阵的有效秩effective-rank.md",
    "低秩近似之路一伪逆.md",
    "低秩近似之路二svd.md",
    "低秩近似之路三cr.md",

    # 第三批：优化理论（15篇）
    "初探mup超参数的跨模型尺度迁移规律.md",
    "高阶mup更简明但更高明的谱条件缩放.md",
    "从谱范数梯度到新式权重衰减的思考.md",
    "muon优化器赏析从向量到矩阵的本质跨越.md",
    "muon续集为什么我们选择尝试muon.md",
    "为什么梯度裁剪的默认模长是1.md",
    "重新思考学习率与batch-size二平均场.md",
    "为什么adam的update-rms是02.md",
    "流形上的最速下降1-sgd-超球面.md",
    "流形上的最速下降2-muon-正交.md",
    "adamw的weight-rms的渐近估计.md",
    "从hessian近似看自适应学习率优化器.md",
    "moe环游记3换个思路来分配.md",
    "通过梯度近似寻找normalization的替代品.md",
    "重新思考学习率与batch-size一现状.md",

    # 第四批：扩散模型（15篇）
    "生成扩散模型漫谈一ddpm-拆楼-建楼.md",
    "生成扩散模型漫谈三ddpm-贝叶斯-去噪.md",
    "生成扩散模型漫谈四ddim-高观点ddpm.md",
    "生成扩散模型漫谈五一般框架之sde篇.md",
    "生成扩散模型漫谈六一般框架之ode篇.md",
    "生成扩散模型漫谈十八得分匹配-条件得分匹配.md",
    "生成扩散模型漫谈二十八分步理解一致性模型.md",
    "测试函数法推导连续性方程和fokker-planck方程.md",
    "生成扩散模型漫谈十四构建ode的一般步骤上.md",
    "生成扩散模型漫谈十五构建ode的一般步骤中.md",
    "生成扩散模型漫谈十七构建ode的一般步骤下.md",
    "生成扩散模型漫谈二十一中值定理加速ode采样.md",
    "生成扩散模型漫谈二十二信噪比与大图生成上.md",
    "生成扩散模型漫谈十六w距离-得分匹配.md",
    "生成扩散模型漫谈十三从万有引力到扩散模型.md",

    # 第五批：Transformer/Attention（18篇）
    "transformer升级之路6旋转位置编码的完备性分析.md",
    "transformer升级之路7长度外推性与局部注意力.md",
    "transformer升级之路8长度外推性与位置鲁棒性.md",
    "transformer升级之路9一种全局长度外推的新思路.md",
    "transformer升级之路10rope是一种β进制编码.md",
    "transformer升级之路11将β进制位置进行到底.md",
    "transformer升级之路12无限外推的rerope.md",
    "transformer升级之路13逆用leaky-rerope.md",
    "transformer升级之路14当hwfa遇见rerope.md",
    "transformer升级之路15key归一化助力长度外推.md",
    "transformer升级之路16复盘长度外推技术.md",
    "transformer升级之路17多模态位置编码的简单思考.md",
    "transformer升级之路18rope的底数选择原则.md",
    "transformer升级之路20mla好在哪里上.md",
    "transformer升级之路21mla好在哪里下.md",
    "为什么线性注意力要加short-conv.md",
    "低精度attention可能存在有偏的舍入误差.md",
    "相对位置编码transformer的一个理论缺陷与对策.md",

    # 第六批：梯度分析与训练技巧（18篇）
    "vq的旋转技巧梯度直通估计的一般推广.md",
    "vq的又一技巧给编码表加一个线性变换.md",
    "vq一下keytransformer的复杂度就变成线性了.md",
    "简单得令人尴尬的fsq四舍五入超越了vq-vae.md",
    "diveq一种非常简洁的vq训练方案.md",
    "梯度视角下的lora简介分析猜测及推广.md",
    "对齐全量微调这是我看过最精彩的lora改进一.md",
    "对齐全量微调这是我看过最精彩的lora改进二.md",
    "配置不同的学习率lora还能再涨一点.md",
    "流形上的最速下降3-muon-stiefel.md",
    "流形上的最速下降4-muon-谱球面.md",
    "流形上的最速下降5-对偶梯度下降.md",
    "moe环游记2不患寡而患不均.md",
    "moe环游记4难处应当多投入.md",
    "softmax后传寻找top-k的光滑近似.md",
    "adam的epsilon如何影响学习率的scaling-law.md",
    "当batch-size增大时学习率该如何随之变化.md",
    "通向最优分布之路概率空间的最小化.md",
]


def parse_frontmatter(content):
    """解析markdown文件的frontmatter"""
    frontmatter = {}
    body = content

    match = re.match(r'^---\s*\n(.*?\n)---\s*\n(.*)', content, re.DOTALL)
    if match:
        fm_text = match.group(1)
        body = match.group(2)

        for line in fm_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                frontmatter[key.strip()] = value.strip()

    return frontmatter, body


def has_derivation_section(content):
    """检查文章是否包含公式推导部分"""
    return '## 公式推导与注释' in content


def add_derivation_tag(file_path, dry_run=False):
    """为文章添加"详细推导"标签"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已有推导部分
    if not has_derivation_section(content):
        return False, "无推导部分"

    frontmatter, body = parse_frontmatter(content)

    # 获取现有tags
    tags_str = frontmatter.get('tags', '')
    existing_tags = [t.strip() for t in tags_str.split(',') if t.strip()]

    # 检查是否已有"详细推导"标签
    if '详细推导' in existing_tags:
        return False, "已有标签"

    # 添加"详细推导"标签（放在最前面）
    new_tags = ['详细推导'] + existing_tags
    new_tags_str = ', '.join(new_tags)

    # 更新frontmatter
    if not dry_run:
        # 重新构建frontmatter
        fm_lines = []
        fm_lines.append('---')

        # 保持原有顺序，但更新tags
        if content.startswith('---'):
            match = re.match(r'^---\s*\n(.*?\n)---\s*\n', content, re.DOTALL)
            if match:
                fm_text = match.group(1)
                for line in fm_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        if key == 'tags':
                            fm_lines.append(f'tags: {new_tags_str}')
                        else:
                            fm_lines.append(f'{key}: {value.strip()}')
                    elif line.strip():
                        fm_lines.append(line)

        fm_lines.append('---')

        # 写回文件
        new_content = '\n'.join(fm_lines) + '\n' + body
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    return True, f"添加标签: {' | '.join(new_tags)}"


def main(dry_run=False):
    """主函数"""
    print(f"{'[预览模式] ' if dry_run else ''}为已完成推导的文章添加标签...\n")

    success_count = 0
    skipped_count = 0
    not_found_count = 0

    for filename in COMPLETED_ARTICLES:
        file_path = BLOGS_RAW_DIR / filename

        if not file_path.exists():
            print(f"❌ 未找到文件: {filename}")
            not_found_count += 1
            continue

        success, message = add_derivation_tag(file_path, dry_run=dry_run)

        if success:
            print(f"✓ {filename}")
            print(f"  {message}")
            success_count += 1
        else:
            print(f"⊙ {filename} - {message}")
            skipped_count += 1

    print(f"\n{'[预览模式] ' if dry_run else ''}统计:")
    print(f"  成功添加标签: {success_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  未找到: {not_found_count}")
    print(f"  总计: {len(COMPLETED_ARTICLES)}")

    if dry_run:
        print("\n提示: 使用 --apply 参数来实际应用更改")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='为已完成数学推导的文章添加标签')
    parser.add_argument('--apply', action='store_true', help='实际应用更改（默认为预览模式）')

    args = parser.parse_args()

    main(dry_run=not args.apply)
