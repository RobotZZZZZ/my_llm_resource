"""
@author : Hyunwoong
@when : 2019-12-22
@homepage : https://github.com/gusdnd852
"""
import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if stats[0] == 0 or stats[1] == 0:  # 检查hypothesis和reference长度
        return 0
    
    # 检查是否所有n-gram匹配都为0
    n_gram_matches = stats[2::2]  # 提取所有n-gram匹配数
    if all(match == 0 for match in n_gram_matches):
        return 0
    
    (c, r) = stats[:2]
    
    # 计算精确度，避免除零错误
    precisions = []
    for match, total in zip(stats[2::2], stats[3::2]):
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(float(match) / total)
    
    # 如果所有精确度都为0，返回0
    if all(p == 0 for p in precisions):
        return 0
    
    # 计算几何平均
    log_bleu_prec = sum([math.log(p) for p in precisions if p > 0]) / len([p for p in precisions if p > 0])
    
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def idx_to_word(x, vocab):
    """
    将索引序列转换为单词字符串，处理各种边界情况
    """
    words = []
    special_tokens = {'<pad>', '<unk>', '<sos>', '<eos>'}
    
    # 处理空输入
    if not x:
        return ""
    
    for i in x:
        # 检查索引是否为有效整数
        if not isinstance(i, (int, np.integer)):
            continue
            
        # 检查索引是否在有效范围内
        if 0 <= i < len(vocab.itos):
            word = vocab.itos[i]
            # 只过滤特殊的token，不过滤所有包含'<'的词
            if word not in special_tokens:
                words.append(word)
        # 对于无效索引，跳过（不添加任何词）
        
    return " ".join(words)
