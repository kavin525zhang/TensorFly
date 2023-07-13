import re
import codecs
import jieba
from jieba.analyse import *
import math

pattern = "^\d*\[\d+\]"

def is_chinese(line):
    """判断一个unicode是否是汉字"""
    count = 0
    for word in line:
        if '\u4e00' <= word <= '\u9fff':
            count += 1
    if count >= 2:
        return True
    return False

def compute_r(n_k, l, m=9):
    return n_k * math.exp(n_k/m)/(m * min(l, 50)) * max(l-20, 0)/l

for filename in ["CN101919099A"]:
    with codecs.open("./patent_without_abstract/{}.txt".format(filename), "r", "utf-8") as fr:
        with codecs.open("./patent_without_abstract_clean/{}.txt".format(filename), "w", "utf-8") as fw:
            for line in fr:
                line = line.strip()
                if not line or not is_chinese(line):
                    continue
                line = re.sub(pattern, "", line)
                if compute_r(len(textrank(line, withWeight=True)), len(list(jieba.cut(line)))) >= 0.37:
                    fw.write(line + "\n")
