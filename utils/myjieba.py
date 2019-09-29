import jieba
import re
from utils.const import _const as const


class myjieba():

    def __init__(self):
        self.stopwords = self.__stopwordslist()

    # 创建停用词列表
    def __stopwordslist(self):
        stopwords = [line.strip() for line in open(const.jieba.stopwords, encoding='UTF-8').readlines()]
        return stopwords

    # 文本是否是数字或者是百分点
    def __is_number(self, text):
        size = len(text)
        if text[size - 1] == '%':
            text = text[:-1]
            size -= 1
        i = 0
        while i < size:
            if '0' > text[i] or text[i] > '9':
                if text[i] == '.':
                    i += 1
                    break
                return False
            i += 1
        while i < size:
            if '0' > text[i] or text[i] > '9':
                return False
            i += 1
        return True

    # 对句子进行中文分词
    def seg_depart(self, sentence):
        # 对文档中的每一行进行中文分词
        sentence_depart = jieba.cut(sentence.strip())
        # 输出结果为outstr
        outstr = ''
        # 去停用词
        for word in sentence_depart:
            if self.__is_number(word):
                continue
            if re.match("\s+", word):
                continue
            if word not in self.stopwords:
                outstr += word
                outstr += " "
        return outstr