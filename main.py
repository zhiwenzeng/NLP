import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir(r'utils')
from utils.myjieba import myjieba
from utils.textCnn import textCnn
from utils.const import _const as const
from utils.data import text2data
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

m = myjieba()
model = textCnn().model

def mhelp():
    print(r'''
        -h 显示帮助文档
        -f input_file.txt 第一个参数是新闻内容，输出标签
        -d input_dir out_file 第一个参数是目录，输出out_file标签，输出是一个json文件和一个.csv文件
    ''')


def txt(path):
    file = open(path, 'r', encoding='utf-8')
    lines = file.readlines()
    content = ""
    for line in lines:
        content = content + ' '+ line.strip()
    res = m.seg_depart(content)
    x = [res.strip().split(' ')]
    x = text2data(x)
    return x


def predict(x):
    y = model.predict(x)[0]
    return y, const.cnn.to_labels[y.argmax()]


def predicts(dir):
    dir = os.path.abspath(dir)
    filenames = [os.path.join(dir, filename) for filename in os.listdir(dir) if '.txt' == filename[-4:]]
    size = len(filenames)
    cnt = size // 100 + 1
    if cnt == 0:
        cnt = 1
    ys = []
    print("start...")
    for i in range(size):
        x = txt(filenames[i])
        ys.append(x[0])
        if i % cnt == 0:
            print("已完成: %.2f%%" % (i/cnt))
    print("end...")
    ys = model.predict(np.array(ys)).tolist()
    res = np.argmax(ys, axis=1)
    labels = list(map(lambda i: const.cnn.to_labels[i], res))
    return filenames, ys, labels


def main():
    argv = sys.argv[1: ]
    size = len(argv)
    if size == 0 or argv[0] not in ['-f', '-d']:
        mhelp()
        return
    if argv[0] == '-f' and size == 2:
        x = txt(argv[1])
        p, y = predict(x)
        fig = plt.figure('预测结果', figsize=(6, 4), dpi=100, )
        axes = fig.add_subplot(111)
        x = range(len(const.cnn.labels))
        rect = axes.bar(x, p, width=0.5)
        axes.set_xticks(x)
        axes.set_xticklabels(const.cnn.labels, fontsize=14)
        # 编辑文本
        for r in rect:
            height = r.get_height()
            axes.text(r.get_x() + r.get_width() / 2, height, "%.3f" % height, fontsize=18, ha="center", va="bottom")
        axes.legend([y], loc='upper left')
        plt.show()
    elif argv[0] == '-d' and size == 3:
        filenames, ps, ys = predicts(argv[1])
        df = pd.DataFrame({'filename': filenames, 'probability': ps, 'label': ys})
        vs = pd.Series(ys).value_counts()
        counts = []
        labels = []
        for label, count in vs.items():
            counts.append(count)
            labels.append(label + ':' + str(count))
        fig = plt.figure(figsize=(6, 4), dpi=100)
        axes = fig.add_subplot(111)
        axes.pie(counts, labels=labels, autopct='%1.1f%%', fontsize=12)
        plt.show()
        outfile = argv[2]
        if len(outfile) > 4 and '.csv' == outfile[-4: ]:
            df.to_csv(outfile)
        else:
            df.to_csv(outfile+'.csv')


if __name__ == "__main__":
    sys.exit(main())