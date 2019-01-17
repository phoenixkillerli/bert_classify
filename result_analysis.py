import codecs
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

label_list = ["其他侵入", "溜门", "攀爬翻窗和阳台", "暴力破锁", "技术开锁/插片开锁", "撬窗", "踹门撞门暴力破门", "翻墙", "砸窗"]


def read_csv(input_file):
    """Reads a tab separated value file."""
    with codecs.open(input_file, "r", "utf-8") as f:
        reader = csv.reader(f, quotechar='"', dialect='excel', delimiter=',')
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def report(pred_label, label):
    print(classification_report(label, pred_label))


def draw_map(pred_label, label, label_list):
    label_id = [label_list.index(x) for x in label]
    pred_label_id = [label_list.index(x) for x in pred_label]
    cfm = confusion_matrix(label_id, pred_label_id)
    # 查看中文字体
    # fc-list :lang=zh
    mpl.rcParams['axes.unicode_minus'] = False
    zh_font = FontProperties(fname='/usr/share/fonts/wps-office/FZWBK.TTF', size=15)

    # 设置画布大小
    plt.figure(figsize=(12, 8))

    plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('demo')
    plt.colorbar()
    xlocations = np.array(range(len(label_list)))
    plt.xticks(xlocations, label_list, rotation=90, fontproperties=zh_font)
    plt.yticks(xlocations, label_list, fontproperties=zh_font)
    plt.ylabel('实际分类', fontproperties=zh_font)
    plt.xlabel('预测分类', fontproperties=zh_font)

    tick_marks = np.array(range(len(label_list))) + 0.5

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    ind_array = np.arange(len(label_list))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cfm[y_val][x_val]
        if (c > 0.01):
            plt.text(x_val, y_val, c, color='red', fontsize=10, va='center', ha='center')
    plt.show()


if __name__ == '__main__':
    result = read_csv('./data/test_results.csv')
    pred_label = []
    label = []
    for x in result:
        pred_label.append(x[-1])
        label.append(x[-2])
    report(pred_label, label)
    draw_map(pred_label, label, label_list)
