import codecs
import csv
import os
import re

import tokenization

# 提取特征相关子句（处理长度超过max_seq_length）
pattern = r'([^，。；：）\]、]*?(?:门|窗|锁|撬|墙|阳台|挑|水管|爬|撬|踹|砸|钥匙|顺手|[^警案害疑]人|家[^中属]|玻璃).*?[，。；：）\]、])'


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# 数据格式(csv)
# id, texta, [textb], ... , label
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        return self._create_examples(DataProcessor._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            DataProcessor._read_csv(os.path.join(data_dir, "valid.csv")),
            "valid")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            DataProcessor._read_csv(os.path.join(data_dir, "test.csv")), "test")

    @staticmethod
    def get_labels():
        return ["暴力破锁", "其他侵入", "翻窗", "暴力破锁", "技术开锁插片开锁", "溜门", "特征不明显",
                "踹门撞门暴力破门", "翻墙", "撬窗", "砸窗"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = ''.join(re.findall(pattern, line[1]))
            if len(text_a) < 3:
                text_a = '特征不明显'
            text_a = re.sub(r'\d+', '', tokenization.convert_to_unicode(text_a))
            text_b = tokenization.convert_to_unicode(line[2]) if len(line) > 3 else None
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _read_csv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", "utf-8") as f:
            reader = csv.reader(f, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
