import codecs
import csv
import os

import tokenization
from common_tool import InputExample, label_list, preprocess_txt


# 数据格式(csv)
# id, texta, [textb], ... , label
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @staticmethod
    def get_train_examples(data_dir):
        return DataProcessor._create_examples(DataProcessor._read_csv(os.path.join(data_dir, "train.csv")), "train")

    @staticmethod
    def get_dev_examples(data_dir):
        return DataProcessor._create_examples(
            DataProcessor._read_csv(os.path.join(data_dir, "valid.csv")),
            "valid")

    @staticmethod
    def get_test_examples(data_dir):
        return DataProcessor._create_examples(
            DataProcessor._read_csv(os.path.join(data_dir, "test.csv")), "test")

    @staticmethod
    def get_labels():
        return label_list

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = preprocess_txt(line[1])
            text_b = tokenization.convert_to_unicode(line[2]) if len(line) > 3 else None
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _read_csv(input_file):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", "utf-8") as f:
            reader = csv.reader(f, quotechar='"', dialect='excel', delimiter=',')
            lines = []
            for line in reader:
                lines.append(line)
            return lines
