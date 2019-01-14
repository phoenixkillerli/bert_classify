import csv
import os

import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
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
        return ["报案人", "受害人", "嫌疑人", "警察"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2]) if len(line) > 3 else None
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines