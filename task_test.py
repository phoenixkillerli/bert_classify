import tensorflow as tf

import task


class TaskTest(tf.test.TestCase):

    def test_get_test_example(self):
        data_dir = './data'
        test_example = task.DataProcessor.get_test_examples(data_dir)
        for x in test_example:
            print(x.guid, x.text_a)


if __name__ == "__main__":
    tf.test.main()
