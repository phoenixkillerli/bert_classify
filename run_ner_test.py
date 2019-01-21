import tensorflow as tf
from run_ner import NerProcessor, convert_single_example
import tokenization

class NerProcessorTest(tf.test.TestCase):
    def test_get_test_example(self):
        result = NerProcessor().get_test_examples('./data/ner')
        for x in result[:3]:
            print(x.guid, x.text, x.label, '\n')

    def test_convert_single_example(self):
        example = NerProcessor().get_test_examples('./data/ner')
        features = convert_single_example(example[0], 512, tokenization.FullTokenizer())
        print(features.input_ids)
        print(features.input_mask)
        print(features.segment_ids)
        print(features.label_ids)

if __name__ == "__main__":
    tf.test.main()
