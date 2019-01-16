import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from common_tool import convert_single_example, InputExample, preprocess_txt
from task import DataProcessor

model_path = './saved_model/1547608578'


def restore_model(model_path):
    with tf.Session() as sess:
        # load model
        meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

        # get signature
        signature = meta_graph_def.signature_def

        # get tensor name
        in_tensor = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs
        out_tensor = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs
        for k, v in in_tensor.items():
            print(k, v)
            print(in_tensor[k].name)
            print('==>', sess.graph.get_tensor_by_name(in_tensor[k].name))
        for k, v in out_tensor.items():
            print(k, v)

        s = '2017年5月13日，安徽省阜南县黄岗镇发生一起入户盗窃案件。2017年5月13日12时05分，报案人（姓名杨新，身份证号码：342127197003104938；手机号码：18712675387，家庭住址：安徽省阜南县黄岗镇杨店村东店组28号）报案称：2016年12月27日12时许，在安徽省阜南县黄岗镇杨店村东店组28号杨新家，嫌疑人通过剪断钢筋防盗窗窗梁进到室内，经翻动将杨新家放在一楼东间卧室内一个三屉桌抽屉内的一百多元硬币、一张十元纸人民币和一张一半五元纸人民币盗走，将放在二楼客厅西边北侧一间卧室内三箱价值约一千元人民币的五粮醇白酒盗走。损失总价值1115元人民币。杨新家房屋后东侧靠楼梯的一个窗户的钢筋防盗窗窗梁被剪断三根，钢筋有折痕。现场勘验号：K3412255200002018050005。'
        case = InputExample(guid=None, text_a=preprocess_txt(s), text_b=None, label=DataProcessor.get_labels()[0])
        case = convert_single_example(case)
        input = {
            sess.graph.get_tensor_by_name(in_tensor['segment_ids'].name): [case.segment_ids],
            sess.graph.get_tensor_by_name(in_tensor['input_ids'].name): [case.input_ids],
            sess.graph.get_tensor_by_name(in_tensor['input_mask'].name): [case.input_mask],
            sess.graph.get_tensor_by_name(in_tensor['label_ids'].name): [[0] * 256],
        }

        output = sess.graph.get_tensor_by_name(out_tensor['probabilities'].name)

        # run
        conv_r = sess.run(output, feed_dict=input)
        print(conv_r)


if __name__ == '__main__':
    restore_model(model_path)
