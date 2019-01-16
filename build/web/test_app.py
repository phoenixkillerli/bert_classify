import unittest
from common.bert_util import *
import requests
import operator
from name_address.app import app


class TestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.content = [
            '2018年10月28日12时30分许，河北红旗水泥有限公司赵平原报案称：当日10时30分许，该到本厂位于平涉路路南的新厂配电室查看线路情况时，发现配电室内两个变压器（1996年购买，800千瓦变压器价值80000元左右，630千瓦变压器价值60000元左右，都未在使用中，型号品牌不详）、八块大闸（1996年购买，价值10000元左右，型号品牌不详）被盗，总价值约150000元。',
            '2018年10月1日5时许，郭会肖起床后发现放在大门洞内的电动二轮车、200袋面粉、两箱挂面被盗，被盗的电车是黑红色名马牌的，一年前花1800元购买，被盗的面粉价值1400元，被盗的挂面价值200元。'
        ]
        self.url = 'http://192.168.1.64:8501/v1/models/name_address:predict'

    def tearDown(self):
        pass

    def test_predict_ner(self):
        result = predict_ner(self.content, self.url)
        print(result)
        assert operator.eq(result[0]['name'], ['赵平原'])

    def test_http_suinterface(self):
        url = 'http://192.168.1.64:5050/name_address'
        response = requests.post(url, data=json.dumps(self.content))
        print(response.json())

    def test_split_content(self):
        result = split_short_content(self.content)
        print(result)
        assert len(result[0]) == 3

    def test_predict(self):
        result = predict(self.content, self.url)
        assert operator.eq(result[1]['name'], ['郭会肖'])
        print(result)

