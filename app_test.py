import unittest
import operator
import json
from app import app, _predict


class TestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.content = [
            '2018年6月6日7时许芜湖市鸠江区湾里镇发生一起盗窃案件。2018年6月6日7时54分，报警人（江庭友，男，1966年07月22日出生，汉族，初中文化程度，户籍所在地安徽省芜湖市鸠江区窑区3排307号，现住安徽省芜湖市鸠江区窑区3排307号，现在个体工作，居民身份证号码340223196607224117，联系电话13705530371）来我所报案称：大概在2018年6月5日23时30分将裤子放在房间沙发上，沙发摆放在窗子边上，到2018年6月6日0时左右从浴室出来发现沙发上的一条藏蓝色西裤，裤子口袋内有一个钱包（钱包内有3740元左右，面值是37张100元和一些10元、20元零钱以及三张来壹龙自助餐卡，价值2000元左右、一张本人身份证、一本驾驶证、银行卡、钥匙等物品）不见了，后来在窗子旁边发现一根长铁丝。作案手段为“钓鱼”盗窃，拟作刑事案件。',
            '2018年5月14日15时许在宿州市埇桥区西关办事处丰收巷88号发生一起入室盗窃案件。2018年5月14日15时许接报警（报警人：王秀华，身份证号：342201196502201246，家庭住址：宿州市埇桥区水木清华8栋2单元704，联系方式：15005571989）称：其于2017年10月19日上午8时许其从宿州市埇桥区西关办事处丰收巷88号家中离开，家中门窗以上锁，后于2018年5月14日11时许回到家中拿东西时发现，其家大门被破坏，其放在家中二楼仓库内的白酒被移动到一楼走道内，白酒未被盗，被损坏大门为铁门，价值400余元。'
        ]
        self.url = 'http://localhost:8501/v1/models/case_type:predict'

    def tearDown(self):
        pass

    def test_predict(self):
        result = _predict(self.content, self.url)
        assert operator.eq(result, ['其他侵入', '踹门撞门暴力破门'])

    def test_http_suinterface(self):
        response = self.app.post('/predict', data=json.dumps(self.content))
        assert operator.eq(response.json(), ['其他侵入', '踹门撞门暴力破门'])

