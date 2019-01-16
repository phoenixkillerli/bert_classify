import json

from common.bert_util import *
from flask import Flask, Response, request


class MyResponse(Response):
    default_mimetype = 'application/json;charset=utf-8'


class MyFlask(Flask):
    response_class = MyResponse


app = MyFlask(__name__)


# 提取姓名、地址接口
# 参数格式：['case1', 'case2']
# 返回结果：[{'name':['name1', 'name2'], 'addr':['addr1', 'addr2']}, {'name':['name1', 'name2'], 'addr':['addr1', 'addr2']}]
@app.route('/name_address', methods=['POST'])
def parse_name_address():
    param = json.loads(request.data)
    result = predict(param)
    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
