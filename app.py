import json

import requests
from flask import Flask, Response, request

from common_tool import *


class MyResponse(Response):
    default_mimetype = 'application/json;charset=utf-8'


class MyFlask(Flask):
    response_class = MyResponse


app = MyFlask(__name__)
SERVER_URL = 'http://model:8501/v1/models/case_type:predict'


# 安徽案件小类推断接口
# 参数格式：['case1', 'case2']
# 返回结果：['case type1', 'case type2']
@app.route('/predict', methods=['POST'])
def predict_case_type():
    param = json.loads(request.data)
    result = _predict(param)
    return json.dumps(result, ensure_ascii=False)


def _predict(cases, url=None):
    if url is None:
        url = SERVER_URL
    features = [convert_single_example(InputExample('0', preprocess_txt(case))) for case in cases]
    data = [{'input_ids': x.input_ids,
             'input_mask': x.input_mask,
             'segment_ids': x.input_ids,
             'label_ids': [0] * 256
             } for x in features]
    response = requests.post(url, data=json.dumps({'instances': data}))
    response.raise_for_status()
    result = []
    for x in response.json()['predictions']:
        result.append(label_list[x.index(max(x))])
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
