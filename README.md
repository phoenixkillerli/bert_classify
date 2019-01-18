#### 基于bert的分类模型
根据自己的实际情况，对bert的代码进行了清理重构
* 删除了tpu相关配置及代码(工作中暂时用不到)
* 对代码进行了清理，方便更好的理解代码
* 为所有配置参数设置了默认值，方便运行代码
* 增加了saved_model模型保存及调用测试代码
* 编写了基于docker的镜像编译及部署脚本，方便一条命令执行模型部署
* 增加了测试结果数据的分析显示，方便进行结果分析

#### 使用方法
* data 目录下保存训练，验证，测试数据
    * train.csv
    * valid.csv
    * test.csv
* 数据格式为：id, content, label
* task.py是数据读取处理脚本
* common_tool.py是公共函数包，为方便服务打包最小依赖而独立出来
* model 目录下存放google发布的中文预训练模型，有以下几个文件
    * bert_config.json
    * bert_model.ckpt.data-00000-of-00001
    * bert_model.ckpt.index
    * bert_model.ckpt.meta
    * vocab.txt
* saved_model 目录保存训练后生成的saved_model模型供服务部署使用
* output 目录保存训练，测试过程中的输出
* 执行方法：
    * python run_classifier.py
    * 如需修改参数可以参考run.sh脚本内容，
* 模型发布：
    * build.sh
    * 该脚本生成两个docker镜像
        * 一个是tensorflow_serving模型推断服务
        * 一个是flask的web接口服务

#### 后续工作
* 修改flask web接口实现为生产模式
* 增加ner模型训练脚本
    
    