### 同LSTM序贯模型预测股价

#### 目录：

>* LSTM单步预测
>  * core #核心构建LSTM源码
>    * data_processor 用于加载数据，构建训练集和测试集
>    * model 用于构建模型，训练模型，模型预测
>  * data # 数据集
>  * saved_models # 模型保存目录
>  * config.json # 配置文件
>* LSTM多步预测
>  * core #核心构建LSTM源码
>    * data_processor 用于加载数据，构建训练集和测试集
>    * model 用于构建模型，训练模型，模型预测
>  * data # 数据集
>  * saved_models # 模型保存目录
>  * config.json # 配置文件

#### 1、LSTM单步预测

>配置文件config.json：
>
>```json
>{
>  "data": {
>    "filename": "sh600031.csv",
>    "columns": [
>      "close",
>      "volume"
>    ],
>    "sequence_length": 15,
>    "train_test_split": 0.85,
>    "normalise": true
>  },
>  "training": {
>    "epochs": 32,
>    "batch_size": 64,
>    "verbose": 1,
>    "shuffle": "False",
>    "validation_freq": 1
>  },
>  "model": {
>    "loss": "mse",
>    "optimizer": "adam",
>    "save_dir": "saved_models",
>    "layers": [
>      {
>        "type": "lstm",
>        "neurons": 100,
>        "return_seq": true
>      },
>      {
>        "type": "dropout",
>        "rate": 0.2
>      },
>      {
>        "type": "lstm",
>        "neurons": 100,
>        "return_seq": true
>      },
>      {
>        "type": "lstm",
>        "neurons": 100,
>        "return_seq": false
>      },
>      {
>        "type": "dropout",
>        "rate": 0.2
>      },
>      {
>        "type": "dense",
>        "neurons": 1,
>        "activation": "linear"
>      }
>    ]
>  }
>}
>```
>
>##### 主要设置data部分
>
>* 修改'columns'字段，设置对数据进行采样的特征值
>
>  注意的是特征值必须存在传入的训练数据中，否则报错
>
>* 修改'sequence_length'，设置数据序列的长度
>
>  实验将用sequence_length-1数量的数据进行训练
>
>  预测每间隔第sequence_length个数据
>
>* 修'normalise'，设置是否对数据进行归一化
>
>##### 其次可以修改training部分
>
>* 修改'epochs',设置训练次数，过多也不会生效，代码中采用了提前停止的功能
>* 修改'batch_size'，设置分支大小
>* 'verbose','shuffle','validation_freq'
>
>##### 如需自己添加神经网络层
>
>可以在model中的layers添加：
>
>```json
>{
>  "type": "lstm",
>  "neurons": 100,
>  "return_seq": true
>}
>```

#### 2、LSTM单步预测

>配置文件config.json：
>
>```json
>{
>  "data": {
>    "columns": [
>      "close",
>      "volume"
>    ],
>    "sequence_length": 30,
>    "input_timesteps": 25,
>    "input_dim": 2,
>    "train_test_split": 0.70,
>    "normalise": true
>  },
>  "training": {
>    "epochs": 32,
>    "batch_size": 64,
>    "verbose": 1,
>    "shuffle": "False",
>    "validation_freq": 1
>  },
>  "model": {
>    "loss": "mse",
>    "optimizer": "adam",
>    "save_dir": "saved_models",
>    "layers": [
>      {
>        "type": "lstm",
>        "neurons": 100,
>        "return_seq": true
>      },
>      {
>        "type": "dropout",
>        "rate": 0.2
>      },
>      {
>        "type": "lstm",
>        "neurons": 100,
>        "return_seq": true
>      },
>      {
>        "type": "lstm",
>        "neurons": 100,
>        "return_seq": false
>      },
>      {
>        "type": "dropout",
>        "rate": 0.2
>      },
>      {
>        "type": "dense",
>        "neurons": 5,
>        "activation": "linear"
>      }
>    ]
>  }
>}
>```
>
>#### 与单步预测不一样的地方：
>
>##### 主要设置：
>
>* 'sequence_length ':数据序列长度
>* 'input_timesteps':步进长度，即用input_timesteps数量的数据预测（sequence_length -input_timesteps）天的数据
>* 'input_dim'：特征量
>* 'neurons':最后输出层的神经元数量，保持等于（sequence_length -input_timesteps）

