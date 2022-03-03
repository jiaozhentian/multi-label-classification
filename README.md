# 多标签分类

## 算法网络结构

使用tensorflow自带MobileNetV2作为主干网络，最后使用GAP代替全连接进行输出，输出层激活函数为sigmoid，使用二进制交叉熵作为loss，二进制准确度作为监督。

## 环境

python >=3.8.7

```shell
pip install requirement.txt
```

## 训练

入参

| 参数名称          | 用途                 | 默认值   |
| ------------- | ------------------ | ----- |
| batch_size    | 规定训练的batch size    | 32    |
| epochs        | 训练多少个epochs        | 10    |
| learning_rate | 学习率                | 0.001 |
| gpu           | 使用那块卡进行训练          | 0     |
| dataset_id    | data文件夹下数据集放在哪个id下 | 0     |

执行命令

```shell
python main.py
```
