### QA-RNN说明 ###

#### 环境说明： ####

python = 3.6.3

tensorflow=1.14

#### 数据集说明： ####

**文件的形式：**

```reStructuredText
ID<空格>text
ID<空格>text
ID<空格>question<tab>answer<tab>supporting fact IDS.
```

ID是给定的每一个“故事”并且从1开始，在文件中若是ID又从1开始，那么这表示另一个“故事”。

**例如：**

```text
1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom        1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway 4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel?      hallway 4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom        8
1 Sandra travelled to the office.
2 Sandra went to the bathroom.
3 Where is Sandra?      bathroom        2
```



#### 训练模型： ####

```bash
python QAtrain_weights.py --data_path=./tasks_1-20_v1-2/en/ --model=small --ifcontinue=False --data_type=1
```

data_path: 是指数据集的路径

model:  是训练模型的大小，可以选择small、medium和large

ifcontinue:：第一次训练时设置为False，再次训练时设置为True

data_type：选择哪一种数据作为训练的模型，这里选择第一种(范围是1-20)

#### 测试模型: ####

data_type：和训练设置的一致
==注意：==首次预测，加载模型比较慢

```bash
python QAanswer.py --data_path=./tasks_1-20_v1-2/en/ --model=small --data_type=1
```

#### 结果展示： ####

```bash
# 开始输入文档和问题
输入文档和问题(输入问题后空行回车):
John went back to the bathroom.
Mary travelled to the bathroom.
Where is John?
(此处是一个空行,用来区分问题,直接回车)

# 回答问题
try to answer
The Question answer is:  bathroom
```

