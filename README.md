# 记录表

| 序号  |                                              模型                                               |               数据增强                | 训练轮数 |  准确率   | F1-Score |  比赛得分  |
|:---:|:---------------------------------------------------------------------------------------------:|:---------------------------------:|:----:|:------:|:--------:|:------:|
|  1  | Linear(256)<br>LSTM(256,forward)<br>Tanh<br>Dropout(0.5)<br>Linear(256)<br>ReLU<br>Linear(6)  |               随机裁剪                | 100  |        |          |        |
|  2  | Linear(256)<br>LSTM(256,forward)<br>Tanh<br>Dropout(0.5)<br>Linear(256)<br>ReLU<br>Linear(6)  |       随机裁剪<br>音量增强<br>语速增强        | 100  | 0.5968 |  0.5267  | 0.5083 |
|  3  | Linear(256)<br>LSTM(256,forward)<br>Tanh<br>Dropout(0.5)<br>Linear(256)<br>ReLU<br>Linear(6)  |       随机裁剪<br>音量增强<br>语速增强        | 200  | 0.6583 |  0.6124  | 0.5067 |
|  4  | Linear(512)<br>LSTM(256,bidirect)<br>Tanh<br>Dropout(0.5)<br>Linear(256)<br>ReLU<br>Linear(6) |       随机裁剪<br>音量增强<br>语速增强        | 100  |        |          |        |
|  5  | Linear(512)<br>LSTM(256,bidirect)<br>Tanh<br>Dropout(0.5)<br>Linear(256)<br>ReLU<br>Linear(6) | 随机裁剪<br>音量增强<br>语速增强<br>(specaug) | 100  |        |          |        |
|  6  | Linear(512)<br>LSTM(256,bidirect)<br>Tanh<br>Dropout(0.5)<br>Linear(256)<br>ReLU<br>Linear(6) | 随机裁剪<br>音量增强<br>语速增强<br>(specaug) | 500  | 0.9207 |  0.9200  | 0.5250 |
