# 记录表

| 序号  |    模型     | 是否使用预训练 |      预处理       |                    数据增强                    | 训练轮数 |   准确率   | F1-Score |  比赛得分  |
|:---:|:---------:|:-------:|:--------------:|:------------------------------------------:|:----:|:-------:|:--------:|:------:|
|  1  | EcapaTdnn |    是    | melspectrogram |            随机裁剪<br>音量正确<br>语速增强            |  30  | 0.9679  |  0.9678  | 0.4667 |
|  2  | EcapaTdnn |    否    | melspectrogram |            随机裁剪<br>音量正确<br>语速增强            |  30  | 0.9691  |  0.9681  | 0.4500 |
|  3  | EcapaTdnn |    否    | melspectrogram |                    随机裁剪                    |  30  | 0.9961  |  0.9958  | 0.4333 |
|  4  | EcapaTdnn |    是    | melspectrogram |                    随机裁剪                    |  30  | 0.9943  |  0.9941  | 0.4567 |
|  5  | EcapaTdnn |    是    | melspectrogram |    随机裁剪<br>音量正确<br>语速增强<br>SpecAugment     |  30  | 0.58868 |  0.5421  |        |
|  6  | EcapaTdnn |    是    | melspectrogram | 随机裁剪<br>音量正确<br>语速增强<br>SpecAugment(无频率屏蔽) |  30  | 0.56554 |  0.4990  |   	    |
|  7  | EcapaTdnn |    否    |  spectrogram   |                    随机裁剪                    |  30  | 0.9935  |  0.9933  |        |
|  8  | EcapaTdnn |    否    |  spectrogram   |            随机裁剪<br>音量正确<br>语速增强            |  30  | 0.9458  |  0.9445  |        |
|  9  | EcapaTdnn |    否    |  spectrogram   |    随机裁剪<br>音量正确<br>语速增强<br>SpecAugment     |  30  | 0.5633  |  0.5076  |        |
| 10  | EcapaTdnn |    是    |  spectrogram   |            随机裁剪<br>音量正确<br>语速增强            |  30  | 0.9911  |  0.9914  |        |
