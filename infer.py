import argparse
import functools
import os

import numpy as np
import paddle
from tqdm import tqdm

from modules.ecapa_tdnn import EcapaTdnn
from data_utils.reader import load_audio, CustomDataset
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_dir',        str,    'dataset/Data_A_F',               '需要识别的音频文件夹')
add_arg('use_model',        str,    'ecapa_tdnn',                     '所使用的模型')
add_arg('num_class',        int,    6,                                '分类的类别数量')
add_arg('label_list_path',  str,    'dataset/label_list.txt',         '标签列表路径')
add_arg('model_path',       str,    'output/models/model.pdparams',   '模型保存的路径')
add_arg('feature_method',   str,    'melspectrogram',          '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
args = parser.parse_args()

train_dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取分类标签
with open(args.label_list_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
class_labels = [l.replace('\n', '') for l in lines]
# 获取模型
if args.use_model == 'ecapa_tdnn':
    model = EcapaTdnn(num_class=args.num_class, input_size=train_dataset.input_size)
else:
    raise Exception(f'{args.use_model} 模型不存在!')
model.set_state_dict(paddle.load(args.model_path))
model.eval()


def infer(audio_path):
    data = load_audio(audio_path, mode='infer', feature_method=args.feature_method)
    data = data[np.newaxis, :]
    data = paddle.to_tensor(data, dtype='float32')
    data_length = paddle.to_tensor([1], dtype='float32')
    # 执行预测
    output = model(data, data_length)
    result = paddle.nn.functional.softmax(output).numpy()
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    label = class_labels[lab]
    return label


def main():
    f_result = open(f'{os.path.dirname(os.path.dirname(args.model_path))}/submission.csv', 'w', encoding='utf-8')
    audios = os.listdir(args.audio_dir)
    for audio in tqdm(audios):
        audio_path = os.path.join(args.audio_dir, audio)
        label = infer(audio_path)
        f_result.write(f'{audio}, {label}\n')
    f.close()


if __name__ == '__main__':
    main()
