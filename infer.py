import argparse
import functools

from ppser.predict import PPSERPredictor
from ppser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/bi_lstm.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                    '是否使用GPU预测')
add_arg('audio_path',       str,    'dataset/audios/angry/audio_0.wav', '音频路径')
add_arg('model_path',       str,    'models/BiLSTM_CustomFeature/best_model/',     '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = PPSERPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

label, score = predictor.predict(audio_data=args.audio_path)

print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
