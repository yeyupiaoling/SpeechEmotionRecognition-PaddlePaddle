import argparse
import functools

import paddle
from paddle.static import InputSpec

from modules.model import Model
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_class',        int,    6,                                   '分类的类别数量')
add_arg('model_path',       str,    'output/models/model.pdparams',      '模型保存的路径')
add_arg('save_path',        str,    'output/inference/inference',        '模型保存的路径')
args = parser.parse_args()

# 获取模型
model = Model(num_class=args.num_class)
model.set_state_dict(paddle.load(args.model_path))
# 加上Softmax函数
model = paddle.nn.Sequential(model, paddle.nn.Softmax())

# 保存预测模型
paddle.jit.save(layer=model, path=args.save_path, input_spec=[InputSpec(shape=(-1, 312))])
