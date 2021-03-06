import argparse
import functools
import os
import time
from datetime import datetime, timedelta

import paddle
import yaml
from paddle.io import DataLoader
from paddle.metric import accuracy
from visualdl import LogWriter

from data_utils.noise_perturb import NoisePerturbAugmentor
from data_utils.reader import CustomDataset
from data_utils.speed_perturb import SpeedPerturbAugmentor
from data_utils.volume_perturb import VolumePerturbAugmentor
from modules.model import Model
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    8,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    300,                      '训练的轮数')
add_arg('num_class',        int,    6,                        '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('scaler_path',      str,    'dataset/standard.m',     '测试数据的数据列表路径')
add_arg('save_model_dir',   str,    'output/models/',         '模型保存的路径')
add_arg('augment_conf_path',str,    'configs/augment.yml',    '数据增强的配置文件，为json格式')
add_arg('resume',           str,    None,                     '恢复训练的模型文件夹，当为None则不使用恢复模型')
add_arg('pretrained_model', str,    None,                     '预训练模型的模型文件夹，当为None则不使用预训练模型')
args = parser.parse_args()


# 评估模型
@paddle.no_grad()
def evaluate(model, test_loader):
    model.eval()
    accuracies = []
    for batch_id, (audio, label) in enumerate(test_loader()):
        output = model(audio)
        label1 = paddle.reshape(label, shape=(-1, 1))
        acc = accuracy(input=output, label=label1)
        # 真实标签
        accuracies.append(acc.numpy()[0])
    model.train()
    acc = float(sum(accuracies) / len(accuracies))
    return acc


def train():
    # 日志记录器
    writer = LogWriter(logdir='output/log')
    # 获取数据增强器
    augmentors = None
    if args.augment_conf_path is not None:
        augmentors = {}
        with open(args.augment_conf_path, encoding="utf-8") as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
        augmentors['noise'] = NoisePerturbAugmentor(**configs['noise'])
        augmentors['speed'] = SpeedPerturbAugmentor(**configs['speed'])
        augmentors['volume'] = VolumePerturbAugmentor(**configs['volume'])
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path,
                                  scaler_path=args.scaler_path,
                                  mode='train',
                                  sr=16000,
                                  chunk_duration=3,
                                  augmentors=augmentors)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    # 测试数据
    eval_dataset = CustomDataset(args.test_list_path,
                                 scaler_path=args.scaler_path,
                                 mode='eval',
                                 sr=16000,
                                 chunk_duration=3)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    model = Model(num_class=args.num_class)
    paddle.summary(model, input_size=(1, 312))

    # 初始化epoch数
    last_epoch = 0
    # 学习率衰减
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.learning_rate, T_max=args.num_epoch)
    # 设置优化方法
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                          learning_rate=scheduler,
                                          momentum=0.9,
                                          weight_decay=paddle.regularizer.L2Decay(3e-5))

    # 加载预训练模型
    if args.pretrained_model is not None:
        model_dict = model.state_dict()
        param_state_dict = paddle.load(os.path.join(args.pretrained_model, 'model.pdparams'))
        for name, weight in model_dict.items():
            if name in param_state_dict.keys():
                if weight.shape != list(param_state_dict[name].shape):
                    print('{} not used, shape {} unmatched with {} in model.'.
                          format(name, list(param_state_dict[name].shape), weight.shape))
                    param_state_dict.pop(name, None)
            else:
                print('Lack weight: {}'.format(name))
        model.set_dict(param_state_dict)
        print('成功加载预训练模型参数')

    # 恢复训练
    if args.resume is not None:
        model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
        optimizer_state = paddle.load(os.path.join(args.resume, 'optimizer.pdopt'))
        optimizer.set_state_dict(optimizer_state)
        # 获取预训练的epoch数
        last_epoch = optimizer_state['LR_Scheduler']['last_epoch']
        print(f'成功加载第 {last_epoch} 轮的模型参数和优化方法参数')

    # 获取损失函数
    loss = paddle.nn.CrossEntropyLoss()
    train_step, test_step = 0, 0
    sum_batch = len(train_loader) * (args.num_epoch - last_epoch)
    # 开始训练
    for epoch in range(last_epoch, args.num_epoch):
        loss_sum = []
        accuracies = []
        start = time.time()
        for batch_id, (audio, label) in enumerate(train_loader()):
            output = model(audio)
            # 计算损失值
            los = loss(output, label)
            los.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 计算准确率
            label = paddle.reshape(label, shape=(-1, 1))
            acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
            accuracies.append(acc.numpy()[0])
            loss_sum.append(los.numpy()[0])
            # 多卡训练只使用一个进程打印
            if batch_id % 10 == 0:
                eta_sec = ((time.time() - start) * 1000) * (sum_batch - (epoch - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print(f'[{datetime.now()}] '
                      f'Train epoch [{epoch}/{args.num_epoch}], '
                      f'batch: [{batch_id}/{len(train_loader)}], '
                      f'loss: {(sum(loss_sum) / len(loss_sum)):.5f}, '
                      f'accuracy: {(sum(accuracies) / len(accuracies)):.5f}, '
                      f'lr: {scheduler.get_lr():.8f}, '
                      f'eta: {eta_str}')
                writer.add_scalar('Train/Loss', los.numpy()[0], train_step)
                writer.add_scalar('Train/Accuracy', (sum(accuracies) / len(accuracies)), train_step)
                train_step += 1
            start = time.time()
        # 执行评估和保存模型
        s = time.time()
        acc = evaluate(model, eval_loader)
        eta_str = str(timedelta(seconds=int(time.time() - s)))
        print('='*70)
        print(f'[{datetime.now()}] Test {epoch}, Accuracy: {acc:.5f}, time: {eta_str}')
        print('='*70)
        writer.add_scalar('Test/Accuracy', acc, test_step)
        # 记录学习率
        writer.add_scalar('Train/Learning rate', scheduler.last_lr, epoch)
        test_step += 1
        # 保存模型
        os.makedirs(args.save_model_dir, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(args.save_model_dir, 'model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(args.save_model_dir, 'optimizer.pdopt'))
        scheduler.step()


if __name__ == '__main__':
    print_arguments(args)
    train()
