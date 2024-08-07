import math

import paddle


def cosine_decay_with_warmup(learning_rate, step_per_epoch, fix_epoch=1000, warmup_epoch=5, min_lr=0.0):
    """
    :param learning_rate: 学习率
    :param step_per_epoch: 每个epoch的步数
    :param fix_epoch: 最大epoch数
    :param warmup_epoch: 预热步数
    :param min_lr: 最小学习率
    :return:
    """
    # 预热步数
    boundary = []
    value = []
    warmup_steps = warmup_epoch * step_per_epoch
    # 初始化预热步数
    for i in range(warmup_steps + 1):
        if warmup_steps > 0:
            alpha = i / warmup_steps
            lr = learning_rate * alpha
            value.append(lr)
        if i > 0:
            boundary.append(i)

    max_iters = fix_epoch * int(step_per_epoch)
    warmup_iters = len(boundary)
    # 初始化最大步数
    for i in range(int(boundary[-1]), max_iters):
        boundary.append(i)
        # 如果当前步数小于最大步数，则将当前步数设置为最小学习率
        if i < max_iters:
            decayed_lr = min_lr + (learning_rate - min_lr) * 0.5 * (math.cos(
                (i - warmup_iters) * math.pi / (max_iters - warmup_iters)) + 1)
            value.append(decayed_lr)
        else:
            value.append(min_lr)
    return paddle.optimizer.lr.PiecewiseDecay(boundary, value)
