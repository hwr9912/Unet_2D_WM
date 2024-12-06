import numpy as np


def pixel_accuracy(label, predict):
    """
    输出像素准确率
    :param label: 标记mask数据
    :param predict: 预测mask数据
    :return:像素准确率
    """
    return ((label == predict).sum()) / label.size


# Intersection-Over-Union (IoU)，也称为 Jaccard 指数，是语义分割中最常用的指标之一
def inter_over_union(label, predict):
    """
    交并比Intersection-Over-Union (IoU)，也称为 Jaccard 指数，是语义分割中最常用的指标之一，计算方式是用TP/(TP+FP+FN)
    :param label: 标记mask数据
    :param predict: 预测mask数据
    :return:交并比Intersection-Over-Union (IoU)
    """
    assert (len(label.shape) == len(predict.shape))
    # 两者相乘值为1的部分为交集
    intersection = np.multiply(label, predict)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(label + predict > 0, np.float32)
    iou = intersection.sum() / union.sum() if union.sum() > 0 else 0
    return iou
