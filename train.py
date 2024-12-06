# -*- coding:utf-8 -*-
# 此模块为所有的指标都输出

import torch

import argparse
import numpy as np
import os
import time
from utils.dataset_v6 import MyDataset
from torch.utils import data
from tqdm import tqdm
import datetime  # 获取指定日期和时间

from torch.utils.tensorboard import SummaryWriter

from utils.utils import make_one_hot, DiceLoss, SegmentationMetric
from utils.model_builder import build_model

def train_model(args):
    model = build_model(args.model, args.classes).to(args.device)

    train_loader, test_loader = data_loader(args)

    results_file = "{}_{}results.txt".format(args.model,args.dataset)

    weights_save_path = "{}/{}_{}".format(args.save_dir, args.model, args.dataset)
    if not os.path.exists(weights_save_path): os.makedirs(weights_save_path)

    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            model.load_state_dict(torch.load(args.resume))
            print("Loaded checkpoint '{}'! ".format(args.resume))
            start_epoch = int(os.path.basename(args.resume).split(".")[0].split("_")[1])
        else:
            print("No checkpoint found at '{}'!".format(args.resume))

    metric = SegmentationMetric(args.classes)
    writer = SummaryWriter(args.directory)
    criterion1 = DiceLoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_scheduler_gamma)

    for epoch in range(start_epoch, args.epochs):
        # Train ====================================================================================
        train_start_time = time.time()
        average_epoch_loss_train  = train(args, model, train_loader, optimizer, criterion1)
        train_end_time = time.time()

        print("Epoch:{:3d}\ttrain_loss:{:4f}\tTime:{:2f}min".format(epoch + 1, average_epoch_loss_train , (train_end_time - train_start_time) / 60))
        writer.add_scalar("Train Loss", average_epoch_loss_train , epoch + 1)
        scheduler.step()
        if epoch != 0 and epoch % 100 == 0: # model.state_dict()只保存训练好的权重，dir绝对路径加文件名
            torch.save(model.state_dict(), '{}/{}_{}.pth'.format(weights_save_path, args.model, epoch))

        # Validation ===============================================================================
        if epoch % 1 == 0 or epoch == (args.epochs - 1):

            val_start_time = time.time()
            eval_loss, pa_, mPA_, IOU_,mIOU_, Pr_, Recall_, F1_ = val(args, model, test_loader, criterion1, metric)
            val_end_time = time.time()

            print("Epoch:{:3d}"
                  "\tEval_loss:{:4f}" # \t代表一个tab
                  "\tpa:{:.4f}"
                  "\tmPA:{:.4f}"
                  "\tIOU:{}"
                  "\nmIOU:{:.4f}"
                  "\tPrecision:{:.4f}"
                  "\tRecall:{:.4f}"
                  "\tF1 Score:{:.4f}"
                  "\tTime:{:2f}min".format(epoch + 1, eval_loss, pa_, mPA_,IOU_, mIOU_, Pr_, Recall_, F1_, (val_end_time - val_start_time) / 60))

            writer.add_scalar("Eval Loss", eval_loss, epoch + 1)  # 名称，值，横坐标
            writer.add_scalar("mPA", mPA_, epoch + 1)
            writer.add_scalar("mIOU", mIOU_, epoch + 1)
            writer.add_scalar("Precision", Pr_, epoch + 1)
            writer.add_scalar("Recall", Recall_, epoch + 1)
            writer.add_scalar("F1 Score", F1_, epoch + 1)

            # writer.close()
            with open(os.path.join(weights_save_path,results_file), "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标，mean_loss训练过程平均损失
                # python数字格式化方法，:.4f保留小数点后四位
                train_info = f"[epoch: {epoch+1}]\n" \
                             f"loss: {average_epoch_loss_train}\n" \
                             f"time: {(train_end_time - train_start_time) / 60}\n" \

                val_info=("epoch:{:3d}"
                  "\tEval_loss:{:4f}" # \t代表一个tab
                  "\tpa:{:.4f}"
                  "\tmPA:{:.4f}"
                  "\tIOU:{}"
                  "\nmIOU:{:.4f}"
                  "\tPrecision:{:.4f}"
                  "\tRecall:{:.4f}"
                  "\tF1 Score:{:.4f}"
                  "\tTime:{:2f}min".format(epoch + 1, eval_loss, pa_, mPA_,IOU_, mIOU_, Pr_, Recall_, F1_, (val_end_time - val_start_time) / 60))
                f.write(train_info+val_info+ "\n\n")
    torch.save(model.state_dict(), '{}/{}_{}.pth'.format(weights_save_path, args.model, args.epoch))

def train(args, model, train_loader, optimizer, criterion1):
    model.train()
    epoch_loss = []
    for x, y in tqdm(train_loader):  # tqdm库的主要作用是可视化当前网络训练的进程
        optimizer.zero_grad()
        inputs = x.to(args.device)
        labels = y.to(args.device)

        labels_for_make_one_hot = torch.unsqueeze(labels, dim=1)
        labels_one_hot = make_one_hot(labels_for_make_one_hot, num_classes=args.classes)
        labels_one_hot = labels_one_hot.to(args.device)

        outputs = model(inputs)

        loss1 = criterion1(outputs, labels_one_hot)

        loss = loss1

        loss.backward()

        optimizer.step()
        epoch_loss.append(loss.item())

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train


def val(args, model, test_loader, criterion1, metric):
    model.eval()
    eval_loss = 0.0
    PA, cPA, mPA, IOU, mIOU, Pr, Recall, F1 = 0, 0, 0, 0, 0, 0, 0, 0
    len_test_loader = len(test_loader)
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            inputs = x.to(args.device)
            labels = y.to(args.device)

            labels_for_make_one_hot = torch.unsqueeze(labels, dim=1)
            labels_one_hot = make_one_hot(labels_for_make_one_hot, num_classes=args.classes)
            labels_one_hot = labels_one_hot.to(args.device)

            outputs = model(inputs)

            loss1 = criterion1(outputs, labels_one_hot)

            loss=loss1
            eval_loss += loss.item()

            pred = outputs.data.cpu().numpy()
            target = labels.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            hist = metric.addBatch(pred, target)
            pa = metric.pixelAccuracy()
            cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            IoU = metric.IntersectionOverUnion()
            mIoU = metric.meanIntersectionOverUnion()
            pr = metric.precision()
            rcl = metric.recall()
            f1 = (2 * pr * rcl) / (pr + rcl)

            pr_list = []
            rcl_list = []
            f1_list = []
            for i in range(len(pr)):
                if not np.isnan(pr[i]):
                    pr_list.append(pr[i])
                if not np.isnan(rcl[i]):
                    rcl_list.append(rcl[i])
                if not np.isnan(f1[i]):
                    f1_list.append(f1[i])
            pa += pa
            mPA += mpa
            IOU +=IoU
            mIOU += mIoU
            Pr += np.mean(pr_list)
            Recall += np.mean(rcl_list)
            F1 += np.mean(f1_list)

    pa_= pa/ len_test_loader
    mPA_ = mPA / len_test_loader
    IOU_ = IOU/ len_test_loader
    mIOU_ = mIOU / len_test_loader
    Pr_ = Pr / len_test_loader
    Recall_ = Recall / len_test_loader
    F1_ = F1 / len_test_loader

    return eval_loss, pa_, mPA_, IOU_,mIOU_, Pr_, Recall_, F1_


from torch.utils.data import random_split
def data_loader(args):
    train_dataset = MyDataset("data/{}".format(args.dataset), resize_h=args.size[0], resize_w=args.size[1])
    n_val = int(len(train_dataset) * 0.1)
    n_train = len(train_dataset) - n_val
    train, val = random_split(train_dataset, [n_train, n_val])
    train_loader = data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_loader = data.DataLoader(dataset=val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    return train_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation")
    parser.add_argument("--model", type=str, default='unet')
    parser.add_argument("--dataset", type=str, default="dataset_0")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=210)
    parser.add_argument("--size", type=tuple, default=(512, 512), help="(H, W)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=0.012)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--milestones", type=list, default=[100,200,500])
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--resume", type=str, default=r"", help="Load checkpoint for continuing training.")
    parser.add_argument('--experiment-start-time', type=str,
                    default=datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'))
    args = parser.parse_args()

    directory = f"runs/{args.model}_{args.dataset}_{args.experiment_start_time}"

    args.directory = directory
    return args


if __name__ == '__main__':
    args = parse_args()
    print("args: ", args, "\n")
    train_model(args)
