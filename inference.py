# -*- coding:utf-8 -*-

import torch
import numpy as np
import os
import cv2
from utils.utils import convert_pixels_v2
import argparse
from utils.model_builder import build_model

def inference(args):
    model = build_model(args.model, args.classes).to(args.device)

    model.load_state_dict(torch.load(f"weights/{args.model}_{args.dataset}/{args.model}_{args.epochs}.pth"))
    model.eval()
    output_dir = "{}/{}".format(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    test_img_path = args.valset
    img_list = sorted(os.listdir(test_img_path))
    for i in img_list:
        img_path = test_img_path + "/{}".format(i)
        try:
            cv2img = cv2.imread(img_path).astype(np.float32)

            img_ame = os.path.basename(img_path).split(".")[0]
            H, W, _ = cv2img.shape
            img_x = cv2.resize(cv2img, (args.size[1], args.size[0]))
            img_x = np.transpose(img_x, axes=(2, 0, 1))
            img_x = torch.from_numpy(img_x).type(torch.FloatTensor).to(args.device)
            img_x = torch.unsqueeze(img_x, dim=0)

            pred_out = model(img_x)

            pred = pred_out.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)  # axis=1表示每一行的最大值的索引 [1,c,h,w]

            pred = pred[0, :, :]

            converted_image = convert_pixels_v2(pred, args.classes, args.size[0], args.size[1])
            cv2.imwrite("{}/{}".format(output_dir, i), converted_image)

        except Exception as Error:
            print(Error)

# 预测前修改一下epochs、验证集valset、算法model
def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--dataset", type=str, default="dataset_0")
    parser.add_argument("--valset", type=str, default="data/val")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--size", type=tuple, default=(256, 256), help="(H, W)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=0.012)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--milestones", type=list, default=[100, 200, 500])
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--resume", type=str, default="", help="Load checkpoint for continuing training.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("args: ", args)
    inference(args)

