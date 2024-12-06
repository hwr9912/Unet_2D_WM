import os
import cv2
import torch
import numpy as np


def loading_model(model,
                  model_params_file,
                  device=torch.device(0)):
    # 将模型加载到计算卡
    model.to(device)
    # 加载模型参数
    state_dict = torch.load(model_params_file)
    model.load_state_dict(state_dict)

    return model, device


def generating_mask_video(original_video_path: str,
                          mask_video_path: str,
                          model,
                          device=torch.device(0)):
    # 读取视频文件
    cap = cv2.VideoCapture(original_video_path)
    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义编解码器并创建 VideoWriter 对象:输出为灰度帧视频
    out = cv2.VideoWriter(mask_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 当获取完最后一帧就结束
            input = cv2.resize(frame.astype(np.float32), (512, 512))
            input = np.transpose(input, axes=(2, 0, 1))
            input = torch.from_numpy(input).type(torch.FloatTensor).to(device)
            input = torch.unsqueeze(input, dim=0)
            predict = model(input)
            # 把张量转化为np.array：(1, 2, 256, 256)
            predict = predict.data.cpu().numpy()
            # 由输出的元素概率矩阵转化为mask矩阵
            predict = np.argmax(predict, axis=1)  # axis=1表示每一行的最大值的索引 [1,c,h,w]
            # 压缩掉无用的矩阵的元素维度
            predict = np.float32(predict.squeeze())
            predict = cv2.resize(predict, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            # 二值化处理
            connect = np.uint8(predict > 0.5) * 255
            # 输出mask图像
            out.write(connect)
    else:
        print('视频打开失败！')

    cap.release()
    out.release()
    return 0


def recording_location(mask_video_path: str,
                       loc_table_save_path: str):
    # 读取视频
    capture = cv2.VideoCapture(mask_video_path)
    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            if not ret:
                break  # 当获取完最后一帧就结束
            # 转换为灰度图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 最大连通区域
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(frame > 127), connectivity=8)
            # 保存参
            try:
                if stats.shape[0] > 1:
                    res = np.insert(res, res.shape[0], stats[np.argmax(stats[1:, 4]) + 1, :], axis=0)
            except:
                # 第一帧数据插入
                res = np.array([stats[np.argmax(stats[1:, 4]) + 1, :]])
    else:
        print('视频打开失败！')
    # 存储路径
    res = np.float32(res)
    res[:, 0:2] = res[:, 0:2] + 0.5 * res[:, 2:4]
    np.save(file=loc_table_save_path, arr=res)
    return 0


def generating_location_heatmap(mask_video_path: str,
                                loc_table_save_path: str):
    return 0

# if __name__ == "__main__":
#     os.chdir(r"D:\Python\computer_vision\unet_water_maze")
#     # 加载模型
#     from model.unet import UNet
#     model, device = loading_model(model=UNet(classes=2),
#                                   model_params_file=r"weights\unet_dataset_0\unet_200.pth")
#     # 生成识别视频
#     generating_mask_video(original_video_path=r"data\crop_20240822WM3.avi",
#                           mask_video_path=r"data\crop_20240822WM3_out.avi",
#                           model=model, device=device)
#     # 记录位置
#     recording_location(mask_video_path=r"data\crop_20240822WM3_out.avi",
#                        loc_table_save_path=r"data\loc\crop_20240822WM3_loc.npy")