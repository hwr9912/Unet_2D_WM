import numpy as np
import pandas as pd


class OpenField():
    def __init__(self, path_array: np.array, width, height):
        if path_array.shape[1] != 2:
            raise ValueError("Shape of input should be (N, 2)!")
        self.path = path_array
        self.len = path_array.shape[0]
        self.width = width
        self.height = height
        print(f"Total {self.len} frames, size of area is {width:.2f} * {height:.2f}")

    def average_speed(self, fps):
        # 计算相邻点之间的距离
        distances = np.sqrt(np.sum(np.diff(self.path, axis=0) ** 2, axis=1))

        # 计算总距离
        total_distance = np.sum(distances)

        # 计算总时间 (每行代表 1/30 秒)
        total_time = len(self.path) / fps

        # 计算平均速度
        average_speed = total_distance / total_time

        return average_speed

    def center_dot_count(self, lower_ratio=0.25, upper_ratio=0.75):
        # 定义矩形区域的边界
        lower_x = lower_ratio * self.width
        upper_x = upper_ratio * self.width
        lower_y = lower_ratio * self.height
        upper_y = upper_ratio * self.height

        # 检查每个点是否在矩形区域内
        in_rectangle = ((self.path[:, 0] >= lower_x) & (self.path[:, 0] <= upper_x) &
                        (self.path[:, 1] >= lower_y) & (self.path[:, 1] <= upper_y))

        # 计数满足条件的点
        count = np.sum(in_rectangle)

        return count

    def center_staying_time(self, fps, lower_ratio=0.25, upper_ratio=0.75):
        return self.center_dot_count(lower_ratio, upper_ratio) / fps

    def center_staying_time_ratio(self, fps, lower_ratio=0.25, upper_ratio=0.75):
        return self.center_dot_count(lower_ratio, upper_ratio) / self.len