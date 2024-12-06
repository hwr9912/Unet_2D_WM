import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_min_enclosing_circles(image_path):
    """
    计算图像上两个用户选择的最小外接圆，并返回它们的圆心坐标和半径。

    参数:
    - image_path: 图像的文件路径

    返回:
    - results: 一个列表，其中包含两个最小外接圆的圆心坐标和半径。
               格式为 [(center1, radius1), (center2, radius2)]
    """

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", image)

    def get_min_enclosing_circle():
        if len(points) < 2:
            print("请至少点击两个点来确定一个最小外接圆。")
            return None, None
        points_array = np.array(points, dtype=np.float32)
        center, radius = cv2.minEnclosingCircle(points_array)
        return (int(center[0]), int(center[1])), int(radius)

    # 加载图片
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError("无法加载图片。请检查路径。")

    results = []
    for i in range(2):
        print(f"开始第 {i + 1} 次点击，请在图片上点击多个点来确定最小外接圆。第一个外接圆对应水池，第二个对应平台")
        points = []  # 清空之前的点
        image = original_image.copy()  # 每次操作开始前重置图像
        cv2.imshow("Image", image)
        cv2.setMouseCallback("Image", mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # 回车键确认
                center, radius = get_min_enclosing_circle()
                if center and radius:
                    print(f"圆心坐标: {center}, 半径: {radius}")
                    results.append(RoundObject(center, radius))
                    break
            elif key == 27:  # 按下 ESC 键退出
                print("退出")
                cv2.destroyAllWindows()
                return results

    cv2.destroyAllWindows()
    return results[0], results[1]


class RoundObject:
    def __init__(self, centroid: tuple[float, float] = (0, 0), radius: float = 0):
        """
        :param centroid: tuple[float, float]，圆心位置坐标，默认为(0, 0)
        :param radius: float，半径长度，默认为0
        """
        self.centroid = centroid
        self.radius = radius

    def copy(self):
        new = RoundObject(self.centroid, self.radius)
        return new

    def save(self, dir):
        """
        在numpy.save基础上的包装，保存文件为npy，依次存储圆心坐标和半径
        :param dir: 文件位置
        :return:
        """
        pos = np.array([self.centroid[0],
                        self.centroid[1],
                        self.radius])
        np.save(file=dir, arr=pos)
        return 0

    def load(self, dir):
        """
        在numpy.load基础上的包装，读取文件为npy，依次读取圆心坐标和半径
        :param dir:
        :return:
        """
        pos = np.load(dir)
        self.centroid = (pos[0], pos[1])
        self.radius = pos[2]
        return self

    def adjust(self, relative_centroid: tuple[float, float] = (0, 0)):
        out = self.copy()
        out.centroid = (self.centroid[0] - relative_centroid[0],
                        self.centroid[1] - relative_centroid[1])
        return out

    def cv2plt(self):
        out = self.copy()
        out.centroid = (-self.centroid[1], self.centroid[0])
        return out

class WaterMaze:
    def __init__(self, path_array: np.ndarray, maze: RoundObject, platform: RoundObject):
        """
        水迷宫实验类
        :param path_array: (N, 2)的numpy.ndarray类，提供路径参数
        :param maze: animal.WaterMaze.RoundObject类，提供迷宫参数
        :param platform: animal.WaterMaze.RoundObject类，提供平台参数
        """
        if path_array.shape[1] != 2:
            raise ValueError("Shape of input array should be (N, 2)!")
        self.path = path_array
        self.len = path_array.shape[0]
        self.maze = maze
        self.platform = platform

        # print(f"Total {self.len} frames, "
        #       f"centroid of maze area is ({self.maze.centroid[0]}, {self.maze.centroid[1]}), "
        #       f"radius is {self.maze.radius:.2f}; "
        #       f"centroid of platform is ({self.platform.centroid[0]}, {self.platform.centroid[1]}), "
        #       f"radius is {self.platform.radius:.2f}.")

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

    def visualizing_roadmap(self,
                            fig_save_path: str = "data/wm_roadmap.pdf",
                            figsize: tuple[float, float] = (8, 8),
                            save: bool = False):
        # opencv以图片右下角为原点确立坐标，且0维是y轴，1维是x轴，且x轴是反向的
        x = self.path[:, 0] - self.maze.centroid[0]
        y = -(self.path[:, 1] - self.maze.centroid[1])

        # 重新绘制图像，去掉坐标轴
        plt.figure(figsize=figsize)

        plt.plot(x, y, color='black', linewidth=2)

        # 标记起点
        plt.scatter(x[0], y[0], marker="^", color='darkred', s=100, zorder=5)
        # 标记终点
        plt.scatter(x[-1], y[-1], marker="x", color='darkblue', s=100, zorder=5)

        # 绘制水迷宫范围
        adjust_maze = self.maze.adjust(self.maze.centroid).cv2plt()
        perimeter = plt.Circle(adjust_maze.centroid, adjust_maze.radius, color='black', fill=False, linewidth=3)
        plt.gca().add_patch(perimeter)

        # 绘制平台
        adjust_platform = self.platform.adjust(self.maze.centroid).cv2plt()
        platform = plt.Circle(adjust_platform.centroid, adjust_platform.radius, color='red', fill=False, linewidth=2)
        plt.gca().add_patch(platform)

        # 绘制平行于 x 轴的直线
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

        # 绘制平行于 y 轴的直线
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # 去掉坐标轴
        plt.axis('off')

        # 保存为PDF文件
        if save:
            plt.savefig(fig_save_path, format='pdf')
        else:
            plt.show()

        # 关闭图像以节省内存
        plt.close()


# if __name__ == "__main__":
#     os.chdir(r"D:\Python\computer_vision\unet_water_maze")
#     maze_dir = r"data/maze.npy"
#     platform_dir = r"data/platform.npy"
#     # 确定迷宫及平台位置
#     if os.path.exists(maze_dir) and os.path.exists(platform_dir):
#         maze = RoundObject().load(maze_dir)
#         platform = RoundObject().load(platform_dir)
#     else:
#         maze, platform = calculate_min_enclosing_circles(r"data\20240822WM2\frame_00000.jpg")
#         maze.save(maze_dir)
#         platform.save(platform_dir)
#
#     # 加载路径
#     path = np.load(r"data\loc\crop_20240822WM3_loc.npy")
#     # 构建水迷宫对象
#     experiment20240822 = WaterMaze(path_array=path[:, 0:2],
#                                    maze=maze,
#                                    platform=platform)
#     # 绘制路径图
#     experiment20240822.visualizing_roadmap(save=True)
