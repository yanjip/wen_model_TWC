# time: 2024/3/9 10:05
# author: YanJP
# import numpy as np
# average_power_loss = 1e-7
# sigma = np.sqrt(average_power_loss / 2)
# 假定H矩阵，4x4复数矩阵表示基站到4个用户的信道
# 生成随机的复数形式的信道信息（实部和虚部分别独立生成）
import numpy as np
import matplotlib.pyplot as plt

def zipf_pic():
    # 定义视频数量
    num_videos = 16

    # 生成Zipf分布的概率
    a = 2.8  # Zipf分布的参数，可以调整
    zipf_probs = np.random.zipf(a, num_videos)

    # 归一化概率，使其总和为1
    zipf_probs_normalized = zipf_probs / np.sum(zipf_probs)

    # 打印生成的概率
    print("生成的Zipf分布概率：", zipf_probs_normalized)

    # 绘制直方图
    plt.bar(range(1, num_videos + 1), zipf_probs_normalized)
    plt.xlabel('视频')
    plt.ylabel('概率')
    plt.title('Zipf分布概率')
    plt.show()

if __name__ == '__main__':
    # zipf_pic()

    x=[[] for _ in range(3)]
    x[1].append(9)
    print(x)


