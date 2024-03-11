# time: 2024/3/9 16:02
# author: YanJP
import numpy as np


np.random.seed(1)  # 随机种子
R = 2  # 基站到基站的距离 0.8
PL = lambda d: 128.1 + 37.6 * np.log10(d)  # 路损模型，d--km
U = 7  # 用户个数，每个蜂窝中有一个用户
C = 7  # 基站个数
P = 50  # 最大发射功率43 dBm
sigma2 = -105  # 噪声功率 -100 dBm
shadowing_std = 8  # 阴影衰落的标准差-8 dB

B = 10e6  # 10Mhz
def channel_generate(U, R, PL, shadowing_std):
    # 在正六边形蜂窝小区中撒点
    cell_loc = np.array([[0, 0],
                         [R * np.cos(np.pi / 6), R * np.sin(np.pi / 6)],
                         [0, R],
                         [-R * np.cos(np.pi / 6), R * np.sin(np.pi / 6)],
                         [-R * np.cos(np.pi / 6), -R * np.sin(np.pi / 6)],
                         [0, -R],
                         [R * np.cos(np.pi / 6), -R * np.sin(np.pi / 6)]])

    C = 7  # 蜂窝小区个数

    L = R * np.tan(np.pi / 6)  # 六边形的边长

    # 产生用户位置
    user_loc = np.zeros((U, 2))
    i = 0
    while i < U:
        x = 2 * L * np.random.rand(2) - 1 * L
        if (abs(x[0]) + abs(x[1]) / np.sqrt(3)) <= L and abs(x[1]) <= L * np.sqrt(3) / 2:
            i += 1
            user_loc[i - 1, :] = x + cell_loc[i - 1, :]

    # 计算距离
    dis = np.zeros((U, C))
    for i in range(U):
        for j in range(C):
            dis[i, j] = np.linalg.norm(cell_loc[j, :] - user_loc[i, :])

    # 计算信道增益，考虑服从对数正态分布的阴影衰落
    H_gain = -PL(dis) - shadowing_std * np.random.randn(U, C)

    return H_gain


if __name__ == '__main__':
    H_gain = channel_generate(U, R, PL, shadowing_std)  # 随机产生用户位置，并产生信道增益
    H_gain = 10 ** ((H_gain - sigma2) / 10)  # 为了优化方便，将噪声功率归一化
    print(H_gain)