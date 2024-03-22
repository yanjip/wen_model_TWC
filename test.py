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
import matplotlib.patches as mpatches
def pic_grid():


    # 生成随机的4行25列的numpy矩阵，数值范围在0到5之间
    # data = np.random.randint(-1,5, size=(4, 25))
    data=np.load('runs/simulation_res/bitrates.npy').T

    # 定义颜色映射
    colors = [ 'forestgreen', 'limegreen','white','gold', 'orange','gray']

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(15, 5))
    # 计算每个格子的宽度
    cell_width = 0.8
    # 绘制格子图

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.add_patch(plt.Rectangle((j * cell_width, -i), cell_width, 1,  facecolor=colors[data[i, j]], edgecolor='black',linewidth=1))
            # ax.add_patch(plt.Rectangle((j, -i), 1, 1, color=colors[data[i, j]]))
            # ax.text(j + 0.5, -i + 0.5, str(data[i, j]), ha='center', va='center', color='black')
            ax.text((j + 0.5) * cell_width, -i + 0.5, str(data[i, j]), ha='center', va='center', color='black')
    # 创建自定义图例
    legend_patches = [mpatches.Patch(color=colors[i], label=str(i)) for i in range(-1,5)]
    plt.legend(handles=legend_patches, title="Array Value", loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=6)
    # 调整子图之间的间距
    # 设置图例的左边添加一个标题叫“user”
    plt.gca().add_artist(plt.text(-0.05, 0.6, 'Users', transform=ax.transAxes, fontsize=12))

    # 设置图例的上边添加一个标题叫“tile”
    plt.gca().add_artist(plt.text(0.45, 1.02, 'Tiles', transform=ax.transAxes, fontsize=12, ha='center'))

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(0.5, 21.3, 0.8))
    # ax.set_xticks(np.arange(0.5, 26.5, 1))

    # ax.set_xticklabels(range(25))
    ax.set_yticks(np.arange(-0.5, -4.5, -1))
    # ax.set_yticklabels(range(4))

    # 设置坐标轴标签
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # 设置图形标题
    # ax.set_title('Grid Plot')

    # 去除坐标轴
    ax.axis('off')
    # 调整布局，防止图例遮挡图形内容
    plt.tight_layout()

    # 显示图形
    plt.show()
def cdf():
    # 生成一组随机数据
    data = np.random.normal(loc=0, scale=1, size=500)

    # 对数据进行排序
    sorted_data = np.sort(data)

    # 计算累积概率
    cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # 绘制CDF图
    plt.plot(sorted_data, cumulative_prob, linestyle='-.',linewidth=2)
    plt.xlabel('Data')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF Plot')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # zipf_pic()

    # x=[[] for _ in range(3)]
    # x[1].append(9)
    # print(x)

    # res=np.load('runs/rewards/2024_03_13-10_30_46_reward.npy')
    # print(res[0:5])

    pic_grid()
    #
    # cdf()