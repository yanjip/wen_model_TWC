# time: 2023/11/16 20:14
# author: YanJP
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
# import torch
import pandas as pd
import para
import datetime
import pickle
from matplotlib.font_manager import FontProperties  # 导入字体模块

# 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
def chinese_font():
    try:
        font = FontProperties(
            # 系统字体路径
            fname='C:\\Windows\\Fonts\\方正粗黑宋简体.ttf', size=14)
    except:
        font = None
    return font

# 用于平滑曲线，类似于Tensorboard中的smooth
def smooth(data, weight=0.9):
    '''
    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_rewards(rewards,time,  path=None,):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    # plt.title("PPO Algorithm")
    plt.rc('font', size=15)
    plt.xlabel('Epsiodes', fontsize=17, fontweight='bold', labelpad=-1)
    plt.ylabel('Reward', fontsize=17, fontweight='bold', labelpad=-1)
    plt.grid(linestyle="--", color="gray", linewidth="0.5", axis="both")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    s_r1 = smooth(rewards)
    plt.plot(rewards, alpha=0.5, color='c')
    plt.plot(s_r1, linewidth='1.5', )
    # plt.plot(s_r2,linewidth='1.5', label='clipped probability ratio=0.5')
    # plt.ylim(-1)
    # plt.legend()
    a = time
    plt.savefig(f"{path}/{a}_power.png")
    plt.show()

def process_res(res):
    proposed=res[:,0]
    b1=res[:,1]
    b2=res[:,2]
    b3=res[:,3]
    b4=res[:,4]
    return proposed,b1,b2,b3,b4
    pass
def plot_user(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    proposed, b1, b2, b3, b4=process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 2')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 4')  # 使用三角形节点
    plt.rc('font', size=13)

    plt.legend(loc='lower center', ncol=3)
    plt.ylim(2)

    plt.xlabel('Number of Users',fontsize=13)
    plt.ylabel('Video Quality',fontsize=13)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Users-" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.show()
def plot_Nc(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    proposed, b1, b2, b3, b4=process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed')  # 使用三角形节点
    # plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 1',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 2',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 3',markerfacecolor='none')  # 使用三角形节点
    plt.rc('font', size=13)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.legend(loc='lower right', ncol=2)
    plt.ylim(4)

    plt.xlabel('Number of Sub-Carriers',fontsize=15)
    plt.ylabel('Video Quality',fontsize=15)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # plt.savefig("runs/baseline/Nc" + a, dpi=600,bbox_inches='tight', pad_inches=0.01)
    # 显示图形
    plt.show()

def plot_VQ(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    proposed, b1, b2, b3, b4=process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 2')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 4')  # 使用三角形节点
    plt.rc('font', size=13)
    plt.legend(loc='lower center', ncol=3)
    # plt.ylim(2)
    plt.xlabel('Video Quality Threshold',fontsize=13)
    plt.ylabel('Video Quality',fontsize=13)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/VQthreshold" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.show()

def plot_power(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    proposed, b1, b2, b3, b4=process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed Scheme')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 2')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 4')  # 使用三角形节点
    plt.rc('font', size=13)
    plt.legend(loc='lower center', ncol=2)
    plt.ylim(70)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Max Power (dBm)',fontsize=13)
    plt.ylabel('Video Quality',fontsize=13)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Power" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.show()
def plot_loss(x,res):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    proposed, b1, b2, b3, b4 = process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 2')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 4')  # 使用三角形节点
    plt.rc('font', size=13)
    plt.legend(loc='lower center', ncol=3)
    # plt.ylim(2)
    plt.xlabel('Channel Power Loss', fontsize=13)
    plt.ylabel('Video Quality', fontsize=13)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/loss" + a, dpi=600, bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.show()
def plot_BW(x,res):
    x=x/1000
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    proposed, b1, b2, b3, b4 = process_res(res)
    plt.plot(x, proposed, marker='>', markersize=8, label='Proposed Scheme')  # 使用三角形节点
    plt.plot(x, b1, marker='o', markersize=8, label='Baseline 1',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b2, marker='s', markersize=8, label='Baseline 2',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b3, marker='d', markersize=8, label='Baseline 3',markerfacecolor='none')  # 使用三角形节点
    plt.plot(x, b4, marker='*', markersize=8, label='Baseline 4',markerfacecolor='none')  # 使用三角形节点
    plt.rc('font', size=13)
    plt.legend(loc='lower center', ncol=2)
    plt.ylim(70)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.xlabel('Bandwidth (KHz)', fontsize=13)
    plt.ylabel('Video Quality', fontsize=13)
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Bandwidth" + a, dpi=600, bbox_inches='tight', pad_inches=0.01)
    # 显示图形
    plt.show()
def rainbow():
    np.random.seed(1)
    data_num = 200
    train_num = 3
    x = np.linspace(0, 2 * np.pi, data_num)
    # noise1 = []
    # for i in range(train_num):
    #     temp = []
    #     for j in range(data_num):
    #         temp.append(np.random.uniform(-np.exp(-j * 0.02) * 0.1, np.exp(-j * 0.02) * 0.1))
    #     noise1.append(temp)

    # y = np.tanh(x)
    # z = 1 / (1 + np.exp(-x))

    data1 = np.load("runs/rewards/2024_03_02-10_07_37_reward.npy")
    data2 = np.load("runs/rewards/2024_03_02-10_19_27_reward.npy")

    # data1 = np.load("runs/rewards/2023_12_13-21_03_15_reward.npy").tolist()
    # # data2 = np.load("runs/rewards/2023_12_13-10_12_56_reward.npy").tolist()
    # data11=np.append(data1,data1[-1])
    # data2=np.delete(data11,0)
    sr1=smooth(data1)
    sr2=smooth(data2)
    data=[sr1,sr2]

    y_mean = np.mean((np.array(data)), axis=0)
    y_std = np.std((np.array(data)), axis=0)
    y_max = y_mean + y_std * 0.99
    y_min = y_mean - y_std * 0.99


    x = np.arange(0, len(data1), 1)

    fig = plt.figure(1)
    plt.plot(x, y_mean, label='method1', color='#e75840')
    plt.fill_between(x, y_max, y_min, alpha=0.5, facecolor='#e75840')
    plt.legend()
    plt.grid(True)
    plt.show()
import warnings
warnings.filterwarnings('ignore')
def sns_pic():

    linestyle = ['-', '--', ':', '-.']
    color = ['r', 'g', 'b', 'k']
    label = ['algo1', 'algo2', 'algo3', 'algo4']

    def smooth2(data, sm=1):
        smooth_data = []
        if sm > 1:
            for d in data:
                z = np.ones(len(d))
                y = np.ones(sm) * 1.0
                d = np.convolve(y, d, "same") / np.convolve(y, z, "same")
                smooth_data.append(d)
        return smooth_data

    data = []
    # 下载好数据后，根据自己文件路径，修改np.load()中的代码路径
    data1 = np.load("runs/rewards/2024_03_02-10_07_37_reward.npy")
    data2 = np.load("runs/rewards/2024_03_02-10_19_27_reward.npy")
    # data1 = np.load("runs/rewards/2023_12_13-21_03_15_reward.npy").tolist()
    # # data2 = np.load("runs/rewards/2023_12_13-10_12_56_reward.npy").tolist()
    # data11=np.append(data1,data1[-1])
    # data2=np.delete(data11,0)

    data.append(data1)
    data.append(data2)
    # data.append(data3)
    # data.append(data4)
    # data.append(data5)
    # data.append(data6)
    # data.append(data7)
    # data.append(data8)
    # data.append(data9)
    # data.append(data10)
    y_data = smooth2(data, 5)
    x_data = np.arange(0, len(y_data[0]), 1)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot( time=x_data,data=y_data, color=color[0], linestyle=linestyle[0])
    plt.show()
def test_pic():
    sns.set()

    # rewards1 = np.array([0, 0.1, 0, 0.2, 0.4, 0.5, 0.6, 0.9, 0.9, 0.9],dtype=np.float64)
    # rewards2 = np.array([0, 0, 0.1, 0.4, 0.5, 0.5, 0.55, 0.8, 0.9, 1])
    data1 = np.load("runs/rewards/2024_03_02-10_07_37_reward.npy")
    data2 = np.load("runs/rewards/2024_03_02-10_19_27_reward.npy")
    sr1=smooth(data1)
    sr2=smooth(data2)
    # rewards = np.vstack((rewards1, rewards2))  # 合并为二维数组
    rewards = np.vstack((sr1, sr2))  # 合并为二维数组

    df = pd.DataFrame(rewards).melt(var_name='episode', value_name='reward')

    sns.lineplot(x="episode", y="reward", data=df)
    plt.show()
def plot_rewards_from_file(rewards,times, path=None,):
    # sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    # plt.title("PPO Algorithm")
    plt.rc('font', size=13)
    plt.xlabel('Epsiodes', fontsize=13)
    plt.ylabel('Reward', fontsize=13)
    plt.grid(linestyle="--",color="gray",linewidth="0.5",axis="both")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    s_r1=smooth(rewards[0])
    s_r2=smooth(rewards[1])
    # s_r1=smooth(s_r1)
    # s_r2=smooth(s_r2)
    # plt.plot(rewards, label='rewards')
    plt.plot(rewards[1],alpha=0.5,color='c')
    plt.plot(s_r2,linewidth='1.5', color='c',label='Learning rate=0.001')
    plt.plot(rewards[0],alpha=0.5,color='g')
    plt.plot(s_r1, linewidth='1.5', color='g',label='Learning rate=0.0001')

    # plt.ylim(0)

    plt.legend()
    a = times
    # plt.savefig('runs/simulation_res/'+a+"_rewards.png",dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()
def pic_reward():
    # r1=np.load('runs/rewards/2024_03_06-11_01_31_reward.npy')
    # r2=np.load('runs/rewards/2024_03_06-10_48_05_reward.npy')
    r1=np.load('runs/rewards/2024_03_05-22_51_52_reward.npy')
    r2=np.load('runs/rewards/2024_03_06-10_29_31_reward.npy')
    res=[r1,r2]
    plot_rewards_from_file(res,times='12_25')

def draw_baseline(data):
    categories = ['1', '2', '3', '4']  # 这是柱子对应的类别标签
    # 使用matplotlib绘制柱状图
    plt.bar(categories, data)

    # 添加标题和坐标轴标签
    plt.title('四组数据的柱状图')
    plt.xlabel('类别')
    plt.ylabel('数值')

    # 显示图形
    plt.show()
def draw_baseline_propose(data):
    categories = ['Proposed','1', '2', '3', '4']  # 这是柱子对应的类别标签
    # 使用matplotlib绘制柱状图
    plt.bar(categories, np.array(data)/para.K)

    # 添加标题和坐标轴标签
    # plt.title('四组数据的柱状图')
    # plt.xlabel('类别')
    plt.ylabel('Rewards')
    # 显示图形
    plt.show()

if __name__ == '__main__':
    # rainbow()
    # sns_pic()  #superior
    # test_pic()  #慢

    # data1 = np.load("runs/rewards/2023_12_14-20_41_14_reward.npy").tolist()
    # plot_rewards(data1,time='11',path='runs/pic')

    # Nc=np.array([150,160,170,180,190,200])-60
    # res=np.load('runs/simulation_res/Nc_12_21.npy')
    # plot_Nc(Nc,res)

    # Bws=np.array([16,18,20,22,24])
    # res=np.load('runs/simulation_res/bandwidth12_21.npy')
    # plot_BW(Bws,res)

    pic_reward()