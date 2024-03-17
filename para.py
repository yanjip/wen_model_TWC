# time: 2024/3/9 10:05
# author: YanJP

import numpy as np
seed=1
np.random.seed(seed)

KHz=1e3
MHz=1e6
B=118*KHz  #132最好
Pmax=25  #W ---> 44dBm  传输速率：[19.7110071  20.1255056  19.15279786 19.29318813]
bps_max=23
N0=3.981e-20  #-174dBm
max_transit_bits=236e6  # 250Mbit 不能超过它 否则就认定为feasible
# -----------------------------------------------------

K=4  #用户数
N=5 # 基站数

T_tile=25  # 3*3*3
N_fov=8  # #每个fov里面的tile数量
len_Si=4

env=None

bitrate_levels=5
bitrates=np.array([30e6,90e6,180e6,250e6,400e6])/T_tile  #[0.3e6,0.7e6,1.6e6, 3.7e6, 8.6e6, 20e6]   [1e6,2.5e6,5e6,8e6,16e6]

e=0.5  # 转一个码率等级所消耗的能量

###-----------------------------Agent设置----------------------
action_dim=bitrate_levels
state_dim=T_tile +1 # 加上distance

### -----------------------------转码设置------------------------
capability=0.1e9  # 0.5G cycle/s
data_one_cycle=0.5 #0.05K bit/cycle
Bit_max=236e6 # UE最多处理40Mbit的数据



### -------------------------------Zipf设置------------------------
Ds=[0.5,0.8,1.5,2.5]
# Ds=[1,1,1,1]
# Ds=[0,0,0,0]


D_matrix = np.random.choice(Ds, size=(T_tile, K))

a=2.8

def get_hot_zipf():
    # 生成Zipf分布的概率
    zipf_probs = np.random.zipf(a, T_tile)
    # 归一化概率，使其总和为1
    zipf_probs_normalized = zipf_probs / np.sum(zipf_probs)
    # 打印生成的概率
    # print("生成的Zipf分布概率：", zipf_probs_normalized)
    return zipf_probs_normalized
tile_prob=get_hot_zipf()
def request():
    random_request = np.random.choice(np.arange(T_tile), size=N_fov, replace=False, p=tile_prob)  #, replace=False表示不重复抽取
    return random_request

### ----------------------------无线信道-----------------------------------------------
# 其中p为第K个用户的发射功率，g为第n个用户的大尺度信道增益，包括路径损耗和阴影，
# h~CN( 0 , 1)为第K个用户的瑞利衰落系数，N0为噪声功率谱密度。
def wireless_Channel(K, d):  #
    path_loss = 128.1 + 37.6 * np.log10(d / 1000)  # dB    小区半径为500m
    # the shadowing factor is set as 6 dB.
    shadow_factor = 4  # dB
    # h_large = 10 ** (-(path_loss ) / 10)
    sigma = np.sqrt(1 / 2)
    h_small = sigma * np.random.randn(K,N) + 1j * sigma * np.random.randn(K,N)
    H_gain = -path_loss -shadow_factor * h_small
    # h = abs(h_small) ** 2
    # snr = h_large * h * p / (N0_dBm * W)
    return H_gain

def get_h():
    H = np.zeros((K, N), dtype=complex)  # Create a K x N zero matrix
    average_power_loss = 1e-4
    sigma = np.sqrt(average_power_loss / 2)

    # Generating channel matrix H with complex Gaussian entries
    for i in range(K):
        h_real = sigma * np.random.randn(N, 1)
        h_imag = sigma * np.random.randn(N, 1)
        H[i, :] = h_real.flatten() + 1j * h_imag.flatten()
    return H
import cvxpy as cp

def cvx_W():
    Kr = h.shape[0]  # Number of users
    D = np.eye(N)
    var = 1e-9
    gamma_dB = 30  # SINR in dB
    gamma = 10 ** (gamma_dB / 10)  # Convert dB to linear scale
    gammavar = gamma * var
    POWER = 0.5
    W = cp.Variable((N, Kr), complex=True)

    # Problem definition
    constraints = []
    for k in range(Kr):
        hkD = np.zeros((Kr, N), dtype=complex)
        for i in range(Kr):
            hkD[i, :] = h[k, :] @ D
        # constraints.append(cp.imag(hkD[k, :] @ W[:, k]) == 0)  # Useful link is real-valued
        constraints.append(cp.real(hkD[k, :] @ W[:, k]) >= cp.norm(
            [1] + hkD[k, :] @ W[:, np.r_[0:k, k + 1:Kr]] / np.sqrt(var)) * np.sqrt(gammavar))

    for a in range(N):
        constraints.append(cp.norm(W[a, :], 'fro') <= np.sqrt(POWER))  # 等式左边求了根号，所以对右边也求

    problem = cp.Problem(cp.Minimize(0), constraints)
    # 指定求解器 ['ECOS', 'ECOS_BB', 'MOSEK', 'OSQP', 'SCIPY', 'SCS'] ECOS_BB这个效果可以
    solver = cp.ECOS_BB
    problem.solve(solver=solver)

    # Analyzing results
    feasible = 'optimal' in problem.status
    print(f"feasible: {feasible}")

    if feasible:
        Wsolution = W.value
        powers = []
        for n in range(N):
            powers.append(np.linalg.norm(Wsolution[n, :]) ** 2)
        # p = np.linalg.norm(Wsolution, 'fro')
        print(f"Power: ", powers)
        sinr = np.zeros(K)
        for k in range(K):
            noise = h[k, :] @ Wsolution[:, np.r_[0:k, k + 1:Kr]]
            sinr[k] = np.abs(h[k, :] @ Wsolution[:, k]) ** 2 / (np.linalg.norm(noise) ** 2 + var)
        print("SINR of Users:")
        print(sinr)
        sinrdB = 10 * np.log10(sinr)
        print("SINR in dB:")
        print(sinrdB)
        print("bps:")
        print(np.log2(1 + sinr))
        return Wsolution
h=get_h()
W_solution=np.ones((N,len_Si))

def watconvert(power_dbm):
    power_watt = np.power(10, (power_dbm - 30) / 10)
    return power_watt
afa=5
beta=10
gamma=1.5
def get_QoE(D,bitrate):
    Qoe=afa/D*np.log(beta*bitrate/max(bitrates)+gamma)
    return Qoe
lambda1=0.6  # 训练用的0.6
def get_energy(max_bitratel,bitrate_l):
    ek=1.5
    return ek*(max_bitratel-bitrate_l)

if __name__ == '__main__':
    # x=get_hot_zipf()
    # print(x)
    # print(request())
    h=wireless_Channel(K,200)
    print(h)