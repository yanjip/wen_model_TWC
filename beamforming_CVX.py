# time: 2024/3/9 13:45
# author: YanJP
import numpy as np
import cvxpy as cp
seed=0
np.random.seed(seed)
K = 4  # user number
N = 4  # base station number
var = 1e-8
gamma_dB =60  # SINR in dB
gamma = 10 ** (gamma_dB / 10)  # Convert dB to linear scale
print("log(1+gamma)=",np.log2(1+gamma))
gammavar = gamma * var
POWER = 2.5

H = np.zeros((K, N), dtype=complex)  # Create a K x N zero matrix
average_power_loss = 1e-4
sigma = np.sqrt(average_power_loss / 2)

# Generating channel matrix H with complex Gaussian entries
for i in range(K):
    h_real = sigma * np.random.randn(N, 1)
    h_imag = sigma * np.random.randn(N, 1)
    H[i, :] = h_real.flatten() + 1j * h_imag.flatten()

# H=H.transpose()
Kr = H.shape[0]  # Number of users
D = np.eye(N)

W = cp.Variable((N, Kr), complex=True)

# Problem definition
constraints = []
for k in range(Kr):
    hkD = np.zeros((Kr, N), dtype=complex)
    for i in range(Kr):
        hkD[i, :] = H[k, :] @ D
    constraints.append(cp.imag(hkD[k, :] @ W[:, k]) == 0)  # Useful link is real-valued
    constraints.append(cp.real(hkD[k, :] @ W[:, k]) >= cp.norm([1] + hkD[k, :] @ W[:, np.r_[0:k, k + 1:Kr]] / np.sqrt(var)) * np.sqrt(gammavar))

for a in range(N):
    constraints.append(cp.norm(W[a, :], 'fro') <= np.sqrt(POWER))  # 等式左边求了根号，所以对右边也求

problem = cp.Problem(cp.Minimize(0), constraints)
# 指定求解器 ['ECOS', 'ECOS_BB', 'MOSEK', 'OSQP', 'SCIPY', 'SCS'] ECOS_BB这个效果可以
solver = cp.ECOS_BB
problem.solve(solver =solver)

# Analyzing results
feasible = 'optimal' in problem.status
print(f"feasible: {feasible}")

if feasible:
    Wsolution = W.value
    powers=[]
    for n in range(N):
        powers.append(np.linalg.norm(Wsolution[n,:])**2)
    # p = np.linalg.norm(Wsolution, 'fro')
    print(f"Power: ",powers)
    sinr = np.zeros(K)
    for k in range(K):
        noise = H[k, :] @ Wsolution[:, np.r_[0:k, k + 1:Kr]]
        sinr[k] = np.abs(H[k, :] @ Wsolution[:, k]) ** 2 / (np.linalg.norm(noise) ** 2 + var)
    print("SINR of Users:")
    print(sinr)
    sinrdB = 10 * np.log10(sinr)
    print("SINR in dB:")
    print(sinrdB)
    print("bps:")
    print(np.log2(1 + sinr))


    # 验证
    for k in range(Kr):
        hkD = np.zeros((Kr, N), dtype=complex)
        for i in range(Kr):
            hkD[i, :] = H[k, :] @ D
        if np.real(hkD[k, :] @ Wsolution[:, k]) >= np.linalg.norm([1] + hkD[k, :] @ Wsolution[:, np.r_[0:k, k + 1:Kr]] / np.sqrt(var)) * np.sqrt(gammavar):
            print(True)
        else:
            print(False)
            pass
