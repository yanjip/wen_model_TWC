# time: 2024/3/14 22:13
# author: YanJP
import cvxpy as cp
import numpy as np


def socp(h, gamma, sigma_2, N, K):
    """
    Solve the power minimization problem using SOCP.

    Parameters:
    - h: channel coefficients matrix of size (N, K)
    - gamma: SINR threshold
    - sigma_2: noise power
    - N: number of antennas
    - K: number of users

    Returns:
    - minimum_power: the minimized power
    """
    # Define the complex variable for W
    W = cp.Variable((N, K), complex=True)

    # Objective function
    objective = cp.Minimize(cp.norm(W, 'fro'))

    # Constraints
    constraints = []
    for i in range(K):
        h_i = h[:, i]
        W_i = W[:, i]
        constraints.append(cp.imag(h_i.T @ W_i) == 0)
        constraints.append(cp.sqrt(1 + 1 / gamma) * cp.real(h_i.T @ W_i) >= cp.norm(cp.hstack([h_i.T @ W, sigma_2])))

    # Problem definition
    prob = cp.Problem(objective, constraints)

    solver = cp.ECOS_BB
    prob.solve()

    # Calculate the minimum power
    minimum_power = cp.norm(W, 'fro').value ** 2

    Wsolution = W.value
    powers = []
    for n in range(N):
        powers.append(np.linalg.norm(Wsolution[n, :]) ** 2)
    # p = np.linalg.norm(Wsolution, 'fro')
    print(f"Power: ", powers)
    sinr = np.zeros(K)
    for k in range(K):
        noise = H[k, :] @ Wsolution[:, np.r_[0:k, k + 1:K]]
        sinr[k] = np.abs(H[k, :] @ Wsolution[:, k]) ** 2 / (np.linalg.norm(noise) ** 2 + sigma_2)
    print("SINR of Users:")
    print(sinr)
    sinrdB = 10 * np.log10(sinr)
    print("SINR in dB:")
    print(sinrdB)
    print("bps:")
    print(np.log2(1 + sinr))

    return minimum_power

K = 4  # user number
N = 4  # base station number
H = np.zeros((K, N), dtype=complex)  # Create a K x N zero matrix
# average_power_loss = 1e-4
sigma = np.sqrt(1 / 2)

# Generating channel matrix H with complex Gaussian entries
for i in range(K):
    h_real = sigma * np.random.randn(N, 1)
    h_imag = sigma * np.random.randn(N, 1)
    H[i, :] = h_real.flatten() + 1j * h_imag.flatten()

gamma_dB =10  # SINR in dB
gamma = 10 ** (gamma_dB / 10)
ans=socp(H,gamma,sigma**2,N,K)
print(ans)