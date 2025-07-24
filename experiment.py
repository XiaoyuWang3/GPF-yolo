import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

A = np.array([[0, 10], [-10, 0]])
B = np.array([[0], [1]])
N = 4 # N个智能体
M = 4 # M条边
edges = [(0, 1), (1, 2), (1, 3), (2, 3)] # 边的表示
edge_map = {(0, 1):0, (1, 0):0, (1, 2):1, (2, 1):1, (1, 3):2, (3, 1):2, (2, 3):3, (3, 2):3}
neighbors = [{1}, {0, 2, 3}, {1, 3}, {1, 2}]
degrees = [1, 3, 2, 2]
D = np.array([[ 1, 0, 0, 0],
              [-1, 1, 1, 0],
              [0, -1, 0, 1],
              [0, 0, -1, -1]])
L = D @ D.T # 拉普拉斯矩阵
I = np.eye(2)  # 二维单位矩阵
P = solve_continuous_are(A, B, I, 1) # 解ARE方程
Gamma = P @ B @ B.T @ P
eig_Gamma = np.linalg.eigvals(Gamma) # 计算Gamma的特征值
eig_P = np.linalg.eigvals(P) # 计算P的特征值
k = np.max(eig_Gamma) / np.min(eig_P)  # λ_max(Γ)/λ_min(P)
K = -B.T @ P

# 初始化智能体位置状态
np.random.seed(42)  #42
x = np.random.uniform(0, 10, (N, 2))  # 位置状态
#print(x)
sign = 2
zeta = np.random.uniform(0.1,0.5, M)
zeta_max = zeta
theta = np.random.uniform(0.1, 0.5, M)
rho = np.random.uniform(0, 10, M)
rho_bar = np.copy(rho)

# 仿真参数
dt = 0.001  # 仿真步长
T = 3.0  # 仿真时间
steps = int(T / dt)
t = 0

# 记录变量
x_history = np.zeros((steps, N, 2))
x_history[0] = x
event_history = {i: [] for i in range(M)}
zeta_history = np.zeros((steps, M))
theta_history = np.zeros((steps, M))
last_event = np.zeros(M)  # 每条边上一次事件时间

z_real = {}
z_hat = {}
e = np.zeros((M, 2))
d_e = np.zeros((M, 2))
for (i, j), idx in edge_map.items():
    if idx < M:  # 只处理无向边
        z_real[idx] = x[i] - x[j]
print(z_real)

for idx in range(M):
    z_hat[idx] = z_real[idx]
print(z_hat)
for step in range(steps):
    t = dt * step
    u = np.zeros(N)
    for i in range(N):
        for j in neighbors[i]:
            # 找到对应的边索引
            idx = edge_map[(i, j)]
            u[i] += K @ (zeta[idx] * z_hat[idx])  # 结果自动为标量

    for i in range(N):
        # 使用元素级乘法替代矩阵乘法
        dx = A @ x[i] + B.T * u[i]
        dx = dx[0]
        x[i] += dt * dx

    for idx in range(M):
        dz_hat = A @ z_hat[idx]
        z_hat[idx] += dz_hat * dt

    for (i, j), idx in edge_map.items():
        #if idx < M:  # 只处理无向边
        z_real[idx] = x[i] - x[j]
    for idx in range(M):
        # e = z_real - z_hat
        #e[idx] = z_real[idx] - z_hat[idx]
        d_e[idx] = e[idx] @ A + B.T * u[idx]
        e[idx] = d_e[idx] * dt

    for idx in range(M):
        # 计算dzeta
        dzeta = z_hat[idx].T @ Gamma @ z_hat[idx]
        temp = zeta[idx]
        zeta[idx] += dzeta * dt
        zeta_max[idx] = max(temp, zeta[idx])
        # 计算dtheta
        dtheta = e[idx].T @ Gamma @ e[idx]
        theta[idx] += dtheta * dt

    for edge_idx in range(M):
        # 获取对应的智能体对
        i, j = None, None
        for (a, b), idx_val in edge_map.items():
            if idx_val == edge_idx:
                i, j = a, b
                break

        # 计算触发函数 - 添加数值检查
        alpha_i = 0.5
        alpha_j = 0.5

        # 计算触发函数
        #term1 = (alpha_i / (2 * degrees[i])) * zeta[edge_idx] * z_hat[edge_idx].T @ Gamma @ z_hat[edge_idx]
        #term2 = zeta[edge_idx] * z_hat[edge_idx].T @ Gamma @ e[edge_idx]
        #term3 = theta[edge_idx] * e[edge_idx].T @ Gamma @ e[edge_idx]

        # Phi_ij = term1 + term2 - term3

        # 检查睡眠时间

        # a = (alpha / degree) * zeta + degree * (degree - 1) * zeta_max
        a = (degrees[i] / alpha_i) * zeta[edge_idx] + degrees[i] * (degrees[i] - 1) * zeta_max[i]
        # b = 2 * ((alpha / degree) * zeta + 1)
        b = 2 * ((degrees[i] / alpha_i) * zeta[edge_idx] + 1)
        # c =  (alpha / degree) * zeta + 2 * theta
        c = (degrees[i] / alpha_i) * zeta[edge_idx] + 2 * theta[edge_idx]
        # drho = -np.sign(rho) * k * (a * rho**2 + b * rho + c) * dt
        # while rho[edge_idx] > 0:
        drho = -np.sign(rho[edge_idx]) * k * (a * rho[edge_idx] ** 2 + b * rho[edge_idx] + c)
        rho[edge_idx] += drho * dt
        #if edge_idx==1:
            #print(rho[edge_idx])
        if rho[edge_idx] <= 0:
        #if t % 0.2 ==0:
            # 计算触发函数
            term1 = (alpha_i / (2 * degrees[i])) * zeta[edge_idx] * z_hat[edge_idx].T @ Gamma @ z_hat[edge_idx]
            term2 = zeta[edge_idx] * z_hat[edge_idx].T @ Gamma @ e[edge_idx]
            term3 = theta[edge_idx] * e[edge_idx].T @ Gamma @ e[edge_idx]

            Phi_ij = term1 + term2 - term3
            #print(Phi_ij)
            if Phi_ij > 0:
                continue
            # 触发事件
            rho[edge_idx] = 0
            event_history[edge_idx].append(t)
            last_event[edge_idx] = t

            # 重置估计器
            #z_hat[edge_idx] = z_real[edge_idx]
            z_real[edge_idx] = x[i] - x[j]
            #z_real[edge_idx] = -z_real[edge_idx]
            #z_hat[edge_idx] = x[j] - x[i]

            # 重置误差
            e[edge_idx] = np.zeros(2)

            # 重置rho (随机值)
            #rho[edge_idx] = np.random.uniform(0, 10)
            #if edge_idx==1:
            #    print("重置")
            #    print(t)
            #rho[edge_idx] = rho_bar[edge_idx]
            rho[edge_idx] = np.random.uniform(0,10)
            #if edge_idx==1:
            #    print(rho_bar[edge_idx])

    x_history[step] = x.copy()
    zeta_history[step] = zeta.copy()
    theta_history[step] = theta.copy()

# 可视化结果
plt.figure(figsize=(12, 10))

# 1. 智能体轨迹
plt.subplot(2, 2, 1)
colors = ['r', 'g', 'b', 'y']
for i in range(N):
    plt.plot(x_history[:, i, 0], x_history[:, i, 1], color=colors[i], label=f'Agent {i + 1}')
    plt.scatter(x_history[0, i, 0], x_history[0, i, 1], marker='*', s=100, color=colors[i])
plt.title('Agent Trajectories')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)

# 2. 事件间隔
plt.subplot(2, 2, 2)
'''for edge_idx in range(M):
    events = np.array(event_history[edge_idx])
    if len(events) > 1:
        intervals = events[1:] - events[:-1]
        plt.stem(events[:-1], intervals, linefmt=f'C{edge_idx}-', markerfmt=f'C{edge_idx}o',
                 basefmt=f'C{edge_idx}', label=f'Edge {edge_idx + 1}', use_line_collection=True)
'''
edge_idx = 1
events = np.array(event_history[edge_idx])
if len(events) > 1:
    intervals = events[1:] - events[:-1]
    plt.stem(events[:-1], intervals, linefmt=f'C{edge_idx}-', markerfmt=f'C{edge_idx}o',
            basefmt=f'C{edge_idx}', label=f'Edge {edge_idx + 1}', use_line_collection=True)
plt.title('Event Intervals')
plt.xlabel('Time')
plt.ylabel('Interval Duration')
plt.legend()
plt.grid(True)

# 3. Zeta参数变化
plt.subplot(2, 2, 3)
time_axis = np.arange(0, T, dt)
for edge_idx in range(M):
    plt.plot(time_axis, zeta_history[:, edge_idx], label=f'Edge {edge_idx + 1}')
plt.title('ζ')
plt.xlabel('Time')
plt.ylabel('ζ Value')
plt.legend()
plt.grid(True)

# 4. Theta参数变化
plt.subplot(2, 2, 4)
for edge_idx in range(M):
    plt.plot(time_axis, theta_history[:, edge_idx], label=f'Edge {edge_idx + 1}')
plt.title('θ')
plt.xlabel('Time')
plt.ylabel('θ Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()