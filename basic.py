import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Robot parameters
# -----------------------------
N = 3
L = np.ones(N)
q = np.zeros(N)  # initial joint angles

target = np.array([1.5, 1.5])  # target position


# -----------------------------
# Forward kinematics
# -----------------------------
def fk_all(q):
    x, y, a = 0.0, 0.0, 0.0
    xs, ys = [0.0], [0.0]
    for i in range(N):
        a += q[i]
        x += L[i] * np.cos(a)
        y += L[i] * np.sin(a)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# -----------------------------
# Jacobian
# -----------------------------
def jacobian(q):
    J = np.zeros((2, N))
    a = 0.0
    angles = []
    for i in range(N):
        a += q[i]
        angles.append(a)
    for j in range(N):
        dx, dy = 0.0, 0.0
        for k in range(j, N):
            dx -= L[k] * np.sin(angles[k])
            dy += L[k] * np.cos(angles[k])
        J[:, j] = [dx, dy]
    return J


# -----------------------------
# Manipulability
# -----------------------------
def manipulability(J):
    return np.sqrt(np.linalg.det(J @ J.T))


# -----------------------------
# IK loop (point-to-point)
# -----------------------------
dt = 0.05
lam = 0.1
steps = 200

snapshots = []
ee_path = []
manips = []

for i in range(steps):
    xs, ys = fk_all(q)
    x_ee = np.array([xs[-1], ys[-1]])
    ee_path.append(x_ee)

    J = jacobian(q)
    w = manipulability(J)
    manips.append(w)

    if i % 10 == 0:
        snapshots.append((xs.copy(), ys.copy(), w))

    e = target - x_ee
    dq = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(2)) @ e
    q += dt * dq

ee_path = np.array(ee_path)
manips = np.array(manips)

w_th = np.percentile(manips, 20)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 6))

for xs, ys, w in snapshots:
    if w < w_th:
        color, alpha = "red", 0.45
    else:
        color, alpha = "tab:blue", 0.25

    plt.plot(xs, ys, color=color, alpha=alpha, linewidth=1.5)
    plt.scatter(xs, ys, color=color, alpha=alpha, s=18)

# End-effector path
plt.plot(ee_path[:, 0], ee_path[:, 1], color="tab:green", label="End-effector path")

# Target
plt.scatter(target[0], target[1], color="black", marker="x", s=80, label="Target")

plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Point-to-Point Motion with Jacobian-based IK")
plt.legend()
plt.show()
