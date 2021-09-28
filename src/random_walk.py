# %%
import numpy as np
from matplotlib import pyplot as plt


def estimate(x, y, yaw, dt, lin_vel, ang_vel):
    x_ = [x]
    y_ = [y]

    yaw_ = yaw + np.cumsum(ang_vel) * dt
    yaw_ = np.concatenate(([yaw], yaw_))
    yaw_ = (yaw_ + np.pi) % (2 * np.pi) - np.pi

    r = lin_vel / ang_vel
    x_ = x + np.cumsum((np.sin(yaw_[1:]) - np.sin(yaw_[:-1])) * r)
    y_ = y + np.cumsum((-np.cos(yaw_[1:]) + np.cos(yaw_[:-1])) * r)

    x_ = np.concatenate(([x], x_))
    y_ = np.concatenate(([y], y_))

    return x_, y_, yaw_


def estimate2(x, y, yaw, dt, lin_vel, ang_vel):
    x_ = [x]
    y_ = [y]
    yaw_ = [yaw]
    for i in range(len(lin_vel)):
        yaw_new = yaw_[-1] + ang_vel[i] * dt
        yaw_.append(yaw_new)

        r = lin_vel[i] / ang_vel[i]
        x_new = x_[-1] + r*(np.sin(yaw_[-1]) - np.sin(yaw_[-2]))
        y_new = y_[-1] + r*(-np.cos(-yaw_[-1]) + np.cos(yaw_[-2]))
        x_.append(x_new)
        y_.append(y_new)

    yaw_ = (np.array(yaw_) + np.pi) % (2 * np.pi) - np.pi
    x_ = np.array(x_)
    y_ = np.array(y_)
    return x_, y_, yaw_


lin_vel = 2 + (np.random.rand(100) - 0.5) * 0.1
ang_vel = (np.random.rand(100) - 0.5) * 2

dt = 0.1

x, y, yaw = 0.0, 0.0, 0.0

x_, y_, yaw_ = estimate(x, y, yaw, dt, lin_vel, ang_vel)

x2, y2, yaw2 = estimate2(x, y, yaw, dt, lin_vel, ang_vel)

print(np.linalg.norm(x_-x2), np.linalg.norm(y_-y2), np.linalg.norm(yaw_-yaw2))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
ax1.plot(x_, y_)
ax2.plot(lin_vel)
ax3.plot(ang_vel)

# %%
