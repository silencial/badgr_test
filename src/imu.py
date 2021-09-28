# %%
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import pandas as pd

# %%
bag_name = '../data/bumpy/21-08-24/17-51-18.bag'
# bag_name = './sharp_turn.bag'
topic = '/imu_3dm/imu'

FIRST_TIME = True

t_imu = []
imu_angular = []
imu_linear = []
with rosbag.Bag(bag_name) as bag:
    for _, msg, t in bag.read_messages(topics=topic):
        if FIRST_TIME:
            initial_time = t
            FIRST_TIME = False

        t_imu.append((t - initial_time).to_sec())
        imu_angular.append([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])
        imu_linear.append([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z])

t_imu = np.array(t_imu)
imu_angular = np.array(imu_angular)
imu_linear = np.array(imu_linear)


# %%
def filter(data, window_size=30):
    df = pd.DataFrame(data)
    df_mean = df.rolling(window_size, center=True) \
                .mean() \
                .interpolate('pad') \
                .interpolate('bfill')

    # df = (df - df_mean).abs()
    df = df - df_mean
    return df_mean.to_numpy()


def plot(imu_angular, mask):
    ang_x, ang_y, _ = imu_angular.T
    mag = np.linalg.norm([ang_x, ang_y], axis=0)
    mag_filtered = filter(mag)


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15))
    fig.suptitle('Imu angular velocity')
    ax1.plot(t_imu[mask], ang_x[mask])
    ax1.set_ylabel('x')
    ax2.plot(t_imu[mask], ang_y[mask])
    ax2.set_ylabel('y')

    ax3.plot(t_imu[mask], mag[mask])
    ax3.set_ylabel('mag')
    ax4.plot(t_imu[mask], mag_filtered[mask])
    ax4.set_ylabel('mag filtered')


# %%
# mask = t_imu > -1
# mask = (t_imu > 50) & (t_imu < 110)  # smooth
mask = (t_imu > 120) & (t_imu < 150)  # bumpy
plot(imu_angular, mask)

freq = 100  # Hz
dt = 0.3  # seconds
window_size = int(dt * freq)
plot(filter(imu_angular, window_size), mask)

# %%
# MAG_THRESHOLD = 0.3
# bumpy = mag > MAG_THRESHOLD
# # ax4.plot(t_imu, bumpy)
# ax4.set_ylabel('bumpy')

# bumpy_ind = np.where(mag > MAG_THRESHOLD)[0]
# t_imu_filter = t_imu[bumpy_ind]
# inds = np.where(np.diff(t_imu_filter) > 1)[0]
# print("Bumpy segments after connecting close parts: ", inds.size)

# # Remove single bumpy signal
# tail = bumpy_ind[np.concatenate((inds, [len(t_imu_filter) - 1]))]
# head = bumpy_ind[np.concatenate(([0], inds+1))]
# single_mask = (tail - head) > 1
# tail = tail[single_mask]
# head = head[single_mask]
# print("Bumpy segments after filter out single value: ", head.size)
# bumpy_filter = np.zeros_like(bumpy)
# for h, t in zip(head, tail):
#     bumpy_filter[h:t] = 1
# ax4.plot(t_imu, bumpy_filter)

# plt.savefig('imu_angular.png')

# %%
t_seg = np.stack((t_imu[head], t_imu[tail]), axis=1).flatten()
minute = (t_seg // 60).astype(np.uint8).astype(str)
second = np.around((t_seg % 60), 2).astype(str)
print(np.char.add(np.char.add(minute, np.array(['m'] * t_seg.size)), second).reshape(-1, 2))

# %%
mask = (t_imu > 50) & (t_imu < 110)  # smooth
# mask = (t_imu > 120) & (t_imu < 150)  # bumpy
ang_x, ang_y, _ = imu_angular.T
mag = np.linalg.norm([ang_x, ang_y], axis=0)
bumpy = mag > 0.25

data = np.stack((t_imu, mag, bumpy), axis=-1)
np.savetxt('imu_smooth.csv', data[mask], delimiter=',', fmt='%.2f, %.2f, %d')
