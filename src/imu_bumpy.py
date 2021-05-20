# %%
import numpy as np
import matplotlib.pyplot as plt
import rosbag
from sensor_msgs.msg import Imu, LaserScan

# %%
bag_name = './first_everything.bag'
# bag_name = './sharp_turn.bag'
topics = ['/scan', '/imu_3dm/imu']

FIRST_TIME = False

t_scan = []
scan = []
t_imu = []
imu_angular = []
imu_linear = []
with rosbag.Bag(bag_name) as bag:
    for topic, msg, t in bag.read_messages():
        if not FIRST_TIME:
            initial_time = t
            FIRST_TIME = True
        if topic == topics[0]:
            t_scan.append((t - initial_time).to_sec())
            scan.append(msg.ranges)
        elif topic == topics[1]:
            t_imu.append((t - initial_time).to_sec())
            imu_angular.append([msg.angular_velocity.x,
                                msg.angular_velocity.y,
                                msg.angular_velocity.z])
            imu_linear.append([msg.linear_acceleration.x,
                               msg.linear_acceleration.y,
                               msg.linear_acceleration.z])

# %%
t_scan = np.array(t_scan)
t_imu = np.array(t_imu)
scan = np.array(scan)
imu_angular = np.array(imu_angular)
imu_linear = np.array(imu_linear)

# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
fig.suptitle('Imu angular velocity')
ax1.plot(t_imu, imu_angular[:, 0])
ax1.set_ylabel('x')
ax2.plot(t_imu, imu_angular[:, 1])
ax2.set_ylabel('y')
ax3.plot(t_imu, imu_angular[:, 2])
ax3.set_ylabel('z')

mag = np.linalg.norm(imu_angular[:, :-1], axis=1)
MAG_THRESHOLD = 0.3
collision = mag > MAG_THRESHOLD
# ax4.plot(t_imu, collision)
ax4.set_ylabel('bumpy')

collision_ind = np.where(mag > MAG_THRESHOLD)[0]
t_imu_filter = t_imu[collision_ind]
inds = np.where(np.diff(t_imu_filter) > 1)[0]
print("Collision segments after connecting close parts: ", inds.size)

# Remove single bumpy signal
tail = collision_ind[np.concatenate((inds, [len(t_imu_filter) - 1]))]
head = collision_ind[np.concatenate(([0], inds+1))]
single_mask = (tail - head) > 1
tail = tail[single_mask]
head = head[single_mask]
print("Collision segments after filter out single value: ", head.size)
collision_filter = np.zeros_like(collision)
for h, t in zip(head, tail):
    collision_filter[h:t] = 1
ax4.plot(t_imu, collision_filter)

plt.savefig('imu_angular.png')

# %%
t_seg = np.stack((t_imu[head], t_imu[tail]), axis=1).flatten()
minute = (t_seg // 60).astype(np.uint8).astype(str)
second = (t_seg % 60).astype(str)
print(np.char.add(np.char.add(minute, np.array(['m'] * t_seg.size)), second).reshape(-1, 2))

# %%
