# %%
import tensorflow as tf
import numpy as np

# %%
tf.compat.v1.enable_eager_execution()

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), input_shape=(96, 128, 3), activation='relu', name='obs_im/conv0'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', name='obs_im/conv1'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', name='obs_im/conv2'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', name='obs_im/dense0'),
    tf.keras.layers.Dense(128, activation=None, name='obs_im/dense1'),
])

print(model.summary())

# %%
def yaw_rotmat(yaw):
    return tf.reshape([tf.cos(yaw), -tf.sin(yaw),
                       tf.sin(yaw), tf.cos(yaw)], (2, 2))


x = tf.constant([[1, 0]], dtype=tf.float32)
y = tf.matmul(x, yaw_rotmat(-np.pi/2))

print(y)

# %% See if all the Lidar data has the same length (1009)
import rosbag
from sensor_msgs.msg import LaserScan

bag_name = '../data/circles/21-04-02/circle1.bag'

with rosbag.Bag(bag_name) as bag:
    for _, msg, t in bag.read_messages('/scan'):
        if len(msg.ranges) != 1009:
            print(msg)
            print(len(msg.ranges))


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

square = [[0, 0], [1, 0], [1, 1], [0, 1]]
path = mpltPath.Path(square)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
pos = path.vertices
pos = np.concatenate((pos, pos[0:1]))
ax.plot(pos[:, 0], pos[:, 1], 'r')

point = [0.5, 0.19]
ax.plot(point[0], point[1], 'bo')
print(path.contains_point(point, radius=-0.4))

# %%
