# %%
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import ros_numpy
from sensor_msgs.msg import Image
from loguru import logger
from matplotlib.collections import LineCollection as PltLineCollection
import cv2


#%% Camera
class Imshow(object):
    def __init__(self, ax):
        self._ax = ax
        self._imshow = None

    @property
    def artists(self):
        return [self._imshow]

    def draw(self, im, **kwargs):
        if self._imshow is None:
            self._imshow = self._ax.imshow(im, **kwargs)
            self._ax.get_xaxis().set_visible(False)
            self._ax.get_yaxis().set_visible(False)
        else:
            self._imshow.set_data(im)


class BatchLineCollection(object):
    def __init__(self, ax):
        self._ax = ax
        self._lc = None

    @property
    def artists(self):
        return [self._lc]

    def draw(self, x, y, **kwargs):
        segments = []
        for x_i, y_i in zip(x, y):
            xy_i = np.stack([x_i, y_i], axis=1)
            xy_i = xy_i.reshape(-1, 1, 2)
            segments_i = np.hstack([xy_i[:-1], xy_i[1:]])
            segments.append(segments_i)
        segments = np.concatenate(segments, axis=0)

        if self._lc is None:
            self._lc = PltLineCollection(segments)
            self._ax.add_collection(self._lc)
        else:
            self._lc.set_segments(segments)
        if 'color' in kwargs:
            self._lc.set_color(np.reshape(kwargs['color'], [len(segments), -1]))
        if 'linewidth' in kwargs:
            self._lc.set_linewidth(kwargs['linewidth'])
        self._lc.set_joinstyle('round')
        self._lc.set_capstyle('round')


# Camera paramters
fx, fy, cx, cy = 384.944396973, 384.575073242, 309.668579102, 243.29864502
dim = (640, 480)
K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
D = np.array([[-0.0548635944724, 0.0604563839734, -0.00111321196891, -4.80580529256e-05, -0.0191334541887]]).T


def imrectify(img):
    return cv2.undistort(img, K, D)


def project_points(xy):
    """
    :param xy: [batch_size, horizon, 2]
    :return: [batch_size, horizon, 2]
    """
    batch_size, horizon, _ = xy.shape

    # camera is ~0.35m above ground
    xyz = np.concatenate([xy, -0.35 * np.ones(list(xy.shape[:-1]) + [1])], axis=-1)  # 0.35
    rvec = tvec = (0, 0, 0)
    camera_matrix = K
    dist_coeffs = D

    # x = y, y = -z, z = x
    xyz[..., 0] += 0.6  # NOTE(greg): shift to be in front of image plane
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs)
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def plot_trajectories(img, traj):
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    fpv_imshow = Imshow(ax)
    fpv_batchline = BatchLineCollection(ax)

    fpv_imshow.draw(np.flipud(imrectify(img)), origin='lower')

    im_lims = img.shape
    # xy = np.array([[[0, 0], [1, 0]], [[0, 0], [5, -1]], [[0, 0], [5.5, 3]]])
    pixels = project_points(traj)
    x_list, y_list = [], []
    for pix in pixels:
        pix_lims = (480., 640.)

        assert pix_lims[1] / pix_lims[0] == im_lims[1] / float(im_lims[0])
        resize = im_lims[0] / pix_lims[0]

        pix = resize * pix
        x_list.append(im_lims[1] - pix[:, 0])
        y_list.append(im_lims[0] - pix[:, 1])
    fpv_batchline.draw(x_list, y_list, linewidth=3.)


# %%
HORIZON = 8
STEP_TIME = 0.25
GAP_TIME = 0.5
COLL_DIST = 0.7
OBS_DIST_MIN = 1
OBS_DIST_MAX = 3.5

TOPICS = {'img': '/camera/color/image_raw',
          'odo': '/odometry/filtered',
          'lidar': '/scan'}

bag_name = '../data/circles/21-03-17/16-59-30.bag'

# %% Read bag
img_t = []

odo_t = []
pos = []
yaw = []
cmd = []

lidar_t = []
collision = []
obs_front = []
ranges = []

with rosbag.Bag(bag_name) as bag:
    for topic, msg, t in bag.read_messages(TOPICS.values()):
        if topic == TOPICS['img']:
            img_t.append(msg.header.stamp.to_nsec())
        elif topic == TOPICS['odo']:
            odo_t.append(msg.header.stamp.to_nsec())
            pos.append([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z])
            z = msg.pose.pose.orientation.z
            w = msg.pose.pose.orientation.w
            yaw.append(np.arctan2(2*z*w, 1 - 2*z*z))
            cmd.append([msg.twist.twist.linear.x,
                        msg.twist.twist.angular.z])
        else:  # lidar
            if len(msg.ranges) != 1009:
                logger.error(f'Scan length is {len(msg.ranges)}')
                continue
            lidar_t.append(msg.header.stamp.to_nsec())
            collision.append(np.min(msg.ranges[144:-144]) < COLL_DIST)  # 0 to 180 degrees
            cam_dist = np.min(msg.ranges[304:-304])  # 40 to 140 degrees
            obs_front.append((cam_dist < OBS_DIST_MAX) & (cam_dist > OBS_DIST_MIN))
            ranges.append(msg.ranges)

img_t = np.array(img_t)

odo_t = np.array(odo_t)
pos = np.array(pos).T
yaw = np.array(yaw)
cmd = np.array(cmd).T

lidar_t = np.array(lidar_t)
collision = np.array(collision)
obs_front = np.array(obs_front)
ranges = np.array(ranges)

# %% Sample time
start_time = np.max([img_t[0], odo_t[0], lidar_t[0]])
end_time = np.min([img_t[-1], odo_t[-1], lidar_t[-1]])
inds = np.searchsorted(img_t, [start_time, end_time - int(STEP_TIME*1e9)])

# Sample time based on img frame
t0 = img_t[inds[0]]
sample_time = [t0]
for t in img_t[inds[0]:inds[1]]:
    if t - t0 > GAP_TIME * 1e9:  # half a second
        sample_time.append(t)
        t0 = t
sample_time = np.array(sample_time)

sample_sequence = [sample_time]
for i in range(HORIZON):
    sample_sequence.append(sample_time + int((i+1) * STEP_TIME*1e9))
sample_sequence = np.array(sample_sequence).T.flatten()


# %% Sample messages with interpolation and collision mask
def interpolate(x, xp, yp):
    if yp.ndim == 1:
        return np.interp(x, xp, yp)
    return np.array([np.interp(x, xp, ypi) for ypi in yp])


pos_sample = interpolate(sample_sequence, odo_t, pos).astype(np.float32).T.reshape(-1, HORIZON+1, 3)
yaw_sample = interpolate(sample_sequence, odo_t, yaw).astype(np.float32).reshape(-1, HORIZON+1)
cmd_sample = interpolate(sample_sequence, odo_t, cmd).astype(np.float32).T.reshape(-1, HORIZON+1, 2)
collision_sample = interpolate(sample_sequence, lidar_t, collision).reshape(-1, HORIZON+1)
collision_sample = (collision_sample > 0.5).astype(np.int8)
obs_front_sample = interpolate(sample_sequence, lidar_t, obs_front).reshape(-1, HORIZON+1)
obs_front_sample = (obs_front_sample > 0.5).astype(np.int8)

collision_mask = np.any(collision_sample, axis=-1)
obs_front_mask = np.any(obs_front_sample, axis=-1)

logger.info(f'Total Number of samples: {len(sample_time)}')
logger.info(f'Number with obstacle in front (valid) {np.sum(obs_front_mask)}')
logger.info(f'Number with collision in valid samples: {np.sum(collision_mask & obs_front_mask)}')


# %%
def estimate(pos, yaw, cmd):
    x_ = [pos[0]]
    y_ = [pos[1]]
    yaw_ = [yaw]
    for lin_vel, ang_vel in cmd[1:]:  # Give better estimate of yaw angle than using cmd[:-1]
        yaw_new = yaw_[-1] + ang_vel * STEP_TIME
        yaw_.append(yaw_new)

        r = lin_vel / ang_vel
        x_new = x_[-1] + r*(np.sin(yaw_[-1]) - np.sin(yaw_[-2]))
        y_new = y_[-1] + r*(-np.cos(-yaw_[-1]) + np.cos(yaw_[-2]))
        x_.append(x_new)
        y_.append(y_new)

    pos_ = np.stack((x_, y_, [0] * len(x_)), axis=-1)
    yaw_ = (np.array(yaw_) + np.pi) % (2 * np.pi) - np.pi
    return pos_, yaw_


def yaw_rotmat(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])


def estimate_local(cmd):
    x_ = [0]
    y_ = [0]
    yaw_ = [0]
    for lin_vel, ang_vel in cmd[1:]:  # Give better estimate of yaw angle than using cmd[:-1]
        yaw_new = yaw_[-1] + ang_vel * STEP_TIME
        yaw_.append(yaw_new)

        r = lin_vel / ang_vel
        x_new = x_[-1] + r*(np.sin(yaw_[-1]) - np.sin(yaw_[-2]))
        y_new = y_[-1] + r*(-np.cos(-yaw_[-1]) + np.cos(yaw_[-2]))
        x_.append(x_new)
        y_.append(y_new)

    pos_ = np.stack((x_, y_), axis=-1)
    yaw_ = np.array(yaw_)
    # R = yaw_rotmat(yaw)
    # pos_ = (R @ pos_).T + pos[:2]
    # pos_ = np.concatenate((pos_, np.zeros((HORIZON + 1, 1))), axis=1)
    # yaw_ = (np.array(yaw_) + yaw + np.pi) % (2 * np.pi) - np.pi

    return pos_, yaw_


def ax_plot(ax, title, x, y, y_):
    ax.plot(x, y)
    ax.plot(x, y_)
    ax.legend(['Real', 'Estimate'])
    ax.set_title(title)


def plot(pos, yaw, pos_, yaw_):
    t = np.linspace(0, 2, 9)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax_plot(ax1, "x", t, pos[:, 0], pos_[:, 0])
    ax_plot(ax2, "y", t, pos[:, 1], pos_[:, 1])
    ax_plot(ax3, "yaw", t, yaw*180/np.pi, yaw_*180/np.pi)
    # plt.savefig('pos_predition_by_twist.png')
    plt.show()

    print(np.abs(pos[:, 0] - pos_[:, 0]).mean())
    print(np.abs(pos[:, 1] - pos_[:, 1]).mean())
    print(np.abs(yaw - yaw_).mean() * 180 / np.pi)


def lidar2pos(ranges):
    # -126 ~ 126 degrees, 0.25 degree resolution
    angles = np.linspace(-126, 126, len(ranges)) / 180 * np.pi
    R = np.array([np.cos(angles), np.sin(angles)])
    pos = ranges * R

    return pos.T


def detect_lidar_collision(lidar_pos, p, theta):
    N = lidar_pos.shape[0]
    pos = np.broadcast_to(lidar_pos, (9, N, 2)) - p.reshape(HORIZON+1, 1, 2)  # 9 * N * 2
    R = yaw_rotmat(theta).transpose(2, 0, 1)  # 9 * 2 * 2
    pos = pos @ R
    dist = np.linalg.norm(pos, axis=-1)
    angle = np.arctan2(pos[..., 1], pos[..., 0])

    collision = []
    for dist_i, angle_i in zip(dist, angle):
        collision_i = np.min(dist_i[(angle_i > -np.pi/2) & (angle_i < np.pi/2)]) < COLL_DIST
        collision.append(collision_i)

    return np.array(collision, dtype=np.int8)


# %%
i = 6
pos_diff, yaw_diff = [], []
with rosbag.Bag(bag_name) as bag:
    for topic, msg, _ in bag.read_messages(TOPICS['img']):
        if msg.header.stamp.to_nsec() == sample_time[i]:
            msg.__class__ = Image
            img = ros_numpy.numpify(msg)
            plt.imshow(img, interpolation='nearest')
            plt.show()

            print('pos: ', pos_sample[i])
            print('yaw: ', yaw_sample[i])
            print('cmd: ', cmd_sample[i])
            print('collision: ', collision_sample[i])
            print('obs_front: ', obs_front_sample[i])

            pos_est, yaw_est = estimate(pos_sample[i][0], yaw_sample[i][0], cmd_sample[i])
            print('pos_estimate: ', pos_est)
            print('yaw_estimate: ', yaw_est)

            eps = np.random.normal(0, [0.1, 0.5], size=(9, 9, 2))
            eps[:, 0, :] = 0
            actions = np.concatenate((cmd_sample[i][np.newaxis, :], cmd_sample[i] + eps))

            # Find lidar frame
            ind = np.searchsorted(lidar_t, sample_time[i])
            t_diff = np.abs(lidar_t[ind-1:ind+1] - sample_time[i])
            if t_diff.min() / 1e9 > 0.02:
                logger.info(f'lidar time difference is {t_diff.min() / 1e9}')
                break
            if t_diff[0] < t_diff[1]:
                ind -= 1
            lidar_pos = lidar2pos(ranges[ind])

            pos_aug = []
            for action in actions:
                pos_local, yaw_local = estimate_local(action)
                plot_trajectories(img, np.array(pos_local[np.newaxis, ...]))
                collision = detect_lidar_collision(lidar_pos, pos_local, yaw_local)
                print(collision)

                pos_aug.append(pos_local)

            pos_aug = np.array(pos_aug)

            # plot(pos_sample[i], yaw_sample[i], pos_est, yaw_est)
            # plot(pos_sample[i], yaw_sample[i], pos_local, yaw_local)
            break

# %% Calculate the differences
i = 0
pos_diff, yaw_diff = [], []
pos_local_diff, yaw_local_diff = [], []
collision_diff, collision_naive_diff = [], []
with rosbag.Bag(bag_name) as bag:
    for topic, msg, _ in bag.read_messages(TOPICS['img']):
        if msg.header.stamp.to_nsec() == sample_time[i]:

            pos_est, yaw_est = estimate(pos_sample[i][0], yaw_sample[i][0], cmd_sample[i])
            pos_diff.append(pos_sample[i] - pos_est)
            yaw_diff.append(yaw_sample[i] - yaw_est)

            pos_local, yaw_local = estimate_local(cmd_sample[i])
            ind = np.searchsorted(lidar_t, sample_time[i])

            t_diff = np.abs(lidar_t[ind-1:ind+1] - sample_time[i])
            if t_diff.min() / 1e9 > 0.02:
                logger.info(f'lidar time difference is {t_diff.min()}')
                break
            if t_diff[0] < t_diff[1]:
                ind -= 1
            lidar_pos = lidar2pos(ranges[ind])

            collision = detect_lidar_collision(lidar_pos, pos_local, yaw_local)
            collision_diff.append(collision - collision_sample[i])
            collision_naive_diff.append(np.zeros(9) - collision_sample[i])

            i += 1
            if i >= len(sample_time):
                break

print('pos_diff: ', np.abs(np.concatenate(pos_diff)).mean(axis=0))
print('yaw diff: ', np.abs(yaw_diff).mean() * 180 / np.pi)

print('collision diff: ', np.abs(collision_diff).mean())
print('collision naive diff: ', np.abs(collision_naive_diff).mean())

# print('pos local diff: ', np.abs(np.concatenate(pos_local_diff)).mean(axis=0))
# print('yaw local diff: ', np.abs(yaw_local_diff).mean() * 180 / np.pi)

# %%
