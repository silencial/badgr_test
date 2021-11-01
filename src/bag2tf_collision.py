import argparse
import json
from pathlib import Path

import rosbag
import ros_numpy
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
from loguru import logger


def plot_info(pos, yaw, cmd, bag_name):
    fig_name = bag_name.with_suffix('.png')

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
    ax1.plot(cmd[:, 0])
    ax1.set_title('linear vel')
    ax2.plot(cmd[:, 1])
    ax2.set_title('angular vel')
    ax3.plot(yaw)
    ax3.set_title('yaw')
    pos_diff = np.diff(pos, axis=0)
    ax4.scatter(pos_diff[:, 0], pos_diff[:, 1])
    ax4.set_title('pos diff')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    plt.savefig(fig_name)


def read_bag(bag_name, coll_dist, obs_dist_min, obs_dist_max):
    img_t = []
    odo_t, odo, cmd = [], [], []
    lidar_t, ranges = [], []
    with rosbag.Bag(bag_name) as bag:
        for topic, msg, _ in bag.read_messages(TOPICS.values()):
            if topic == TOPICS['img']:
                img_t.append(msg.header.stamp.to_nsec())
            elif topic == TOPICS['odo']:
                odo_t.append(msg.header.stamp.to_nsec())
                odo.append([msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w])
                cmd.append([msg.twist.twist.linear.x,
                            msg.twist.twist.angular.z])
            else:  # lidar
                if len(msg.ranges) != 1009:
                    logger.warning(f'Scan length is {len(msg.ranges)}')
                    continue
                lidar_t.append(msg.header.stamp.to_nsec())
                ranges.append(msg.ranges)

    img_t = np.array(img_t)

    odo_t = np.array(odo_t)
    odo = np.array(odo)
    cmd = np.array(cmd)
    pos = odo[:, :3]
    yaw = np.arctan2(2 * odo[:, 3] * odo[:, 4], 1 - 2 * odo[:, 3] * odo[:, 3])

    lidar_t = np.array(lidar_t)
    ranges = np.array(ranges)
    collision = np.min(ranges[:, 144:-144], axis=1) < coll_dist  # 0 to 180 degrees
    cam_dist = np.min(ranges[:, 304:-304], axis=1)  # 40 to 140 degrees
    obs_front = (cam_dist < obs_dist_max) & (cam_dist > obs_dist_min)  # 40 to 140 degrees

    assert (np.diff(img_t) > 0).all(), "img Timestamp not sorted"
    assert (np.diff(odo_t) > 0).all(), "odo Timestamp not sorted"
    assert (np.diff(lidar_t) > 0).all(), "lidar Timestamp not sorted"

    return img_t, odo_t, lidar_t, pos, yaw, cmd, ranges, collision, obs_front


def sample_times(img_t, odo_t, lidar_t, gap_time):
    # NOTE: Filter out segements at start/end when the car is not moving
    # t = np.where(np.abs(cmd[0]) > 0.2)[0] # 1500 is the neutral value for v
    # start, end = t[0], t[-1]

    start_time = np.max([img_t[0], odo_t[0], lidar_t[0]])
    end_time = np.min([img_t[-1], odo_t[-1], lidar_t[-1]])
    inds = np.searchsorted(img_t, [start_time, end_time - int(STEP_TIME*1e9)])

    # Sample time based on img frame
    t0 = img_t[inds[0]]
    sample_time = [t0]
    for t in img_t[inds[0]:inds[1]]:
        if t - t0 > gap_time * 1e9:  # half a second
            sample_time.append(t)
            t0 = t
    sample_time = np.array(sample_time)

    return sample_time


def interpolate(x, xp, yp):
    if yp.ndim == 1:
        return np.interp(x, xp, yp)
    y = [np.interp(x, xp, ypi) for ypi in yp.T]
    return np.array(y)


def sample_msgs(t, odo_t, lidar_t, pos, yaw, cmd, collision, obs_front):
    t_ = [t]
    for i in range(HORIZON):
        t_.append(t + int((i+1) * STEP_TIME*1e9))
    t_ = np.array(t_).T

    pos_sample = interpolate(t_, odo_t, pos).transpose(1, 2, 0)
    yaw_sample = interpolate(t_, odo_t, yaw)
    cmd_sample = interpolate(t_, odo_t, cmd).transpose(1, 2, 0)
    collision_sample = interpolate(t_, lidar_t, collision)
    collision_sample = (collision_sample > 0.5)
    obs_front = interpolate(t_, lidar_t, obs_front)
    obs_front = (obs_front > 0.5)

    collision_mask = np.any(collision_sample, axis=1)
    obs_front_mask = np.any(obs_front, axis=1)

    logger.info(f'Total Number of samples: {len(t)}')
    logger.info(f'Number with obstacle in front (valid) {np.sum(obs_front_mask)}')
    logger.info(f'Number with collision in valid samples: {np.sum(collision_mask & obs_front_mask)}')

    return pos_sample, yaw_sample, cmd_sample, collision_sample, obs_front_mask


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(msg, pos, yaw, cmd, collision):
    img = get_img(msg)
    pos = pos.astype(np.float32)
    yaw = yaw.astype(np.float32)
    cmd = cmd.astype(np.float32)
    collision = collision.astype(np.uint8)

    done = collision[1:].cumsum() > 0.5
    done = done.astype(np.uint8)

    feature = {
        'inputs/images/rgb_left': _bytes_feature(img.tobytes()),
        'inputs/jackal/position': _bytes_feature(pos[0].tobytes()),
        'outputs/jackal/position': _bytes_feature(pos[1:].tobytes()),
        'inputs/jackal/yaw': _bytes_feature(yaw[0].tobytes()),
        'outputs/jackal/yaw': _bytes_feature(yaw[1:].tobytes()),
        'inputs/collision/close': _bytes_feature(collision[0].tobytes()),
        'outputs/collision/close': _bytes_feature(collision[1:].tobytes()),
        'inputs/commands/linear_velocity': _bytes_feature(cmd[:-1, 0].tobytes()),
        'inputs/commands/angular_velocity': _bytes_feature(cmd[:-1, 1].tobytes()),
        'outputs/done': _bytes_feature(done.tobytes())
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_img(msg):
    msg.__class__ = Image
    img = ros_numpy.numpify(msg)

    # scale
    shape = [96, 128]
    assert (img.shape[0] / img.shape[1] == shape[0] / shape[1])  # maintain aspect ratio
    height, width = shape

    im = PIL.Image.fromarray(img)
    im = im.resize((width, height), PIL.Image.LANCZOS)
    im = np.array(im)

    assert im.dtype == np.uint8, 'Image dtype not np.uint8'
    return im


def yaw_rotmat(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])


def estimate_local(cmd):
    x_ = [0]
    y_ = [0]
    yaw_ = [0]
    for lin_vel, ang_vel in cmd[1:]:  # Give better estimate of yaw angle than using cmd[:-1]
        yaw_new = yaw_[-1] + ang_vel*STEP_TIME
        yaw_.append(yaw_new)

        r = lin_vel / ang_vel
        x_new = x_[-1] + r * (np.sin(yaw_[-1]) - np.sin(yaw_[-2]))
        y_new = y_[-1] + r * (-np.cos(-yaw_[-1]) + np.cos(yaw_[-2]))
        x_.append(x_new)
        y_.append(y_new)

    pos_ = np.stack((x_, y_), axis=-1)
    yaw_ = np.array(yaw_)

    return pos_, yaw_


def lidar2pos(ranges):
    # -126 ~ 126 degrees, 0.25 degree resolution
    angles = np.linspace(-126, 126, len(ranges)) / 180 * np.pi
    R = np.array([np.cos(angles), np.sin(angles)])
    pos = ranges * R

    return pos.T


def detect_lidar_collision(lidar_pos, p, theta, coll_dist):
    N = lidar_pos.shape[0]
    pos = np.broadcast_to(lidar_pos, (9, N, 2)) - p.reshape(HORIZON + 1, 1, 2)  # 9 * N * 2
    R = yaw_rotmat(theta).transpose(2, 0, 1)  # 9 * 2 * 2
    pos = pos @ R
    dist = np.linalg.norm(pos, axis=-1)
    angle = np.arctan2(pos[..., 1], pos[..., 0])

    collision = []
    for dist_i, angle_i in zip(dist, angle):
        collision_i = np.min(dist_i[(angle_i > -np.pi / 2) & (angle_i < np.pi / 2)]) < coll_dist
        collision.append(collision_i)

    return np.array(collision)


def save_tfrecord(bag_name, t, lidar_t, pos, yaw, cmd, ranges, collision, mask, coll_dist):
    bag = rosbag.Bag(bag_name, 'r')
    ind = 0
    tf_name = bag_name.with_suffix('.tfrecord')
    with tf.io.TFRecordWriter(str(tf_name)) as writer:
        for _, msg, _ in bag.read_messages(TOPICS['img']):
            if ind >= len(t):
                break
            if msg.header.stamp.to_nsec() == t[ind]:
                # Write original data
                example = serialize_example(msg, pos[ind], yaw[ind], cmd[ind], collision[ind])
                writer.write(example)

                if not mask[ind]:
                    ind += 1
                    continue

                # Write augmented data
                i = np.searchsorted(lidar_t, t[ind])
                t_diff = np.abs(lidar_t[i-1:i+1] - t[ind])
                if t_diff.min() / 1e9 > 0.02:
                    logger.warning(f'lidar time difference is {t_diff.min() / 1e9}')
                    ind += 1
                    continue
                if t_diff[0] < t_diff[1]:
                    i -= 1
                lidar_pos = lidar2pos(ranges[i])

                if collision[ind].any():
                    eps = np.random.normal(0, [0.25, 0.5], size=(19, 9, 2))
                else:
                    eps = np.random.normal(0, [0.25, 0.5], size=(2, 9, 2))
                eps[:, 0, :] = 0

                for eps_i in eps:
                    cmd_i = cmd[ind] + eps_i
                    pos_local, yaw_local = estimate_local(cmd_i)

                    collision_i = detect_lidar_collision(lidar_pos, pos_local, yaw_local, coll_dist)

                    R = yaw_rotmat(-yaw[ind, 0])
                    pos_i = pos_local@R + pos[ind, 0, :2]
                    pos_i = np.concatenate((pos_i, np.zeros((HORIZON + 1, 1))), axis=1)
                    yaw_i = (yaw_local + yaw[ind] + np.pi) % (2 * np.pi) - np.pi

                    example = serialize_example(msg, pos_i, yaw_i, cmd_i, collision_i)
                    writer.write(example)

                ind += 1

    bag.close()


def bag_to_tfrecord(bag, args):
    img_t, odo_t, lidar_t, pos, yaw, cmd, ranges, collision, obs_front = read_bag(bag, args.coll_dist, args.obs_dist_min, args.obs_dist_max)
    if args.plot:
        plot_info(pos, yaw, cmd, bag)
    t = sample_times(img_t, odo_t, lidar_t, args.gap_time)
    pos, yaw, cmd, collision, obs_front = sample_msgs(t, odo_t, lidar_t, pos, yaw, cmd, collision, obs_front)
    save_tfrecord(bag, t, lidar_t, pos, yaw, cmd, ranges, collision, obs_front, args.coll_dist)


if __name__ == '__main__':
    HORIZON = 8
    STEP_TIME = 0.25

    TOPICS = {'img': '/camera/color/image_raw',
              'odo': '/odometry/filtered',
              'lidar': '/scan'}

    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=str, default=None,
                        help='The directory of bagfiles')
    parser.add_argument('-d', '--coll_dist',
                        type=float, default=0.7,
                        help='Collision radius for lidar')
    parser.add_argument('-o1', '--obs_dist_min',
                        type=float, default=1,
                        help='Close radius for lidar')
    parser.add_argument('-o2', '--obs_dist_max',
                        type=float, default=3.5,
                        help='Close radius for lidar')
    parser.add_argument('-t', '--gap_time',
                        type=float, default=0.5,
                        help='The gap time when sampling')
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='The gap time when sampling')
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        assert path.suffix == '.bag', 'Not a bag file'
        logger.info(path)
        with open(path.with_suffix('.args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        bag_to_tfrecord(path, args)
    else:
        for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
            dirpath = Path(dirpath)
            with open(dirpath / 'args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            for filename in filenames:
                if filename.endswith('.bag'):
                    logger.info(dirpath / filename)
                    bag_to_tfrecord(dirpath / filename, args)
