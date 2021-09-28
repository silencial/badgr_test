import argparse
from os import walk
from pathlib import Path

import rosbag
import ros_numpy
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import PIL
import tensorflow as tf
from loguru import logger


HORIZON = 8
STEP_TIME = 0.25
TOPICS = {'img': '/camera/color/image_raw',
          'odo': '/odometry/filtered'}


def plot_info(cmd, bag_name):
    fig_name = bag_name.with_suffix('.png')

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(cmd[:, 0])
    ax1.set_title('linear vel')
    ax2.plot(cmd[:, 1])
    ax2.set_title('angular vel')

    plt.savefig(fig_name)


def read_boundary():
    bag_name = 'data/bumpy/21-09-20/outer_loop_decker_quad_ccw.bag'
    topics = ['/odometry/filtered']
    pos = []
    with rosbag.Bag(bag_name) as bag:
        for _, msg, _ in bag.read_messages(topics):
            pos.append([msg.pose.pose.position.x,
                        msg.pose.pose.position.y])

    path = mpltPath.Path(pos)
    return path


def read_bag(bag_name):
    img_t = []
    odo_t, pos, yaw, cmd = [], [], [], []
    with rosbag.Bag(bag_name) as bag:
        for topic, msg, _ in bag.read_messages(TOPICS.values()):
            if topic == TOPICS['img']:
                img_t.append(msg.header.stamp.to_nsec())
            else:  # odo
                odo_t.append(msg.header.stamp.to_nsec())
                pos.append([msg.pose.pose.position.x,
                            msg.pose.pose.position.y])
                yaw.append([msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w])
                cmd.append([msg.twist.twist.linear.x,
                            msg.twist.twist.angular.z])

    img_t = np.array(img_t)

    odo_t = np.array(odo_t)
    pos = np.array(pos)
    yaw = np.array(yaw)
    cmd = np.array(cmd)

    yaw = np.arctan2(2 * yaw[:, 0] * yaw[:, 1], 1 - 2 * yaw[:, 0] * yaw[:, 0])

    assert (np.diff(img_t) >= 0).all(), "img Timestamp not sorted"
    assert (np.diff(odo_t) >= 0).all(), "odo Timestamp not sorted"

    return img_t, odo_t, pos, yaw, cmd


def sample_times(img_t, odo_t, gap_time):
    # NOTE: Filter out segements at start/end when the car is not moving
    # t = np.where(np.abs(cmd[0]) > 0.2)[0] # 1500 is the neutral value for v
    # start, end = t[0], t[-1]

    start_time = max(img_t[0], odo_t[0])
    end_time = min(img_t[-1], odo_t[-1])
    inds = np.searchsorted(img_t, [start_time, end_time - int(STEP_TIME*1e9)])

    # Sample time based on img frame
    t0 = img_t[inds[0]]
    sample_time = [t0]
    for t in img_t[inds[0]:inds[1]]:
        if t - t0 > gap_time * 1e9:
            sample_time.append(t)
            t0 = t
    sample_time = np.array(sample_time)

    return sample_time


def interpolate(x, xp, yp):
    if yp.ndim == 1:
        return np.interp(x, xp, yp)
    y = [np.interp(x, xp, ypi) for ypi in yp.T]
    return np.array(y)


def sample_bumpy(t, imu_t, bumpy):
    inds = np.searchsorted(imu_t, t)
    bumpy_sample = np.zeros_like(t)
    for bumpy_, inds_ in zip(bumpy_sample, inds):
        for i in range(HORIZON):
            bumpy_[i+1] = np.any(bumpy[inds_[i]:inds_[i+1]])

    return bumpy_sample


def sample_msgs(t, odo_t, pos, yaw, cmd):
    t_ = [t]
    for i in range(HORIZON):
        t_.append(t + int((i+1) * STEP_TIME*1e9))
    t_ = np.array(t_).T

    pos_sample = interpolate(t_, odo_t, pos).transpose(1, 2, 0)
    yaw_sample = interpolate(t_, odo_t, yaw)
    cmd_sample = interpolate(t_, odo_t, cmd).transpose(1, 2, 0)

    return pos_sample, yaw_sample, cmd_sample


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(msg, cmd, bumpy):
    img = get_img(msg)
    cmd = cmd.astype(np.float32)
    bumpy = bumpy.astype(np.uint8)

    done = np.zeros(HORIZON)
    done = done.astype(np.uint8)

    feature = {
        'inputs/images/rgb_left': _bytes_feature(img.tobytes()),
        'inputs/bumpy': _bytes_feature(bumpy[0].tobytes()),
        'outputs/bumpy': _bytes_feature(bumpy[1:].tobytes()),
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

    assert img.dtype == np.uint8, 'Image dtype not np.uint8'
    return im


def estimate(pos, yaw, cmd):
    x_ = [pos[0]]
    y_ = [pos[1]]
    yaw_ = [yaw]
    for lin_vel, ang_vel in cmd[1:]:
        yaw_new = yaw_[-1] + ang_vel*STEP_TIME
        yaw_.append(yaw_new)

        if abs(ang_vel) < 1e-3:
            ang_vel = 1e-3

        r = lin_vel / ang_vel
        x_new = x_[-1] + r*(np.sin(yaw_[-1]) - np.sin(yaw_[-2]))
        y_new = y_[-1] + r*(-np.cos(-yaw_[-1]) + np.cos(yaw_[-2]))
        x_.append(x_new)
        y_.append(y_new)

    pos_ = np.stack((x_, y_), axis=-1)
    return pos_


def save_tfrecord(bag_name, boundary, t, pos, yaw, cmd):
    bag = rosbag.Bag(bag_name, 'r')
    ind = 0
    tf_name = bag_name.with_suffix('.tfrecord')
    bumpy_all_num, bumpy_partial_num, bumpy_not_num = 0, 0, 0

    AUG = 10  # augmentation number
    with tf.io.TFRecordWriter(str(tf_name)) as writer:
        for _, msg, _ in bag.read_messages(TOPICS['img']):
            if ind >= len(t):
                break
            if msg.header.stamp.to_nsec() == t[ind]:
                eps = np.random.normal(0, [0.2, 0.7], size=(AUG, 9, 2))
                eps[0, ...] = 0
                eps[:, 0, :] = 0

                for eps_i in eps:
                    # cmd_i = cmd[ind] + eps_i  # randomize on existing action
                    cmd_i = eps_i.copy()
                    cmd_i[:, 0] += 1  # randomize on 1m/s vel and 0 steering angle
                    pos_i = estimate(pos[ind, 0, :], yaw[ind, 0], cmd_i)

                    bumpy_i = boundary.contains_points(pos_i)
                    if bumpy_i.all():
                        bumpy_all_num += 1
                    elif bumpy_i.any():
                        bumpy_partial_num += 1
                    else:  # not bumpy
                        bumpy_not_num += 1

                    example = serialize_example(msg, cmd_i, bumpy_i)
                    writer.write(example)

                ind += 1

    logger.info(f'Total Number of samples: {len(t)}')
    logger.info(f'Augmentaion numbers: {AUG}')
    logger.info(f'Number of all bumpy {bumpy_all_num}')
    logger.info(f'Number of partial bumpy {bumpy_partial_num}')
    logger.info(f'Number of no bumpy {bumpy_not_num}')

    bag.close()


def bag_to_tfrecord(bag, args, boundary):
    img_t, odo_t, pos, yaw, cmd = read_bag(bag)
    if args.plot:
        plot_info(cmd, bag)
    t = sample_times(img_t, odo_t, args.gap_time)
    pos, yaw, cmd = sample_msgs(t, odo_t, pos, yaw, cmd)
    save_tfrecord(bag, boundary, t, pos, yaw, cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=str, default=None,
                        help='The directory of bagfiles')
    parser.add_argument('-b', '--bumpy_mag',
                        type=float, default=0.3,
                        help='Bumpy magnitude')
    parser.add_argument('-t', '--gap_time',
                        type=float, default=0.5,
                        help='The gap time when sampling')
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='The gap time when sampling')
    args = parser.parse_args()

    path = Path(args.path)

    boundary = read_boundary()

    for dirpath, dirnames, filenames in walk(path, followlinks=True):
        dirpath = Path(dirpath)
        for filename in filenames:
            if filename.endswith('.bag'):
                logger.info(dirpath / filename)
                bag_to_tfrecord(dirpath / filename, args, boundary)
