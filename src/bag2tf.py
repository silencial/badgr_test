import argparse
from os import walk
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

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
    ax1.plot(cmd[0])
    ax1.set_title('linear vel')
    ax2.plot(cmd[1])
    ax2.set_title('angular vel')
    ax3.plot(yaw)
    ax3.set_title('yaw')
    pos_diff = np.diff(pos[:2])
    ax4.scatter(pos_diff[0], pos_diff[1])
    ax4.set_title('pos diff')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    plt.savefig(fig_name)


def read_bag(bag_name, coll_dist):
    img_t = []
    odo = []
    lidar = []
    with rosbag.Bag(bag_name) as bag:
        for topic, msg, t in bag.read_messages(TOPICS.values()):
            if topic == TOPICS['img']:
                img_t.append(msg.header.stamp.to_nsec())
            elif topic == TOPICS['odo']:
                odo.append([msg.header.stamp.to_nsec(),
                            msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w,
                            msg.twist.twist.linear.x,
                            msg.twist.twist.angular.z])
            else:  # lidar
                ranges = msg.ranges
                collision = np.min(ranges[144:-144]) < coll_dist
                lidar.append([msg.header.stamp.to_nsec(),
                              collision])

    img_t = np.array(img_t)

    odo_t = np.array(odo, dtype=np.int64)[:, 0]
    odo = np.array(odo, dtype=np.float64)[:, 1:].T
    pos = odo[:3]
    yaw = np.arctan2(2 * odo[3] * odo[4], 1 - 2 * odo[3] * odo[3])
    cmd = odo[5:]

    lidar = np.array(lidar).T
    lidar_t = lidar[0]
    collision = lidar[1]

    return img_t, odo_t, lidar_t, pos, yaw, cmd, collision


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

    sample_sequence = [sample_time]
    for i in range(HORIZON):
        sample_sequence.append(sample_time + int((i+1) * STEP_TIME*1e9))
    sample_sequence = np.array(sample_sequence).T.flatten()

    return sample_sequence


def interpolate(x, xp, yp):
    if yp.ndim == 1:
        return np.interp(x, xp, yp)
    y = []
    for ypi in yp:
        y.append(np.interp(x, xp, ypi))
    return np.array(y)


def sample_msgs(t, odo_t, lidar_t, pos, yaw, cmd, collision):
    pos_sample = interpolate(t, odo_t, pos).astype(np.float32).T
    yaw_sample = interpolate(t, odo_t, yaw).astype(np.float32)
    cmd_sample = interpolate(t, odo_t, cmd).astype(np.float32).T
    collision_sample = interpolate(t, lidar_t, collision)
    collision_sample = (collision_sample > 0.5).astype(np.uint8)

    collision_mask = np.any(collision_sample.reshape(-1, 9)[:, 1:], axis=-1)

    logger.info(f'Total Number of samples: {len(t)}')
    logger.info(f'Number of samples with collision: {np.sum(collision_mask)}')

    return pos_sample, yaw_sample, cmd_sample, collision_sample


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(img, in_pos, out_pos, in_yaw, out_yaw, in_collision, out_collision, cmd_linear, cmd_angular, done):
    feature = {
        'inputs/images/rgb_left': _bytes_feature(img.tobytes()),
        'inputs/jackal/position': _bytes_feature(in_pos.tobytes()),
        'outputs/jackal/position': _bytes_feature(out_pos.tobytes()),
        'inputs/jackal/yaw': _bytes_feature(in_yaw.tobytes()),
        'outputs/jackal/yaw': _bytes_feature(out_yaw.tobytes()),
        'inputs/collision/close': _bytes_feature(in_collision.tobytes()),
        'outputs/collision/close': _bytes_feature(out_collision.tobytes()),
        'inputs/commands/linear_velocity': _bytes_feature(cmd_linear.tobytes()),
        'inputs/commands/angular_velocity': _bytes_feature(cmd_angular.tobytes()),
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


def save_tfrecord(bag_name, t, pos, yaw, cmd, collision):
    bag = rosbag.Bag(bag_name, 'r')
    ind = 0
    tf_name = bag_name.with_suffix('.tfrecord')
    with tf.io.TFRecordWriter(str(tf_name)) as writer:
        for topic, msg, _ in bag.read_messages(TOPICS['img']):
            if ind >= len(t):
                break
            if msg.header.stamp.to_nsec() == t[ind]:
                img = get_img(msg)

                in_pos = pos[ind]
                out_pos = pos[ind+1:ind+9]

                in_yaw = yaw[ind]
                out_yaw = yaw[ind+1:ind+9]

                in_collision = collision[ind]
                out_collision = collision[ind+1:ind+9]

                cmd_linear = cmd[ind:ind+8, 0]
                cmd_angular = cmd[ind:ind+8, 1]

                done = np.zeros(HORIZON, dtype=np.uint8)

                example = serialize_example(img, in_pos, out_pos, in_yaw, out_yaw, in_collision, out_collision, cmd_linear, cmd_angular, done)
                writer.write(example)

                ind += HORIZON + 1

    bag.close()


def bag_to_tfrecord(bag, args):
    img_t, odo_t, lidar_t, pos, yaw, cmd, collision = read_bag(bag, args.coll_dist)
    if args.plot:
        plot_info(pos, yaw, cmd, bag)
    t = sample_times(img_t, odo_t, lidar_t, args.gap_time)
    pos, yaw, cmd, collision = sample_msgs(t, odo_t, lidar_t, pos, yaw, cmd, collision)
    save_tfrecord(bag, t, pos, yaw, cmd, collision)


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
    parser.add_argument('-t', '--gap_time',
                        type=float, default=0.5,
                        help='The gap time when sampling')
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='The gap time when sampling')
    args = parser.parse_args()

    path = Path(args.path)

    for dirpath, dirnames, filenames in walk(path):
        dirpath = Path(dirpath)
        for filename in filenames:
            if filename.endswith('.bag'):
                logger.info(filename)
                bag_to_tfrecord(dirpath / filename, args)
