# %%
from os import walk
import time

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf

# %%
tf.compat.v1.enable_eager_execution()
OWNENV = False  # whether use own data

# %%
if OWNENV:
    files = ['../data/circles/21-03-17/16-59-30.tfrecord']
    raw_dataset = tf.data.TFRecordDataset(files)
else:
    COLLISION = True
    files = []
    path = '../../data/tfrecords_collision' if COLLISION else '../../data/tfrecords_bumpy'
    for dirpath, dirnames, filenames in walk(path):
        for filename in filenames:
            if filename.endswith('tfrecord'):
                files.append(dirpath+'/'+filename)
    files.sort()
    # files.append('../../data/tfrecords_collision/08-02-2019_horizon_8/0000.tfrecord')
    print(f'File numbers: {len(files)}')
    raw_dataset = tf.data.TFRecordDataset(files)

# %%
if OWNENV:
    names_shapes_limits_dtypes = (
        ('images/rgb_left', (96, 128, 3), (0, 255), np.uint8),
        ('collision/close', (1, ), (0, 1), np.bool),
        ('jackal/position', (3, ), (-0.5, 0.5), np.float32),
        ('jackal/yaw', (1, ), (-np.pi, np.pi), np.float32),
        ('commands/angular_velocity', (1, ), (-1.0, 1.0), np.float32),
        ('commands/linear_velocity', (1, ), (0.75, 1.25), np.float32)
    )
    collision_names = [
        'images/rgb_left',
        'jackal/position',
        'jackal/yaw',
        'collision/close',
    ]
    action_names = ['commands/angular_velocity', 'commands/linear_velocity']
    names = collision_names
else:
    names_shapes_limits_dtypes = (
        ('images/rgb_left', (96, 128, 3), (0, 255), np.uint8),
        ('images/rgb_right', (96, 128, 3), (0, 255), np.uint8),
        ('images/thermal', (32, 32), (-1, 1), np.float32),
        ('lidar', (360, ), (0., 12.), np.float32),
        ('collision/close', (1, ), (0, 1), np.bool),
        ('collision/flipped', (1, ), (0, 1), np.bool),
        ('collision/stuck', (1, ), (0, 1), np.bool),
        ('collision/any', (1, ), (0, 1), np.bool),
        ('gps/is_fixed', (1, ), (0, 1), np.float32),
        ('gps/latlong', (2, ), (0, 1), np.float32),
        ('gps/utm', (2, ), (0, 1), np.float32),
        ('imu/angular_velocity', (3, ), (-1.0 * np.pi, 1.0 * np.pi), np.float32),
        ('imu/compass_bearing', (1, ), (-np.pi, np.pi), np.float32),
        ('imu/linear_acceleration', (3, ), ((-1., -1., 9.81 - 1.), (1., 1., 9.81 + 1.)), np.float32),
        ('jackal/angular_velocity', (1, ), (-1.0 * np.pi, 1.0 * np.pi), np.float32),
        ('jackal/linear_velocity', (1, ), (-1., 1.), np.float32),
        ('jackal/imu/angular_velocity', (3, ), (-1.0 * np.pi, 1.0 * np.pi), np.float32),
        ('jackal/imu/linear_acceleration', (3, ), ((-1., -1., 9.81 - 1.), (1., 1., 9.81 + 1.)), np.float32),
        ('jackal/position', (3, ), (-0.5, 0.5), np.float32),
        ('jackal/yaw', (1, ), (-np.pi, np.pi), np.float32),
        ('android/illuminance', (1, ), (0., 200.), np.float32),
        ('bumpy', (1, ), (0, 1), np.bool),
        ('commands/angular_velocity', (1, ), (-1.0, 1.0), np.float32),
        ('commands/linear_velocity', (1, ), (0.75, 1.25), np.float32)
    )
    collision_names = [
        'images/rgb_left',
        'jackal/position',
        'jackal/yaw',
        'jackal/angular_velocity',
        'jackal/linear_velocity',
        'jackal/imu/angular_velocity',
        'jackal/imu/linear_acceleration',
        'imu/angular_velocity',
        'imu/linear_acceleration',
        'imu/compass_bearing',
        'gps/latlong',
        'collision/close',
        'collision/stuck',
    ]
    bumpy_names = [
        'images/rgb_left',
        'jackal/imu/angular_velocity',
        'jackal/imu/linear_acceleration',
        'imu/angular_velocity',
        'imu/linear_acceleration',
        'bumpy',
    ]
    action_names = ['commands/angular_velocity', 'commands/linear_velocity']
    names = collision_names if COLLISION else bumpy_names

names_to_dtypes = {}
names_to_shapes = {}
for name, shape, _, dtype in names_shapes_limits_dtypes:
    names_to_dtypes[name] = dtype
    names_to_shapes[name] = shape


def get_names_dtypes_shapes(observation_names, action_names):
    env = dict()
    names = ['inputs/' + name for name in observation_names + action_names] + \
            ['outputs/' + name for name in observation_names if 'rgb' not in name]
    dtypes = [tf.dtypes.as_dtype(names_to_dtypes[name.replace('inputs/', '').replace('outputs/', '')]) for name in names]
    dtypes = [dtype if dtype != tf.bool else tf.uint8 for dtype in dtypes]
    shapes = []
    for name in names:
        name_suffix = name.replace('inputs/', '').replace('outputs/', '')
        shape = list(names_to_shapes[name_suffix])
        if name.startswith('outputs/') or name_suffix in action_names:
            shape = [8] + shape
        shapes.append(shape)

    names.append('outputs/done')
    dtypes.append(tf.uint8)
    shapes.append((8, ))

    env['names'] = names
    env['dtypes'] = dtypes
    env['shapes'] = shapes

    return env


env = get_names_dtypes_shapes(names, action_names)


# %%
def parse_fn(data):
    parsed = tf.parse_single_example(data, {name: tf.FixedLenFeature([], tf.string) for name in env['names']})
    decode_parsed = {name: tf.decode_raw(parsed[name], dtype) for name, dtype in zip(env['names'], env['dtypes'])}
    reshape_decode_parsed = dict()
    for name, shape in zip(env['names'], env['shapes']):
        tensor = decode_parsed[name]
        tensor.set_shape([np.prod(shape)])
        tensor = tf.reshape(tensor, shape)
        reshape_decode_parsed[name] = tensor
    return reshape_decode_parsed


# %%
dataset = raw_dataset.map(parse_fn, num_parallel_calls=6)
# dataset = dataset.filter(lambda x: tf.reduce_any(
#     tf.cast(x['outputs/collision/close'], tf.bool)
# ))
# dataset = dataset.batch(2)
# iterator = dataset.make_one_shot_iterator()

# %%
start_time = time.time()
count = 0
count_close = 0
count_img = 0
for data in dataset:
    count += 1
    if data['outputs/collision/close'].numpy().sum() > 0.5:
        count_close += 1
        continue
    # if data['outputs/done'].numpy().sum() < 0.5:
    #     continue
    # if data['outputs/bumpy'].numpy().sum() < 0.5:
    #     continue
    if count_img > 100:
        break

    # print('-----collsion-----')
    # print(data['inputs/collision/close'].numpy())
    # print(data['outputs/collision/close'].numpy().flatten())
    # print(data['outputs/collision/stuck'].numpy().flatten())

    # print('-----position & yaw-----')
    # print(data['inputs/jackal/position'].numpy())
    # print(data['outputs/jackal/position'].numpy())
    # print(data['inputs/jackal/yaw'].numpy())
    # print(data['outputs/jackal/yaw'].numpy().flatten())

    # print('-----velocity-----')
    # print(data['inputs/jackal/linear_velocity'])
    # print(data['outputs/jackal/linear_velocity'])
    # print(data['inputs/jackal/angular_velocity'])
    # print(data['outputs/jackal/angular_velocity'])

    # print('-----cmd-----')
    # print(data['inputs/commands/linear_velocity'].numpy().flatten())
    # print(data['inputs/commands/angular_velocity'].numpy().flatten())

    # print('-----done-----')
    # print(data['outputs/done'].numpy())

    # jak_lin_acc = data['outputs/jackal/imu/linear_acceleration'].numpy()
    # jak_ang_vel = data['outputs/jackal/imu/angular_velocity'].numpy()
    # lin_acc = data['outputs/imu/linear_acceleration'].numpy()
    # ang_vel = data['outputs/imu/angular_velocity'].numpy()

    # bumpy = data['outputs/bumpy'].numpy()

    # print(np.stack((bumpy.flatten(),
    #                 np.linalg.norm(jak_ang_vel, axis=1),
    #                 np.linalg.norm(ang_vel, axis=1)), axis=-1))

    if count_img < 100:
        img = data['inputs/images/rgb_left'].numpy()
        img = Image.fromarray(img)
        img.save(f'../result/img/{count}.png')
        # plt.imshow(img, interpolation='nearest')
        # plt.show()
    count_img += 1

print(f'Total data number: {count}')
print(f'Close number: {count_close}')
print(f'Time elapsed: {time.time() - start_time}')

# %%
for data in dataset.take(500):
    if data['outputs/collision/close'].numpy().sum() > 0.5:
        continue
    if data['outputs/done'].numpy().sum() < 0.5:
        continue
    print('-----collsion-----')
    print(data['inputs/collision/close'].numpy())
    print(data['outputs/collision/close'].numpy().flatten())

    # print('-----position & yaw-----')
    # print(data['inputs/jackal/position'].numpy())
    # print(data['outputs/jackal/position'].numpy())
    # print(data['inputs/jackal/yaw'].numpy())
    # print(data['outputs/jackal/yaw'].numpy().flatten())

    # # print('-----cmd-----')
    # # print(data['inputs/commands/linear_velocity'])
    # # print(data['inputs/commands/angular_velocity'])

    # print('-----cmd-----')
    # print(data['inputs/commands/linear_velocity'].numpy().flatten())
    # print(data['inputs/commands/angular_velocity'].numpy().flatten())

    print('-----done-----')
    print(data['outputs/done'].numpy())

    img = data['inputs/images/rgb_left'].numpy()
    plt.imshow(img, interpolation='nearest')
    plt.show()


# %%
