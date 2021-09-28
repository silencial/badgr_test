# %%
import rosbag
from sensor_msgs.msg import Image
import numpy as np
import ros_numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as PltLineCollection
import cv2


# %%
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


# original camera parameters
fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 240.000000
dim = (640, 480)
K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
D = np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
balance = 0.5

OWN_ENV = True
if OWN_ENV:
    # own camera paramters
    fx, fy, cx, cy = 384.944396973, 384.575073242, 309.668579102, 243.29864502
    dim = (640, 480)
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    D = np.array([[-0.0548635944724, 0.0604563839734, -0.00111321196891, -4.80580529256e-05, -0.0191334541887]]).T


def image_intrinsics():
    return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)


def imrectify(img):
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    dim = img.shape[:2][::-1]
    new_K = image_intrinsics()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


if OWN_ENV:
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
    if OWN_ENV:
        camera_matrix = K
        dist_coeffs = D
    else:
        camera_matrix = image_intrinsics()
        k1, k2, p1, p2 = D.ravel()
        k3 = k4 = k5 = k6 = 0.
        dist_coeffs = (k1, k2, p1, p2, k3, k4, k5, k6)

    # x = y
    # y = -z
    # z = x
    xyz[..., 0] += 0.6  # NOTE(greg): shift to be in front of image plane
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs)
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def commands_to_positions(linvel, angvel):
    dt = 0.25
    N = len(linvel)
    all_angles = [np.zeros(N)]
    all_positions = [np.zeros((N, 2))]
    for linvel_i, angvel_i in zip(linvel.T, angvel.T):
        angle_i = all_angles[-1] + dt * angvel_i
        position_i = all_positions[-1] + \
                     dt * linvel_i[..., np.newaxis] * np.stack([np.cos(angle_i), np.sin(angle_i)], axis=1)

        all_angles.append(angle_i)
        all_positions.append(position_i)

    all_positions = np.stack(all_positions, axis=1)
    return all_positions


# %%
bag_name = '../../badgr/data/rosbags/collision.bag'
cam_topic = '/cam_left/image_raw'
if OWN_ENV:
    bag_name = '../data/circles/21-03-17/16-59-30.bag'
    cam_topic = '/camera/color/image_raw'

with rosbag.Bag(bag_name) as bag:
    for _, msg, t in bag.read_messages(cam_topic):
        msg.__class__ = Image
        img = ros_numpy.numpify(msg)
        plt.imshow(img, interpolation='nearest')
        plt.show()

        im_1 = imrectify(img)
        plt.imshow(im_1, interpolation='nearest')
        plt.show()

        # im_2 = cv2.fisheye.undistortImage(img, K, D, Knew=image_intrinsics())
        # plt.imshow(im_2, interpolation='nearest')
        # plt.show()
        # print(img.size)
        break

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
fpv_imshow = Imshow(ax)
fpv_batchline = BatchLineCollection(ax)

fpv_imshow.draw(np.flipud(im_1), origin='lower')

im_lims = im_1.shape

linvel = np.array([[1, 1, 1, 1, 1]])
angvel = np.array([[1, 1, 1, 1, 1]])
xy = commands_to_positions(linvel, angvel)

# xy = np.array([[[0, 0], [1, 0]],
#                [[0, 0], [5, -1]],
#                [[0, 0], [5.5, 3]]])
pixels = project_points(xy)
x_list, y_list = [], []
for pix in pixels:
    pix_lims = (480., 640.)

    assert pix_lims[1] / pix_lims[0] == im_lims[1] / float(im_lims[0])
    resize = im_lims[0] / pix_lims[0]

    pix = resize * pix
    x_list.append(im_lims[1] - pix[:, 0])
    y_list.append(im_lims[0] - pix[:, 1])
# x_list = [[300, 300], [100, 500]]
# y_list = [[100, 300], [200, 200]]
fpv_batchline.draw(x_list, y_list, linewidth=3.)


# %%
