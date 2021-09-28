import rospy
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sensor_msgs.msg import Imu, LaserScan, Image


IMAGE_TOPIC = '/camera/color/image_raw'
IMU_TOPIC = '/imu_3dm/imu'
LIDAR_TOPIC = '/scan'


class Replay():
    def __init__(self):
        self.topics = [IMAGE_TOPIC, IMU_TOPIC, LIDAR_TOPIC]
        self.msgs = [Image,
                     Imu,
                     LaserScan]

        window_size = 30
        self.d = {}
        self.d[IMU_TOPIC] = [np.zeros(3) for _ in range(window_size)]
        self.topic_to_func = {IMAGE_TOPIC: self.__process_image,
                              IMU_TOPIC: lambda m: np.array([m.header.stamp.to_sec(), m.angular_velocity.x, m.angular_velocity.y]),
                              LIDAR_TOPIC: lambda m: (m.angle_min, m.angle_max, m.angle_increment, np.array(m.ranges))}

        rospy.init_node('test', anonymous=True)
        for topic, msg in zip(self.topics, self.msgs):
            rospy.Subscriber(topic, msg, callback=self.update_msg, callback_args=(topic,), queue_size=1)

    def update_msg(self, msg, args):
        topic = args[0]
        if topic == IMU_TOPIC:
            msg = self.topic_to_func[topic](msg)
            self.d[topic].pop(0)
            self.d[topic].append(msg)
        else:
            self.d[topic] = self.topic_to_func[topic](msg)

    def __process_image(self, msg):
        msg.__class__ = Image
        return ros_numpy.numpify(msg)


r = Replay()
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
fig, ax = plt.subplots()
array = np.random.randint(0, 100, size=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
l = ax.imshow(array)
t_imu = ax.text(IMAGE_WIDTH, IMAGE_HEIGHT, "", fontsize=16, color='green', bbox=dict(fill=False, edgecolor='red', linewidth=2))
n_collision = 3
step = IMAGE_WIDTH // n_collision
init_pos = step // 4
t_collisons = [ax.text(init_pos + i*step,  IMAGE_HEIGHT // 2, "", fontsize=16, color='red') for i in range(n_collision)]
last_coll_t = 0

while not rospy.is_shutdown():
    if len(r.d.keys()) == 2:
        l.set_data(r.d[r.topics[0]])

        imu_all = np.array(r.d[IMU_TOPIC])
        imu = imu_all[-1]
        # imu = np.abs(imu_all[-1] - np.mean(imu_all, axis=0))
        mag = np.linalg.norm(imu[1:])

        # mag_all = np.linalg.norm(imu_all[:, 1:], axis=1)
        # mag = np.abs(mag_all[-1] - np.mean(mag_all))
        if mag > 0.2:
            t_imu.set_text("bumpy")
            if mag > 0.25:
                # last_coll_t = imu[0]
                pass
        else:
            t_imu.set_text("")

        # ranges = r.d[r.topics[2]][-1]
        # if len(ranges) != 1009:
        #     logger.error(f'Scan length is {len(ranges)}')
        #     logger.error(r.d[r.topics[2]][:-1])
        # # scan = ranges[144:-144]  # 0 to 180 degree
        # scan = ranges[304:-304]  # 40 to 140 degree
        # # scan = ranges[180:-180]  # 0 to 180 degree
        # vis_len = len(scan) // n_collision * n_collision
        # scan = scan[:vis_len].reshape(3, -1)
        # collision = (np.min(scan, axis=1) < 3.5) & (np.min(scan, axis=1) > 1)
        # for i, mask in enumerate(collision[::-1]):
        #     if mask:
        #         t_collisons[i].set_text("Collision")
        #     else:
        #         t_collisons[i].set_text("")

        plt.pause(0.01)
