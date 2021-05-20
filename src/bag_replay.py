import rospy
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sensor_msgs.msg import Imu, LaserScan, Image


class Replay():
    def __init__(self):
        self.topics = ['/camera/color/image_raw',
                       '/imu_3dm/imu',
                       '/scan']
        self.msgs = [Image,
                     Imu,
                     LaserScan]

        self.d = {}
        self.topic_to_func = {self.topics[0]: self.__process_image,
                              self.topics[1]: lambda m: np.array([m.header.stamp.to_sec(), m.angular_velocity.x, m.angular_velocity.y]),
                              self.topics[2]: lambda m: np.array(m.ranges)}

        rospy.init_node('test', anonymous=True)
        for topic, msg in zip(self.topics, self.msgs):
            rospy.Subscriber(topic, msg, callback=self.update_msg, callback_args=(topic,), queue_size=1)

    def update_msg(self, msg, args):
        topic = args[0]
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
t_collisons = [ax.text(init_pos + i*step, IMAGE_HEIGHT // 2, "", fontsize=16, color='red') for i in range(n_collision)]
last_coll_t = 0

while not rospy.is_shutdown():
    if len(r.d.keys()) == 3:
        l.set_data(r.d[r.topics[0]])

        imu = r.d[r.topics[1]]
        mag = np.linalg.norm(imu[1:])
        if mag > 0.25:
            t_imu.set_text("bumpy")
            if mag > 0.25:
                last_coll_t = imu[0]
        else:
            t_imu.set_text("")

        scan = r.d[r.topics[2]][144:-144]  # 0 to 180 degree
        # scan = r.d[r.topics[2]][180:-180]  # 0 to 180 degree
        if len(r.d[r.topics[2]]) != 1009:
            logger.error(f'Scan length is {len(r.d[r.topics[2]])}')
        scan = scan[:721 // n_collision * n_collision].reshape(3, -1)
        collision = np.min(scan, axis=1) < 0.7
        for i, mask in enumerate(collision[::-1]):
            if mask:
                t_collisons[i].set_text("Collision")
            else:
                t_collisons[i].set_text("")

        plt.pause(0.01)
