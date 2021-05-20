import rospy
import numpy as np
import time

from sensor_msgs.msg import MagneticField
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


d = {}
maximum = minimum = 0
def update_msg(msg, args):
    topic = args[0]
    func = args[1]
    d[topic] = func(msg)


topics_msgs_funcs = (
        ('/imu_3dm/magnetic_field', MagneticField, lambda msg: np.array([msg.magnetic_field.x,
                                     msg.magnetic_field.y,
                                     msg.magnetic_field.z])),
        ('/odometry/filtered', Odometry, lambda msg: np.arctan2(2*msg.pose.pose.orientation.w *
                                                msg.pose.pose.orientation.z,
                                                1 - 2 * msg.pose.pose.orientation.z *
                                                msg.pose.pose.orientation.z)),
        ('/imu_um7/compass_bearing', Float32, lambda msg: msg.data)
)

rospy.init_node('compare_yaw_magnet', anonymous=True)
for topic, msg, func in topics_msgs_funcs:
    rospy.Subscriber(topic, msg, callback=update_msg, callback_args=(topic, func))


time.sleep(2)
while not rospy.is_shutdown():
    print(d)
    time.sleep(1)
    # maximum = max(d['/imu_um7/compass_bearing'], maximum)
    # minimum = min(d['/imu_um7/compass_bearing'], minimum)

# print(maximum, minimum)
