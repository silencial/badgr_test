import rosbag
import argparse
import os


def get_next(bag, topic=None):
    try:
        a = bag.next()
        if topic is not None:
            while a[0] != topic:
                a = bag.next()
        return a
    except:
        return None


def get_time(msg):
    return msg[1].header.stamp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, help='output file name')
    parser.add_argument('bag1', type=str, help='carinfo bagfile')
    parser.add_argument('bag2', type=str, help='realsense bagfile')

    args = parser.parse_args()

    if args.o is None:
        args.o = args.bag1.split('.')[0] + '_merge.bag'
    outbag = rosbag.Bag(args.o, 'w')

    topic = '/device_0/sensor_1/Color_0/image/data'

    bag1 = rosbag.Bag(args.bag1).__iter__()
    bag2 = rosbag.Bag(args.bag2).__iter__()

    value1 = get_next(bag1)
    value2 = get_next(bag2, topic)

    while value1 is not None or value2 is not None:
        if value1 is None:
            outbag.write(value2[0], value2[1], get_time(value2))
            value2 = get_next(bag2, topic)
        elif value2 is None:
            outbag.write(value1[0], value1[1], value1[2])
            value1 = get_next(bag1)
        elif value1[2] < get_time(value2):
            outbag.write(value1[0], value1[1], value1[2])
            value1 = get_next(bag1)
        else:
            outbag.write(value2[0], value2[1], get_time(value2))
            value2 = get_next(bag2, topic)
    outbag.close()
