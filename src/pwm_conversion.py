# %%
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from pynverse import inversefunc

# %%
bag = '../data/circles/21-03-17/16-59-30.bag'
odo = pd.read_csv(bag+'_odo.csv',
                  usecols=[
                      'field.header.stamp',
                      'field.pose.pose.position.x', 'field.pose.pose.position.y',
                      'field.pose.pose.orientation.z', 'field.pose.pose.orientation.w',
                      'field.twist.twist.linear.x',
                      'field.twist.twist.angular.z'
                  ])
pwm_cmd = pd.read_csv(bag+'_pwm_cmds.csv', usecols=['field.t_epoch', 'field.pwm_v', 'field.pwm_s'])
uav_cmd = pd.read_csv(bag+'_uav_cmds.csv', usecols=['%time', 'field.data0', 'field.data1'])
velstr_cmd = pd.read_csv(bag+'_velstr_cmds.csv', usecols=['field.stamp', 'field.vel', 'field.str'])

# Rename
odo = odo.rename(
    columns={
        'field.header.stamp': 't',

        'field.pose.pose.position.x': 'x',
        'field.pose.pose.position.y': 'y',

        'field.pose.pose.orientation.z': 'rot_z',
        'field.pose.pose.orientation.w': 'rot_w',

        'field.twist.twist.linear.x': 'lin_vel',

        'field.twist.twist.angular.z': 'ang_vel'

    })
pwm_cmd = pwm_cmd.rename(columns={'field.t_epoch': 't', 'field.pwm_v': 'v', 'field.pwm_s': 's'})
uav_cmd = uav_cmd.rename(columns={'%time': 't', 'field.data0': 'v', 'field.data1': 's'})
velstr_cmd = velstr_cmd.rename(columns={'field.stamp': 't', 'field.vel': 'v', 'field.str': 's'})

# Compute yaw from odometry quaternion
odo['yaw'] = np.arctan2(2 * odo['rot_z'] * odo['rot_w'], 1 - 2 * odo['rot_z'] * odo['rot_z'])
odo.drop(columns=['rot_z', 'rot_w'])

# Change time to seconds, only meaningful when taking difference
odo['t'] /= 1e9
pwm_cmd['t'] /= 1e9
uav_cmd['t'] /= 1e9
velstr_cmd['t'] /= 1e9


# %%
def estimate(x, y, yaw, t, lin_vel, ang_vel):
    x_ = [x[0]]
    y_ = [y[0]]
    yaw_ = [yaw[0]]
    for i in range(len(x) - 1):
        yaw_new = yaw_[-1] + ang_vel[i] * (t[i+1] - t[i])
        yaw_.append(yaw_new)

        r = lin_vel[i] / ang_vel[i]
        x_new = x_[-1] + r*(np.sin(yaw_[-1]) - np.sin(yaw_[-2]))
        y_new = y_[-1] + r*(-np.cos(-yaw_[-1]) + np.cos(yaw_[-2]))
        x_.append(x_new)
        y_.append(y_new)

    yaw_ = (np.array(yaw_) + np.pi) % (2 * np.pi) - np.pi
    x_ = np.array(x_)
    y_ = np.array(y_)
    return x_, y_, yaw_


def ax_plot(ax, title, x, y, y_):
    ax.plot(x, y)
    ax.plot(x, y_)
    ax.legend(['Real', 'Estimate'])
    ax.set_title(title)


def plot(t0, t, x, y, yaw, x_, y_, yaw_):
    print(t[-1] - t[0])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax_plot(ax1, "x", t - t0, x, x_)
    ax_plot(ax2, "y", t - t0, y, y_)
    ax_plot(ax3, "yaw", t - t0, yaw, yaw_)
    # plt.savefig('pos_predition_by_twist.png')
    plt.show()

    print(np.abs(x - x_).mean())
    print(np.abs(y - y_).mean())
    print(np.abs(yaw - yaw_).mean() * 180 / np.pi)


# %%
t = odo['t'].to_numpy()
x = odo['x'].to_numpy()
y = odo['y'].to_numpy()
yaw = odo['yaw'].to_numpy()
lin_vel = odo['lin_vel'].to_numpy()
ang_vel = odo['ang_vel'].to_numpy()

region = range(1600, 1700)
x_, y_, yaw_ = estimate(x[region], y[region], yaw[region], t[region], lin_vel[region], ang_vel[region])
plot(t[0], t[region], x[region], y[region], yaw[region], x_, y_, yaw_)

# %%
siToPwmVel = lambda vel_si: 1500 + 31.32516*vel_si  # linear velocity
phiToJrk = lambda phi: 1434.4941*phi + 2275.5982 + 74.6464 * np.tanh(46.4265*phi + 1.2077)
jrkToPwm = lambda jrk: 0.921 * (jrk-1600) + 945
siToPwmAng = lambda si_ang: jrkToPwm(phiToJrk(si_ang))  # steering_angle

pwmVel2SI = lambda vel_pwm: (vel_pwm-1500) / 31.32516
pwmAng2SI = inversefunc(siToPwmAng)

# Interpolate cmds
L = 0.51  # axel length
lin_vel = np.interp(t, uav_cmd['t'], uav_cmd['v'])
lin_vel = pwmVel2SI(lin_vel)
steering = np.interp(t, uav_cmd['t'], uav_cmd['s'])
steering = pwmAng2SI(steering)
ang_vel = lin_vel * np.tan(steering) / L

x_, y_, yaw_ = estimate(x[region], y[region], yaw[region], t[region], lin_vel[region], ang_vel[region])
plot(t[0], t[region], x[region], y[region], yaw[region], x_, y_, yaw_)

# %%
