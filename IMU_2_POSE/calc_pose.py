
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skinematics.sensors.manual import MyOwnSensor
from skinematics.imus import IMU_Base


def plot_2d(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    # plt.plot(x, y, marker='x')
    # plt.title('Pose Trajectory')
    # plt.show()
    # create a subplot with all possible axes permutations and plot the data
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title('x vs y')
    axs[0, 1].plot(x, z)
    axs[0, 1].set_title('x vs z')
    axs[0, 2].plot(y, z)
    axs[0, 2].set_title('y vs z')
    axs[1, 0].plot(y, x)
    axs[1, 0].set_title('y vs x')
    axs[1, 1].plot(z, x)
    axs[1, 1].set_title('z vs x')
    axs[1, 2].plot(z, y)
    axs[1, 2].set_title('z vs y')
    axs[2, 0].plot(x, x)
    axs[2, 0].set_title('x vs x')
    axs[2, 1].plot(y, y)
    axs[2, 1].set_title('y vs y')
    axs[2, 2].plot(z, z)
    axs[2, 2].set_title('z vs z')
    plt.show()

if __name__ == '__main__':
    # Read in the data
    imu_data_path = '/home/dadi_vardhan/Downloads/escarda/5_row_aruco/Tests/capture_20230509_120504/imu.csv'
    df = pd.read_csv(imu_data_path, delimiter=',')
    time_stamp = df['timestamp_s'].values.tolist()
    acc = df[['accel_x', 'accel_y', 'accel_z']].values
    gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    mag = df[['mag_x', 'mag_y', 'mag_z']].values

    R_init = np.eye(3)
    pos_init = np.ones(3)
    q_type = 'kalman' # 'kalman', 'madgwick', 'mahony', 'analytical'

    in_data = {
        'rate': 125,
        'acc': acc,
        'omega': gyro,
        'mag': mag,
        }

    # Create the sensor object
    my_sensor = MyOwnSensor(
        q_type = q_type,
        R_init = R_init,
        calculate_position = True,
        pos_init = pos_init,
        in_data = in_data,
        )

    pos = my_sensor.pos
    print(f'Pose: {pos.shape}')
    print(f'pose@1753: {pos[1753]}')
    print('-'*50)
    quat = my_sensor.quat
    print(f'Quat: {quat.shape}')
    print('quat@1753: ', quat[1753])
    print('-'*50)
    vel = my_sensor.vel
    print(f'Vel: {vel.shape}')
    print('vel@1753: ', vel[1753])
    print('-'*50)

    # plot_2d(pos)
