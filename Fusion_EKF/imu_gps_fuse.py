
import numpy as np
import pandas as pd
from scipy.linalg import block_diag 
import matplotlib.pyplot as plt

from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise


# read data from CSV
imu_data = pd.read_csv('/media/dadi_vardhan/01D8FC839716D180/Thesis_data/5_row_aruco/Tests/capture_20230509_120504/imu.csv')
gps_data = pd.read_csv('/media/dadi_vardhan/01D8FC839716D180/Thesis_data/5_row_aruco/Tests/capture_20230509_120504/location.csv')  

imu_dt = imu_data['timestamp_s'].diff().mean() # 125 Hz
gps_dt = gps_data[' Time'].diff().mean()/1e9 # 1 Hz

# Fusing GPS and IMU. # 4 states, 2 measurements
# 4 states: x, vx, y, vy (lat, vel_x, long, vel_y)
# 4 measurements: lat, accel_x, long, accel_y 
ekf = ExtendedKalmanFilter(dim_x=4, dim_z=4)

initial_pos_var = (gps_data[" Accuracy"][0]/2)**2
initial_vel_var = (gps_data[" Speed Accuracy"][0]/2)**2

# initial state estimate (x, y, vx, vy)
x = np.array([gps_data[' Latitude'][0], initial_vel_var, 
              gps_data[' Longitude'][0],initial_vel_var])

# initial state covariance (initial uncertainty) (4x4 matrix)
ekf.P = np.diag([initial_pos_var, initial_pos_var, initial_vel_var, initial_vel_var])

# state transition matrix
#dt = 1/125 # imu data is sampled at 125 Hz, can also be updated in real time loop
dt = imu_dt
# taking IMU dt instead of GPS dt because IMU data is sampled at a higher rate
ekf.F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])

# process noise covariance matrix (uncertainty in the model)
std_pos = 0.1 # parameter to tune, starting with 0.1 (0-1)
std_vel = 0.1 # parameter to tune, starting with 0.1 (0-1)

Q_pos = Q_discrete_white_noise(dim=2, dt=gps_dt, var=std_pos**2) # dt =1 fpor gps
Q_accel = Q_discrete_white_noise(dim=2, dt=imu_dt, var=std_vel**2) # dt = 1/125 for imu

ekf.Q = block_diag(Q_pos, Q_accel)

# measurement noise covariance matrix (uncertainty in the measurement)
gps_noise = gps_data[" Accuracy"].mean()
imu_noise = 0.011906238238392809 # for S22, (calibrated-accel_x)

ekf.R = np.diag([gps_noise**2, imu_noise**2, gps_noise**2, imu_noise**2])


# measurement fucntion (how to convert state to measurement)
#Directly observe position and acceleration
ekf.H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

def h(x):
    return np.array([x[0], 0, x[2], 0])

def HJacobian_at(x):
    return np.array([[1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0]])

def imu_update(ekf: ExtendedKalmanFilter, z: np.array):
    ekf.update(z, HJacobian_at, h)
    ekf.predict()
    return ekf

def gps_update(ekf: ExtendedKalmanFilter, z: np.array):
    ekf.update(z, HJacobian_at, h)
    ekf.predict()
    return ekf

pose_x = []
pose_y = []

imu_ix = 0
for i in range(len(gps_data)):
    ekf.predict()

    measurement_x = gps_data[' Latitude'][i]
    measurement_y = gps_data[' Longitude'][i]
    measurement_ax = imu_data['accel_x'][imu_ix:imu_ix+125].mean()
    measurement_ay = imu_data['accel_y'][imu_ix:imu_ix+125].mean()
    imu_ix += 125

    z = np.array([measurement_x, measurement_ax, measurement_y, measurement_ay])

    ekf.update(z, HJacobian_at, h)

    pose_x.append(ekf.x[0])
    pose_y.append(ekf.x[2])

    print(i)
    if i == 100:
        break

plt.figure()
plt.plot(gps_data[' Latitude'], gps_data[' Longitude'], 'r')
plt.plot(pose_x, pose_y, 'g')
plt.legend(['GPS', 'EKF'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('GPS vs EKF (IMU+GPS)')
plt.grid()
plt.savefig('ekf_gps_imu.png')
