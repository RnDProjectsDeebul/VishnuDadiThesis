import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.integrate import cumtrapz

# Read data from CSV
df = pd.read_csv('/media/dadi_vardhan/01D8FC839716D180/Thesis_data/camera_mounted/night/capture_20230522_233429/imu.csv')

# Extracting accel_x and accel_y columns
accel_x_raw = df['accel_x'].values
accel_y_raw = df['accel_y'].values

# Calculating velocity by integrating acceleration data
vel_x_raw = cumtrapz(accel_x_raw, dx=df['timestamp_s'].diff().mean(), initial=0)
vel_y_raw = cumtrapz(accel_y_raw, dx=df['timestamp_s'].diff().mean(), initial=0)

# Calculating position by integrating velocity data
pos_x_raw = cumtrapz(vel_x_raw, dx=df['timestamp_s'].diff().mean(), initial=1)  # initial position (1, 1)
pos_y_raw = cumtrapz(vel_y_raw, dx=df['timestamp_s'].diff().mean(), initial=1)

# Computing dt from time stamps
dt = df['timestamp_s'].diff().mean()
measurement_std = 0.011906238238392809

# initial state estimate and covariance:
x = np.array([1, 1, 0, 0])  # initial state (x, y, vx, vy)
P = np.eye(4)  # initial state covariance

Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.1)  # process noise covariance
R = np.eye(2) * measurement_std**2  # measurement noise covariance

# Create your EKF filter:
ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
ekf.x = x
ekf.P = P
ekf.F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # state transition matrix
ekf.Q = Q
ekf.R = R

# Define your system. In reality, you would need to carefully choose these models to match your system.
def h(x):
    return np.array([x[2], x[3]])  # we are measuring velocity

def HJacobian_at(x):
    return np.array([[0, 0, 1, 0], 
                     [0, 0, 0, 1]])  # Jacobian of the measurement function

# Update the filter with measurements and store the estimates for later plotting
pos_x_ekf = []
pos_y_ekf = []
for ax, ay in zip(accel_x_raw, accel_y_raw):
    ekf.predict()
    ekf.update(np.array([ax, ay]), HJacobian=HJacobian_at, Hx=h)
    pos_x_ekf.append(ekf.x[0])
    pos_y_ekf.append(ekf.x[1])

# Create time vector
time = df['timestamp_s']

# # Create the plot
fig, axs = plt.subplots(2, figsize=(10, 12))

# Plot raw and EKF X position
axs[0].plot(time, pos_x_raw, label='Raw X Position', color='b', alpha=0.5)
axs[0].plot(time, pos_x_ekf, label='EKF X Position', color='r')

# Decorate the plot
axs[0].set_title('X Position: Raw vs. EKF')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('X Position (m)')
axs[0].legend(loc='lower left')
axs[0].grid(True)
axs[0].set_aspect("equal", adjustable='datalim')

# Plot raw and EKF Y position
axs[1].plot(time, pos_y_raw, label='Raw Y Position', color='b', alpha=0.5)
axs[1].plot(time, pos_y_ekf, label='EKF Y Position', color='r')

# Decorate the plot
axs[1].set_title('Y Position: Raw vs. EKF')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Y Position (m)')
axs[1].legend(loc='upper left')
axs[1].grid(True)

# Save the plot
plt.savefig('ekf_vs_raw_2D_position_XYtime.png', dpi=300)

# # Create individual axes plot
# #---------------------------------------------------------
# fig, axs = plt.subplots(2, figsize=(10, 12))

# # Plot raw and EKF X position
# axs[0].plot(pos_x_raw, pos_y_raw, label='Raw Trajectory', color='b', alpha=0.5)
# # axs[0].plot(pos_x_ekf, pos_y_ekf, label='EKF Trajectory', color='r')

# # Decorate the plot
# axs[0].set_title('XY Trajectory: Raw')
# axs[0].set_xlabel('X Position (m)')
# axs[0].set_ylabel('Y Position (m)')
# axs[0].legend(loc='upper right')
# axs[0].grid(True)

# # # Plot raw and EKF Y position
# axs[1].plot(pos_x_ekf, pos_y_ekf, label='EKF Trajectory', color='r')
# # axs[1].plot(time, pos_y_ekf, label='EKF Y Position', color='r')

# # Decorate the plot
# axs[1].set_title('XY Trajectory: EKF')
# axs[1].set_xlabel('X Position (m)')
# axs[1].set_ylabel('Y Position (m)')
# axs[1].legend(loc='upper right')
# axs[1].grid(True)

# # Save the plot
# plt.savefig('ekf_vs_raw_2D_position.png', dpi=300)

# Create trajecory plot
#------------------------------------------
#plt.figure(figsize=(10, 6))

#Plot raw and EKF trajectory
# plt.plot(pos_x_raw, pos_y_raw, label='Raw Trajectory', color='b', alpha=0.5)
# plt.plot(pos_x_ekf, pos_y_ekf, label='EKF Trajectory', color='r')

# # Decorate the plot
# plt.title('2D Position: Raw vs. EKF')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')

# # Set the limits for the x and y axes
# #plt.xlim([min(pos_x_raw.min(), np.array(pos_x_ekf).min()), max(pos_x_raw.max(), np.array(pos_x_ekf).max())])
# #plt.ylim([min(pos_y_raw.min(), np.array(pos_y_ekf).min()), max(pos_y_raw.max(), np.array(pos_y_ekf).max())])

# plt.legend(loc='upper right')
# plt.grid(True)
# plt.gca().set_aspect('auto')

# # Save the plot
# plt.savefig('etest.png', dpi=300)




#'/home/escarda-lab/Vishnu_thesis/5_row_aruco/Tests/capture_20230509_120504/imu.csv'