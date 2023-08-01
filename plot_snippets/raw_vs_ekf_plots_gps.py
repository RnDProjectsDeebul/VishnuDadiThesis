import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Read data from CSV
df = pd.read_csv('/home/escarda-lab/Vishnu_thesis/camera_mounted/night/capture_20230522_233429/location.csv')  

# Extract Latitude and Longitude columns
latitude_raw = df[' Latitude'].values
longitude_raw = df[' Longitude'].values

# Computing dt from time stamps
measurment_accuracy = df[' Accuracy'].mean() # upto how many meters

dt = df[' Time'].diff().mean()/1e9

# Set up your initial state estimate and covariance:
x = np.array([latitude_raw[0], longitude_raw[0]])  # initial state (latitude, longitude)
P = np.eye(2)  # initial state covariance

# Define the process and measurement noises:
Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)  # process noise covariance
R = np.eye(2) * measurment_accuracy**2  # measurement noise covariance

# Create your EKF filter:
ekf = ExtendedKalmanFilter(dim_x=2, dim_z=2)
ekf.x = x
ekf.P = P
ekf.Q = Q
ekf.R = R

# Define your system.
def h(x):
    return x  # we are directly measuring the state

def HJacobian_at(x):
    return np.eye(2)  # Jacobian of the measurement function

# Update the filter with measurements and store the estimates for later plotting
latitude_ekf = []
longitude_ekf = []
for lat, lon in zip(latitude_raw, longitude_raw):
    ekf.predict()
    ekf.update(np.array([lat, lon]), HJacobian=HJacobian_at, Hx=h)
    latitude_ekf.append(ekf.x[0])
    longitude_ekf.append(ekf.x[1])

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    print(len(im))
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(111)

# # Plot raw and EKF trajectory
# ax.plot(longitude_raw, latitude_raw, label='Raw GPS Trajectory', color='b', alpha=0.5)
# ax.plot(longitude_ekf, latitude_ekf, label='EKF GPS Trajectory', color='r')

# # Decorate the plot
# ax.set_title('GPS 2D Position: Raw vs. EKF')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.legend(loc='upper left')
# ax.grid()
# ax.set_aspect('auto')
# #plt.gca().set_aspect("equal", adjustable = "datalim")
# #forceAspect(ax,aspect=1)

# # Save the plot
# fig.savefig('ekf_vs_raw_GPS_XY_trajectory.png', dpi=300)

# # Show the plot
# plt.show()

time = df[' Time'].values/1e9

# Create the plot
fig, axs = plt.subplots(2, figsize=(10, 12))

# Plot raw and EKF X position
axs[0].plot(time, latitude_raw, label='Raw: Latitude', color='b', alpha=0.5)
axs[0].plot(time, latitude_ekf, label='EKF: Latitude', color='r')

# Decorate the plot
axs[0].set_title('Latitude: Raw vs. EKF')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Latitude (m)')
axs[0].legend(loc='lower left')
axs[0].grid(True)
axs[0].set_aspect("auto", adjustable='datalim')

# Plot raw and EKF Y position
axs[1].plot(time, longitude_raw, label='Raw: Altitude', color='b', alpha=0.5)
axs[1].plot(time, longitude_ekf, label='EKF: Altitude', color='r')

# Decorate the plot
axs[1].set_title('Altitude: Raw vs. EKF')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Altitude (m)')
axs[1].legend(loc='upper left')
axs[1].grid(True)

# Save the plot
plt.savefig('ekf_vs_raw_2D_position_ind-axes-xz_gps.png', dpi=300)