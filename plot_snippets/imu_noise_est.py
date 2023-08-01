import pandas as pd
import numpy as np

# Assuming 'data.csv' is the CSV file with your IMU data
df = pd.read_csv('/home/escarda-lab/Vishnu_thesis/QuickShare_2308011349/imu.csv')

# Let's take the acceleration along x-axis as an example
accel_x = df['accel_z']

# Compute mean
mean_x = np.mean(accel_x)

# Compute standard deviation
std_x = np.std(accel_x)

print("Mean of Acceleration (X-axis): ", mean_x)
print("Standard Deviation of Acceleration (X-axis): ", std_x)

# S22_std_x = 0.011906238238392809
# S22_std_y = 0.012634821254569615
# S22_std_z = 0.02345790105991327