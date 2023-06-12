
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_2d(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
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

gps_data = pd.read_csv('/home/dadi_vardhan/Downloads/escarda/5_row_aruco/Tests/capture_20230509_120504/location.csv', delimiter=',')

lat = gps_data[' Latitude'].values
lon = gps_data[' Longitude'].values
alt = gps_data[' Altitude'].values

gps_data = np.array([lat, lon, alt]).T

plot_2d(gps_data)


