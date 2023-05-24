""" 
This file contains python class to extract IMU data from the sensors csv file
present in the raw capture data directory (AI capture app output dir).

author: Vishnu Vardhan Dadi
last updated: 24th May 2023
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

class ExtractIMUData:
    
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.__sensors_df = self.__load_sensors_data()

    def __load_sensors_data(self) -> pd.DataFrame:
        """ Load the sensors csv file present in the data directory and
            returns a dataframe.
        """
        sensors_file = [file for file in self.data_dir.iterdir() if file.suffix == '.csv' and 'sensors' in file.name]

        if len(sensors_file) == 0:
            raise FileNotFoundError('No sensors csv file found')
        elif len(sensors_file) > 1:
            raise FileNotFoundError('Multiple sensors csv files found')
        else:
            sensors_file = sensors_file[0]
        
        sensors_df = pd.read_csv(sensors_file)
        return sensors_df
    
    def extract(self) -> pd.DataFrame:
        """ Extracts the IMU data from the sensors csv file and returns a dataframe
            with columns timestamp_ns, timestamp_s, accel_x, accel_y, accel_z, 
            gyro_x, gyro_y, gyro_z
        """
        imu_df = pd.DataFrame(
            columns=['timestamp_ns', 'timestamp_s', 
                     'accel_x', 'accel_y', 'accel_z',
                     'gyro_x', 'gyro_y', 'gyro_z',
                     'mag_x', 'mag_y', 'mag_z']
                    )
        accel_data = []
        gyro_data = []
        mag_data = []
  
        for _, row in tqdm(self.__sensors_df.iterrows(), total=len(self.__sensors_df),
                           desc='Separating IMU data'):
            timestamp_ns = row[' Timestamp']
            timestamp_s = row[' Time']

            if row['Sensor'] == 0:
                accel_x = row[' Value0']
                accel_y = row[' Value1']
                accel_z = row[' Value2']
                accel_data.append([timestamp_ns, timestamp_s, accel_x, accel_y, accel_z])
            elif row['Sensor'] == 1:
                gyro_x = row[' Value0']
                gyro_y = row[' Value1']
                gyro_z = row[' Value2']
                gyro_data.append([timestamp_ns, timestamp_s, gyro_x, gyro_y, gyro_z])
            elif row['Sensor'] == 3:
                mag_x = row[' Value0']
                mag_y = row[' Value1']
                mag_z = row[' Value2']
                mag_data.append([timestamp_ns, timestamp_s, mag_x, mag_y, mag_z])

        for idx, row in tqdm(enumerate(accel_data), total=len(accel_data),
                             desc='Creating IMU dataframe'):
            timestamp_ns = row[0]
            imu_df.loc[idx, 'timestamp_ns'] = timestamp_ns
            imu_df.loc[idx, 'timestamp_s'] = row[1]
            imu_df.loc[idx, 'accel_x'] = row[2]
            imu_df.loc[idx, 'accel_y'] = row[3]
            imu_df.loc[idx, 'accel_z'] = row[4]
            imu_df.loc[idx, 'gyro_x'] = gyro_data[idx][2]
            imu_df.loc[idx, 'gyro_y'] = gyro_data[idx][3]
            imu_df.loc[idx, 'gyro_z'] = gyro_data[idx][4]
            imu_df.loc[idx, 'mag_x'] = mag_data[idx][2]
            imu_df.loc[idx, 'mag_y'] = mag_data[idx][3]
            imu_df.loc[idx, 'mag_z'] = mag_data[idx][4]

        return imu_df
    
    def save(self, imu_df: pd.DataFrame) -> None:
        """ Saves the imu dataframe to a csv file in the data directory

        Parameters
        ----------
        imu_df : pd.DataFrame
            Dataframe with columns timestamp_ns, timestamp_s, accel_x, accel_y, accel_z, 
            gyro_x, gyro_y, gyro_z
        """
        imu_df.to_csv(self.data_dir / 'imu.csv', index=False)
        print(f'IMU data saved to imu.csv at Path: {self.data_dir}/imu.csv')
            

if __name__ == '__main__':
    data_dir = Path('/home/dadi_vardhan/Downloads/escarda/5_row_aruco/capture_20230509_123316')
    imu_data = ExtractIMUData(data_dir)
    imu_df = imu_data.extract()
    imu_data.save(imu_df)