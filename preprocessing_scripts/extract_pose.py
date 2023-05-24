
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class ExtractPoseData:

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

        pose_df = pd.DataFrame(
            columns=['timestamp_ns', 'timestamp_s', 
                        'x', 'y', 'z', 'w']
                    )
        
        for _, row in tqdm(self.__sensors_df.iterrows(), total=len(self.__sensors_df),
                            desc='Creating rotation dataframe'):
            timestamp_ns = row[' Timestamp']
            timestamp_s = row[' Time']

            if row['Sensor'] == 11:
                x = row[' Value0']
                y = row[' Value1']
                z = row[' Value2']
                w = row[' Value3']
                # use concat instead of append

                pose_df = pd.concat([pose_df, pd.DataFrame(
                    {'timestamp_ns': timestamp_ns, 'timestamp_s': timestamp_s, 
                     'x': x, 'y': y, 'z': z, 'w': w}, index=[0])], ignore_index=True)

                # pose_df = pose_df.append(
                #     {'timestamp_ns': timestamp_ns, 'timestamp_s': timestamp_s, 
                #      'x': x, 'y': y, 'z': z, 'w': w}, 
                #     ignore_index=True)
        
        return pose_df
    
    def save(self, pose_df: pd.DataFrame) -> None:
        pose_df.to_csv(self.data_dir/'rotation.csv', index=False)
        print(f'Pose data saved at {self.data_dir}/rotation.csv')


if __name__ == '__main__':
    data_dir = Path('/home/dadi_vardhan/Downloads/escarda/5_row_aruco/capture_20230509_123316')
    pose_extractor = ExtractPoseData(data_dir)
    pose_df = pose_extractor.extract()
    pose_extractor.save(pose_df)
