
import os
from pathlib import Path
import yaml

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


class VisualOdometry:
    def __init__(self, video_path: Path) -> None:
        self.K , self.P , self.R, self.t = self.__load_camera_parameters()
        self.video_path = video_path
        self.estimated_trajectory = []
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def __load_camera_parameters(self)-> np.ndarray:
        """ Loads the camera parameters from the config file and returns the
          intrinsic matrix, projection matrix, rotation matrix and translation vector.


        Returns
        -------
        np.ndarray
            The intrinsic matrix, projection matrix, rotation matrix and translation vector.
        """
        try:
            with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
                params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


        instrinsics = params['intrinsics']
        fx = instrinsics['f_x']
        fy = instrinsics['f_y']
        cx = instrinsics['c_x']
        cy = instrinsics['c_y']

        # creating the intrinsic matrix
        K = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
        
        rot = params['lens_pose_rotation']
        q = np.array([rot['qw'], rot['qx'], rot['qy'], rot['qz']])

        # Normalize the quaternion to ensure it's a unit quaternion
        q = q / np.linalg.norm(q)

        # convert quaternion to rotation matrix
        R = np.array([
                [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
                [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
                [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)]
            ])
        
        trans = params['lens_pose_translation']
        t = np.array([trans['tx'],trans['ty'], trans['tz']]).reshape(3,1)

        # Concatenate R and t to form the extrinsic matrix
        RT = np.hstack([R, t])

        # Calculate the projection matrix
        P = np.dot(K, RT)

        return K, P, R, t
    
    def read_video(self, video_path: Path):
        """Reads the video frame by frame and returns the frame id and the frame itself"""
        video = cv2.VideoCapture(str(video_path))
        frame_id = 0

        while True:
            success, frame = video.read()

            if not success:
                break

            yield frame_id, frame

            frame_id += 1

        video.release()

    def __get_matches(self, previous_frame: np.ndarray, current_frame: np.ndarray):
        """ Gets the good matches between the previous frame and the current frame
            and returns good keypoints matches position in previous frame and current frame

        Parameters
        ----------
        previous_frame : np.ndarray
            i-1'th frame of the video sequence 
        current_frame : np.ndarray
            i'th frame of the video sequence

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """

        kp1, des1 = self.orb.detectAndCompute(previous_frame, None)
        kp2, des2 = self.orb.detectAndCompute(current_frame, None)

        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
        
        q1 = np.float32([ kp1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ kp2[m.trainIdx].pt for m in good ])

        return q1, q2
    
    def __get_pose(self, q1: np.ndarray, q2: np.ndarray):
        E, _ = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0, None)
        R, t = self.decomp_essential_mat(E, q1, q2)
        return self._form_transf(R,t)
    
    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """


        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)

        
    def run(self):
        previous_frame = None
        current_pose = np.eye(4)
        for frame_id, frame in tqdm(self.read_video(self.video_path)):

            if frame_id  == 2000:
                break

            current_frame = frame
            if previous_frame is None:
                previous_frame = current_frame
                continue

            else:
                q1, q2 = self.__get_matches(previous_frame, current_frame)
                transform = self.__get_pose(q1, q2)
                current_pose = np.matmul(current_pose, np.linalg.inv(transform))
                previous_frame = current_frame
        
            self.estimated_trajectory.append(current_pose)

    
    # def plot_trajectory(self):
    #     trajectory = np.array(self.estimated_trajectory)
    #     x = trajectory[:, 0, 3]
    #     y = trajectory[:, 1, 3]
    #     z = trajectory[:, 2, 3]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     #ax = fig.add_subplot(111)
    #     ax.plot(x, y, z, marker='x')
    #     #ax.plot(x, y, marker='x')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    #     plt.title('Pose Trajectory')
    #     plt.show()

    #     # save the trajectory
    #     np.save('trajectory.npy', trajectory)

    #     # save picture of trajectory
    #     plt.savefig('trajectory.png')


if __name__ == "__main__":
    vo = VisualOdometry(video_path="/home/dadi_vardhan/Downloads/escarda/5_row_aruco/capture_20230509_120504/camera_0_5.mp4")
    vo.run()
    vo.plot_trajectory()
    



        