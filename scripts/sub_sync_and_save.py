#!/usr/bin/env python
import rosbag
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import pcl
import argparse
import numpy as np
import time
import re

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

path_cache = {}

def Log(R):
    theta = 0.0 if R.trace() > 3.0 - 1e-6 else np.arccos(0.5 * (R.trace() - 1))
    K = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.abs(theta) < 0.001:
        return 0.5 * K
    else:
        return 0.5 * theta / np.sin(theta) * K

def Exp(ang):
    ang_norm = np.linalg.norm(ang)
    Eye3 = np.eye(3)
    if ang_norm > 0.0000001:
        r_axis = ang / ang_norm
        K = np.array([[0, -r_axis[2], r_axis[1]],
                      [r_axis[2], 0, -r_axis[0]],
                      [-r_axis[1], r_axis[0], 0]])
        return Eye3 + np.sin(ang_norm) * K + (1.0 - np.cos(ang_norm)) * np.dot(K, K)
    else:
        return Eye3

class BaseSubAndSyncSaver:
    def __init__(self, config):
        self.output_pose_path = config.output_pose_path
        self.output_pcd_dir = config.output_pcd_dir
        self.save_on = config.save_on

        # Initialize ROS node
        rospy.init_node('pointcloud_frame_modifier', anonymous=True)

        # Subscribe to the input point cloud topic
        self.sub = rospy.Subscriber(config.input_topic_name, PointCloud2, self.callback, queue_size=100000)

        # Publisher for the modified point cloud topic
        self.pub = rospy.Publisher(config.output_topic_name, PointCloud2, queue_size=1)

        self.pub_transformed = rospy.Publisher("/transformed_cloud", PointCloud2, queue_size=1)

    def callback(self, msg):
        # Change the frame_id of the PointCloud2 message
        # In the Rviz in Ubuntu 20.04, it does not support the frame_id starting with '/'
        if msg.header.frame_id[0] == '/':
            msg.header.frame_id = "os1_lidar"
    def run(self):
        # Keep the node running
        rospy.spin()
    def save_synced_poses(self, poses_sync, output_file):
        with open(output_file, 'w') as f:
            for pose in poses_sync:
                pose_flattened = pose[:3, :].flatten()
                pose_str = ' '.join(map(str, pose_flattened))
                f.write(pose_str + '\n')

    def get_gt_odometry(self, data_path, indices=None, ext='.txt', return_all=False):
        if data_path not in path_cache:
            path_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return path_cache[data_path]
        else:
            return path_cache[data_path][indices]
    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0
    def interpolate_poses(self, timestamps_of_poses, poses, timestamp):
        assert len(timestamps_of_poses) == len(poses)
        poses_sync = []
        key_times = np.array(timestamps_of_poses)
        key_rots = R.from_matrix([pose[:3, :3] for pose in poses])

        slerp = Slerp(key_times, key_rots)

        if timestamp < key_times[0]:
            print("\033[1;33mGiven timesamp is timestamp < key_times[0]!\033[0m")
            return poses_sync.append(poses[0])
        elif timestamp > key_times[-1]:
            print("\033[1;33mGiven timesamp is timestamp > key_times[-1]!\033[0m")
            return poses_sync.append(poses[-1])
        else:
            interp_rot = slerp(timestamp)
            idx = np.searchsorted(key_times, timestamp) - 1
            pose0 = poses[idx]
            pose1 = poses[idx + 1]

            m = (timestamp - key_times[idx]) / (key_times[idx + 1] - key_times[idx])
            n = 1 - m

            interp_pose = np.eye(4)
            interp_pose[:3, :3] = interp_rot.as_matrix()
            interp_pose[0, 3] = m * pose1[0, 3] + n * pose0[0, 3]
            interp_pose[1, 3] = m * pose1[1, 3] + n * pose0[1, 3]
            interp_pose[2, 3] = m * pose1[2, 3] + n * pose0[2, 3]

            return interp_pose

    def get_closest_idx(self, timestamps_np, timestamp_for_scan):
        closest_idx = np.searchsorted(timestamps_np, timestamp_for_scan) - 1
        if abs(timestamps_np[closest_idx] - timestamp_for_scan) > abs(timestamps_np[closest_idx + 1] - timestamp_for_scan):
            closest_idx = closest_idx + 1
        return closest_idx

class ViVidPlusPlusSubAndSyncSaver(BaseSubAndSyncSaver):
    def __init__(self, config):
        super().__init__(config)
        BaseSubAndSyncSaver(config)
        self.pose_txt_path = config.pose_txt_path
        self.time_txt_path = config.time_txt_path

        if "campus_day2" in self.pose_txt_path:
            self.poses = self.load_campus_pose(self.pose_txt_path)
        elif "city_day2" in self.pose_txt_path:
            self.poses = self.load_city_pose(self.pose_txt_path)
        else:
            raise ValueError("Un-tested sequence is given.")
        # unit: sec
        self.times = np.loadtxt(self.time_txt_path)

        assert len(self.times) == len(self.poses)
        self.key_times = np.array(self.times)
        self.key_rots = R.from_matrix([pose[:3, :3] for pose in self.poses])

        self.slerp = Slerp(self.key_times, self.key_rots)

    def load_campus_pose(self, pose_txt_path):
        poses = []
        poses_kitti_format = np.loadtxt(pose_txt_path)

        for i in range(poses_kitti_format.shape[0]):
            pose = np.eye(4)
            for j in range(12):
                pose[j // 4, j % 4] = poses_kitti_format[i, j]
            poses.append(pose)
        return poses

    def load_city_pose(self, pose_txt_path):
        with open(pose_txt_path, 'r') as file:
            lines = file.readlines()

        poses = []
        for line in lines:
            parts = re.split(r'\s+', line.strip())
            # Extract the translation and quaternion parts
            tx, ty, tz = map(float, parts[2:5])
            qx, qy, qz, qw = map(float, parts[5:])

            # Convert quaternion to rotation matrix
            rotation = R.from_quat([qx, qy, qz, qw])
            rotation_matrix = rotation.as_matrix()

            # Create 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = [tx, ty, tz]

            poses.append(transformation_matrix)

        return poses

    def interpolate_pose(self, timestamp):
        if timestamp < self.key_times[0]:
            print("\033[1;33mGiven timesamp is timestamp < key_times[0]!\033[0m")
            return self.poses[0]
        elif timestamp > self.key_times[-1]:
            print("\033[1;33mGiven timesamp is timestamp > key_times[-1]!\033[0m")
            return self.poses[-1]
        else:
            # Interpolate rotation
            interp_rot = self.slerp(timestamp)
            # Interpolate translation
            idx = np.searchsorted(self.key_times, timestamp) - 1
            pose0 = self.poses[idx]
            pose1 = self.poses[idx + 1]

            m = (timestamp - self.key_times[idx]) / (self.key_times[idx + 1] - self.key_times[idx])
            n = 1 - m

            interp_pose = np.eye(4)
            interp_pose[:3, :3] = interp_rot.as_matrix()
            interp_pose[0, 3] = m * pose1[0, 3] + n * pose0[0, 3]
            interp_pose[1, 3] = m * pose1[1, 3] + n * pose0[1, 3]
            interp_pose[2, 3] = m * pose1[2, 3] + n * pose0[2, 3]

            return interp_pose

    def deskew_scan(self, timestamp_for_scan, points_body_frame, nsecs_for_each_pt):
        time_offset = 0.1
        curr_pose = self.interpolate_pose(timestamp_for_scan)
        next_pose = self.interpolate_pose(timestamp_for_scan + time_offset)

        rel_pose = np.linalg.inv(curr_pose) @ next_pose
        rot_vec = Log(rel_pose[:3, :3])
        ang_vel = rot_vec / time_offset
        # Degree
        ang_norm = np.linalg.norm(rot_vec) * 180.0 / 3.141592

        # Following LIO-SAM, we just deskew rotation, not translation
        if ang_norm < 0.03:
            print("\033[1;33mToo small angle diff. detected. Skip deskewing.\033[0m")
            return points_body_frame
        else:
            deskewed_points = []
            for idx in range(points_body_frame.shape[0]):
                pt = points_body_frame[idx, :]
                pt_time_offset = nsecs_for_each_pt[idx] * 1e-9
                transformed_point = Exp(ang_vel * pt_time_offset) @ np.array([pt[0], pt[1], pt[2]]).T
                deskewed_points.append(transformed_point.T)
            transformed = np.array(deskewed_points, dtype=np.float32)

            return transformed

    def callback(self, msg):
        # Change the frame_id of the PointCloud2 message
        # In the RViz in Ubuntu 20.04, it does not support the frame_id starting with '/'
        if msg.header.frame_id[0]  == '/':
            msg.header.frame_id = "os1_lidar"

        start_time = time.time()

        secs = msg.header.stamp.secs
        nsecs = f"{msg.header.stamp.nsecs:09d}"
        timestamp = float(secs) + float(nsecs) * 1e-9
        print(timestamp, rospy.Time.now())
        print(timestamp, rospy.Time.now())
        print(timestamp, rospy.Time.now())


        # # Convert PointCloud2 message to PCL PointCloud
        # cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        # cloud_np = np.array(cloud_points).reshape(-1, 3)
        #
        # nsecs_for_each_pt = np.array(list(pc2.read_points(msg, field_names=("t"), skip_nans=True)))
        # nsecs_for_each_pt = np.reshape(nsecs_for_each_pt, (-1))
        #
        # # Previous version of interpolation with deskewing
        # # pose_for_scan_time = self.interpolate_pose(timestamp)
        # # deskewed_points = []
        # # for idx in range(cloud_np.shape[0]):
        # #     pt = cloud_np[idx, :]
        # #     pt_time_offset = (nsecs_for_each_pt[idx]) * 1e-9
        # #     pt_timestamp = timestamp + pt_time_offset
        # #     corresponding_pose = self.interpolate_pose(pt_timestamp)
        # #     transformed_point = corresponding_pose @ np.array([pt[0], pt[1], pt[2], 1.0])
        # #     deskewed_points.append(transformed_point[:3])
        # # transformed = np.array(deskewed_points, dtype=np.float32)
        #
        # # I omit deskewing for translation
        # pose_for_scan_time = self.interpolate_pose(timestamp)
        # deskewed_points = self.deskew_scan(timestamp, cloud_np, nsecs_for_each_pt)
        # transformed = pose_for_scan_time @ np.hstack((deskewed_points, np.ones((deskewed_points.shape[0], 1)))).T
        # transformed = transformed.T[:, :3].astype(np.float32)
        #
        # # print(len(cloud_points), len(cloud_points[0]))
        # pcl_cloud = pcl.PointCloud()
        # pcl_cloud.from_array(transformed)
        #
        # if self.save_on:
        #     with open(self.output_pose_path, 'a') as f:
        #         pose_flattened = pose_for_scan_time[:3, :].flatten()
        #         pose_str = ' '.join(map(str, pose_flattened))
        #         f.write(pose_str + '\n')
        #     # Save to PCD file
        #     # Follow the MulRan dataset format
        #     filename = f"{self.output_pcd_dir}/{secs}_{nsecs}.pcd"
        #     pcl.save(pcl_cloud, filename)
        #     rospy.loginfo(f"Saved point cloud to {filename}")
        #
        # # Convert transformed numpy array back to PointCloud2 message
        # header = msg.header
        # header.frame_id = "map"
        # fields = [
        #     PointField('x', 0, PointField.FLOAT32, 1),
        #     PointField('y', 4, PointField.FLOAT32, 1),
        #     PointField('z', 8, PointField.FLOAT32, 1),
        # ]
        # msg_transformed = pc2.create_cloud(header, fields, transformed)
        #
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        #
        # # Publish the modified message
        # self.pub.publish(msg)
        # self.pub_transformed.publish(msg_transformed)
        # print("Published the " + str(timestamp) + " message! (", elapsed_time, "sec taken)")

def get_bag_time(bag_file):
    # Open the bag file
    bag = rosbag.Bag(bag_file)

    # Iterate through the messages in the bag
    try:
        count = 0
        for topic, msg, t in bag.read_messages():
            # Get the current time
            current_time = t.to_sec()
            if topic == "/os1_cloud_node/points":
                print(f"Current time in bag: {current_time} seconds, {topic}")
                count += 1
            if count > 5:
                break
    except KeyboardInterrupt:
        print("Interrupted by user, stopping the loop.")

    # Close the bag file
    bag.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change the frame_id of a PointCloud2 message and save pcd files.")
    parser.add_argument('--output_pose_path', type=str, help='Output file for the interpolated poses.')
    parser.add_argument('--output_pcd_dir', type=str, help='Output file for the accumulated map.')

    parser.add_argument('--save_on', type=bool, default=True, help='Boolean flag to save pcd files.')

    parser.add_argument('--input_topic_name', type=str, default='/os1_cloud_node/points')
    parser.add_argument('--output_topic_name', type=str, default='/os1_cloud_node/points_w_changed_framed_id')

    parser.add_argument('--target_dataset', type=str, default='vivid++', choices=['vivid++', 'newer college'],
                        help='Boolean flag to save pcd files.')
    # For ViVid++ dataset
    parser.add_argument('--pose_txt_path', type=str, default='/home/shapelim/Downloads/vivid++/city_day2_optimized_poses.txt')
    parser.add_argument('--time_txt_path', type=str, default='/home/shapelim/Downloads/vivid++/city_day2_times.txt')

    args = parser.parse_args()

    try:
        if args.target_dataset == 'vivid++':
            print("Using ViVid++ dataset")
            sub_and_sync_saver = ViVidPlusPlusSubAndSyncSaver(args)
        else:
            sub_and_sync_saver = BaseSubAndSyncSaver(args)
        sub_and_sync_saver.run()

    except rospy.ROSInterruptException:
        pass
