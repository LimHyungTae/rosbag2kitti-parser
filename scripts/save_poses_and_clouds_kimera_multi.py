#!/usr/bin/env python
import rosbag
import rospy
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import pcl
import argparse
import numpy as np
import pandas as pd
import time
import os
import re
from utils import *
from tqdm import tqdm
import tf
from cv_bridge import CvBridge
import cv2

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

pd.set_option('display.float_format', lambda x: '%.20g' % x)

path_cache = {}

class BaseSyncSaver:
    def __init__(self, config):
        self.config = config

        self.output_pose_path = config.output_pose_path
        self.output_pcd_dir = config.output_pcd_dir
        self.save_on = config.save_on

        self.file_counter = 0
        # Initialize ROS node
        rospy.init_node('rosbag_data_parser', anonymous=True)

        # Subscribe to the input point cloud topic
        self.sub = rospy.Subscriber(config.input_topic_name, PointCloud2, self.callback, queue_size=100000)

        # Publisher for the modified point cloud topic
        self.pub = rospy.Publisher(config.output_topic_name, PointCloud2, queue_size=10)
        self.pub_transformed = rospy.Publisher("/transformed_cloud", PointCloud2, queue_size=10)
        self.pub_poses = rospy.Publisher("/poses", Odometry, queue_size=10)
        self.pub_img = rospy.Publisher('/forward_image', CompressedImage, queue_size=10)

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

class KimeraMultiSyncSaver(BaseSyncSaver):
    def __init__(self, config):
        super().__init__(config)
        BaseSyncSaver(config)
        self.pose_txt_path = config.pose_txt_path
        self.bag_file = config.bag_path
        self.target_robot = config.target_robot
        self.lidar_topic_name = '/' + self.target_robot + '/lidar_points'
        self.camera_topic_name = '/' + self.target_robot + '/forward/color/image_raw/compressed'
        print("\033[1;32m", self.lidar_topic_name, "\033[0m")
        self.time_offset = 0.0

        # unit: sec
        self.times, self.poses = self.load_pose(self.pose_txt_path)

        assert len(self.times) == len(self.poses)
        self.key_times = np.array(self.times)
        self.key_rots = R.from_matrix([pose[:3, :3] for pose in self.poses])

        self.slerp = Slerp(self.key_times, self.key_rots)

        self.accumulated_clouds = []

    def load_pose(self, csv_path):
        data = pd.read_csv(csv_path, delimiter=',')

        timestamps_of_poses = data['#timestamp_kf'] * 1e-9

        poses = []
        for i in range(len(data)):
            pose = np.eye(4)
            pose[0, 3] = data['x'][i]
            pose[1, 3] = data['y'][i]
            pose[2, 3] = data['z'][i]
            q = [data['qx'][i], data['qy'][i], data['qz'][i], data['qw'][i]]
            pose[:3, :3] = R.from_quat(q).as_matrix()
            poses.append(pose)

        return timestamps_of_poses, poses

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

    def save_pointcloud(self, points_to_be_saved):
        pcl_cloud = pcl.PointCloud()
        pcl_cloud.from_array(points_to_be_saved)
        filename = f"{self.output_pcd_dir}/{self.file_counter:06d}.pcd"
        pcl.save(pcl_cloud, filename)
        rospy.loginfo(f"Saved point cloud to {filename}")
        self.file_counter += 1

    def callback(self, msg):
        # Change the frame_id of the PointCloud2 message
        # In the RViz in Ubuntu 20.04, it does not support the frame_id starting with '/'
        if msg.header.frame_id[0]  == '/':
            msg.header.frame_id = "os1_lidar"

        start_time = time.time()

        secs = msg.header.stamp.secs
        nsecs = f"{msg.header.stamp.nsecs:09d}"
        timestamp = float(secs) + float(nsecs) * 1e-9
        print(f"Timestamp in point cloud topic: {timestamp}")
        timestamp += self.time_offset
        print(f"After: {timestamp}")
        if (timestamp < self.key_times[0]):
            print(f"Timestamp is too early! Waiting for next scans...")
            return
        if (timestamp > self.key_times[-1]):
            print(f"Timestamp is out of our boundary! End to save data...")
            return

        # Convert PointCloud2 message to PCL PointCloud
        cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        cloud_np = np.array(cloud_points).reshape(-1, 3)

        nsecs_for_each_pt = np.array(list(pc2.read_points(msg, field_names=("t"), skip_nans=True)))
        nsecs_for_each_pt = np.reshape(nsecs_for_each_pt, (-1))

        # Previous version of interpolation with deskewing
        # pose_for_scan_time = self.interpolate_pose(timestamp)
        # deskewed_points = []
        # for idx in range(cloud_np.shape[0]):
        #     pt = cloud_np[idx, :]
        #     pt_time_offset = (nsecs_for_each_pt[idx]) * 1e-9
        #     pt_timestamp = timestamp + pt_time_offset
        #     corresponding_pose = self.interpolate_pose(pt_timestamp)
        #     transformed_point = corresponding_pose @ np.array([pt[0], pt[1], pt[2], 1.0])
        #     deskewed_points.append(transformed_point[:3])
        # transformed = np.array(deskewed_points, dtype=np.float32)

        # I omit deskewing for translation
        pose_for_scan_time = self.interpolate_pose(timestamp)
        deskewed_points = self.deskew_scan(timestamp, cloud_np, nsecs_for_each_pt)
        transformed = pose_for_scan_time @ np.hstack((deskewed_points, np.ones((deskewed_points.shape[0], 1)))).T
        transformed = transformed.T[:, :3].astype(np.float32)

        # print(len(cloud_points), len(cloud_points[0]))
        pcl_cloud = pcl.PointCloud()
        pcl_cloud.from_array(transformed)

        if self.save_on:
            with open(self.output_pose_path, 'a') as f:
                pose_flattened = pose_for_scan_time[:3, :].flatten()
                pose_str = ' '.join(map(str, pose_flattened))
                f.write(pose_str + '\n')
            # Save to PCD file
            # Follow the MulRan dataset format
            filename = f"{self.output_pcd_dir}/{secs}_{nsecs}.pcd"
            pcl.save(pcl_cloud, filename)
            rospy.loginfo(f"Saved point cloud to {filename}")

        # Convert transformed numpy array back to PointCloud2 message
        header = msg.header
        header.frame_id = "map"
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg_transformed = pc2.create_cloud(header, fields, transformed)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Publish the modified message
        self.pub.publish(msg)
        self.pub_transformed.publish(msg_transformed)

    def publish_odometry(self, pose, timestamp):
        odom = Odometry()
        odom.header.stamp = rospy.Time.from_sec(timestamp)
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = pose[0, 3]
        odom.pose.pose.position.y = pose[1, 3]
        odom.pose.pose.position.z = pose[2, 3]

        q = tf.transformations.quaternion_from_matrix(pose)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        self.pub_poses.publish(odom)
    def parse_rosbag(self):
        # Open the bag file
        bag = rosbag.Bag(self.bag_file)

        # Iterate through the messages in the bag
        print(self.times)
        try:
            count = 0
            # Note 't' does not match to the `#timestamp_kf` in the `gt_odom` csv
            for topic, msg, t in bag.read_messages(topics=[self.lidar_topic_name]):
                # Get the current time
                timestamp = msg.header.stamp.to_sec()

                if (timestamp < self.key_times[0]):
                    print(f"Timestamp is too early! Skip visualization...")
                    continue
                if (timestamp > self.key_times[-1]):
                    print(f"Timestamp is out of our boundary! End to save data...")
                    continue

                # print("TS of LiDAR msg: ", msg.header.stamp.to_sec(), " <--> TS of bag: ", t.to_sec())
                start_time = time.time()

                # print(f"Current time in bag: {current_time} seconds, {topic}")
                cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
                cloud_np = np.array(cloud_points).reshape(-1, 3)

                nsecs_for_each_pt = np.linspace(0, 0.1, cloud_np.shape[0])
                nsecs_for_each_pt = np.reshape(nsecs_for_each_pt, (-1))

                pose_for_scan_time = self.interpolate_pose(timestamp)
                if self.config.use_deskewing:
                    deskewed_points = self.deskew_scan(timestamp, cloud_np, nsecs_for_each_pt)
                    transformed = pose_for_scan_time @ np.hstack(
                        (deskewed_points, np.ones((deskewed_points.shape[0], 1)))).T
                else:
                    print("\033[1;33m", "Deskewing deactivated.\033[0m")
                    transformed = pose_for_scan_time @ np.hstack(
                        (cloud_np, np.ones((cloud_np.shape[0], 1)))).T

                transformed = transformed.T[:, :3].astype(np.float32)

                pcl_transformed = pcl.PointCloud()
                pcl_transformed.from_array(transformed)
                voxel_filter = pcl_transformed.make_voxel_grid_filter()
                voxel_filter.set_leaf_size(0.1, 0.1, 0.1)
                downsampled_cloud = voxel_filter.filter()
                self.accumulated_clouds.append(downsampled_cloud)

                if self.save_on and self.config.use_deskewing:
                    with open(self.output_pose_path, 'a') as f:
                        pose_flattened = pose_for_scan_time[:3, :].flatten()
                        pose_str = ' '.join(map(str, pose_flattened))
                        f.write(pose_str + '\n')
                    # Save to PCD file
                    # Follow the MulRan dataset format
                    self.save_pointcloud(deskewed_points.astype(np.float32))

                header = msg.header
                header.frame_id = "map"
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                ]
                msg_transformed = pc2.create_cloud(header, fields, transformed)

                self.publish_odometry(pose_for_scan_time, timestamp)

                end_time = time.time()
                elapsed_time = end_time - start_time

                # Publish the modified message
                self.pub.publish(msg)
                self.pub_transformed.publish(msg_transformed)

        except KeyboardInterrupt:
            print("Interrupted by user, stopping the loop.")

        combined_cloud = self.accumulated_clouds[0].to_array()
        for idx, cloud in enumerate(self.accumulated_clouds[1:]):
            if idx % 1000 == 0:
                print(idx, " / ", len(self.accumulated_clouds))
            tmp = cloud.to_array()
            combined_cloud = np.vstack([combined_cloud, tmp])
        pcl_combined_cloud = pcl.PointCloud()
        pcl_combined_cloud.from_array(combined_cloud)
        voxel_filter = pcl_combined_cloud.make_voxel_grid_filter()
        voxel_filter.set_leaf_size(0.1, 0.1, 0.1)
        downsampled_cloud = voxel_filter.filter()

        map_name = f"{self.output_pcd_dir}/map_cloud.pcd"
        pcl.save(downsampled_cloud, map_name)
        # Close the bag file
        bag.close()

class NewerCollegeSyncSaver(BaseSyncSaver):
    def __init__(self, config):
        super().__init__(config)
        BaseSyncSaver(config)
        self.pose_txt_path = config.pose_txt_path
        self.bag_file = config.bag_path
        self.lidar_topic_name = '/os1_cloud_node/points'
        print("\033[1;32m", self.lidar_topic_name, "\033[0m")
        self.time_offset = 0.0

        # unit: sec
        self.times, self.poses = self.load_pose(self.pose_txt_path)

        assert len(self.times) == len(self.poses)
        self.key_times = np.array(self.times)
        self.key_rots = R.from_matrix([pose[:3, :3] for pose in self.poses])

        self.slerp = Slerp(self.key_times, self.key_rots)

        self.accumulated_clouds = []

    def load_pose(self, csv_path):
         data = pd.read_csv(csv_path, delimiter=',')
         # Ensure the 'sec' and 'nsec' columns are treated as integers
         data['sec'] = data['sec'].astype(int)
         data['nsec'] = data['nsec'].astype(int)

         # Extract the timestamps and poses
         timestamps_of_poses = data['sec'] + data['nsec'] * 1e-9
         poses = []
         for i in range(len(data)):
             pose = np.eye(4)
             pose[0, 3] = data['x'][i]
             pose[1, 3] = data['y'][i]
             pose[2, 3] = data['z'][i]
             q = [data['qx'][i], data['qy'][i], data['qz'][i], data['qw'][i]]
             pose[:3, :3] = R.from_quat(q).as_matrix()
             poses.append(pose)
         return timestamps_of_poses, poses

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

    def save_pointcloud(self, points_to_be_saved):
        pcl_cloud = pcl.PointCloud()
        pcl_cloud.from_array(points_to_be_saved)
        filename = f"{self.output_pcd_dir}/{self.file_counter:06d}.pcd"
        pcl.save(pcl_cloud, filename)
        rospy.loginfo(f"Saved point cloud to {filename}")
        self.file_counter += 1

    def callback(self, msg):
        # Change the frame_id of the PointCloud2 message
        # In the RViz in Ubuntu 20.04, it does not support the frame_id starting with '/'
        if msg.header.frame_id[0]  == '/':
            msg.header.frame_id = "os1_lidar"

        start_time = time.time()

        secs = msg.header.stamp.secs
        nsecs = f"{msg.header.stamp.nsecs:09d}"
        timestamp = float(secs) + float(nsecs) * 1e-9
        print(f"Timestamp in point cloud topic: {timestamp}")
        timestamp += self.time_offset
        print(f"After: {timestamp}")
        if (timestamp < self.key_times[0]):
            print(f"Timestamp is too early! Waiting for next scans...")
            return
        if (timestamp > self.key_times[-1]):
            print(f"Timestamp is out of our boundary! End to save data...")
            return

        # Convert PointCloud2 message to PCL PointCloud
        cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        cloud_np = np.array(cloud_points).reshape(-1, 3)

        nsecs_for_each_pt = np.array(list(pc2.read_points(msg, field_names=("t"), skip_nans=True)))
        nsecs_for_each_pt = np.reshape(nsecs_for_each_pt, (-1))

        # Previous version of interpolation with deskewing
        # pose_for_scan_time = self.interpolate_pose(timestamp)
        # deskewed_points = []
        # for idx in range(cloud_np.shape[0]):
        #     pt = cloud_np[idx, :]
        #     pt_time_offset = (nsecs_for_each_pt[idx]) * 1e-9
        #     pt_timestamp = timestamp + pt_time_offset
        #     corresponding_pose = self.interpolate_pose(pt_timestamp)
        #     transformed_point = corresponding_pose @ np.array([pt[0], pt[1], pt[2], 1.0])
        #     deskewed_points.append(transformed_point[:3])
        # transformed = np.array(deskewed_points, dtype=np.float32)

        # I omit deskewing for translation
        pose_for_scan_time = self.interpolate_pose(timestamp)
        deskewed_points = self.deskew_scan(timestamp, cloud_np, nsecs_for_each_pt)
        transformed = pose_for_scan_time @ np.hstack((deskewed_points, np.ones((deskewed_points.shape[0], 1)))).T
        transformed = transformed.T[:, :3].astype(np.float32)

        # print(len(cloud_points), len(cloud_points[0]))
        pcl_cloud = pcl.PointCloud()
        pcl_cloud.from_array(transformed)

        if self.save_on:
            with open(self.output_pose_path, 'a') as f:
                pose_flattened = pose_for_scan_time[:3, :].flatten()
                pose_str = ' '.join(map(str, pose_flattened))
                f.write(pose_str + '\n')
            # Save to PCD file
            # Follow the MulRan dataset format
            filename = f"{self.output_pcd_dir}/{secs}_{nsecs}.pcd"
            pcl.save(pcl_cloud, filename)
            rospy.loginfo(f"Saved point cloud to {filename}")

        # Convert transformed numpy array back to PointCloud2 message
        header = msg.header
        header.frame_id = "map"
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg_transformed = pc2.create_cloud(header, fields, transformed)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Publish the modified message
        self.pub.publish(msg)
        self.pub_transformed.publish(msg_transformed)

    def publish_odometry(self, pose, timestamp):
        odom = Odometry()
        odom.header.stamp = rospy.Time.from_sec(timestamp)
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = pose[0, 3]
        odom.pose.pose.position.y = pose[1, 3]
        odom.pose.pose.position.z = pose[2, 3]

        q = tf.transformations.quaternion_from_matrix(pose)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        self.pub_poses.publish(odom)
    def parse_rosbag(self):
        # Open the bag file
        bag = rosbag.Bag(self.bag_file)

        # Iterate through the messages in the bag
        print(self.times)
        try:
            count = 0
            # Note 't' does not match to the `#timestamp_kf` in the `gt_odom` csv
            for topic, msg, t in bag.read_messages(topics=[self.lidar_topic_name]):
                # Get the current time
                timestamp = msg.header.stamp.to_sec()

                if (timestamp < self.key_times[0]):
                    print(f"Timestamp is too early! Skip visualization...")
                    continue
                if (timestamp > self.key_times[-1]):
                    print(f"Timestamp is out of our boundary! End to save data...")
                    continue

                # print("TS of LiDAR msg: ", msg.header.stamp.to_sec(), " <--> TS of bag: ", t.to_sec())
                start_time = time.time()

                # print(f"Current time in bag: {current_time} seconds, {topic}")
                cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
                cloud_np = np.array(cloud_points).reshape(-1, 3)

                nsecs_for_each_pt = np.linspace(0, 0.1, cloud_np.shape[0])
                nsecs_for_each_pt = np.reshape(nsecs_for_each_pt, (-1))

                pose_for_scan_time = self.interpolate_pose(timestamp)
                if self.config.use_deskewing:
                    deskewed_points = self.deskew_scan(timestamp, cloud_np, nsecs_for_each_pt)
                    transformed = pose_for_scan_time @ np.hstack(
                        (deskewed_points, np.ones((deskewed_points.shape[0], 1)))).T
                else:
                    print("\033[1;33m", "Deskewing deactivated.\033[0m")
                    transformed = pose_for_scan_time @ np.hstack(
                        (cloud_np, np.ones((cloud_np.shape[0], 1)))).T

                transformed = transformed.T[:, :3].astype(np.float32)

                pcl_transformed = pcl.PointCloud()
                pcl_transformed.from_array(transformed)
                voxel_filter = pcl_transformed.make_voxel_grid_filter()
                voxel_filter.set_leaf_size(0.1, 0.1, 0.1)
                downsampled_cloud = voxel_filter.filter()
                self.accumulated_clouds.append(downsampled_cloud)

                if self.save_on and self.config.use_deskewing:
                    with open(self.output_pose_path, 'a') as f:
                        pose_flattened = pose_for_scan_time[:3, :].flatten()
                        pose_str = ' '.join(map(str, pose_flattened))
                        f.write(pose_str + '\n')
                    # Save to PCD file
                    # Follow the MulRan dataset format
                    self.save_pointcloud(deskewed_points.astype(np.float32))

                header = msg.header
                header.frame_id = "map"
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                ]
                msg_transformed = pc2.create_cloud(header, fields, transformed)

                self.publish_odometry(pose_for_scan_time, timestamp)

                end_time = time.time()
                elapsed_time = end_time - start_time

                # Publish the modified message
                self.pub.publish(msg)
                self.pub_transformed.publish(msg_transformed)

        except KeyboardInterrupt:
            print("Interrupted by user, stopping the loop.")

        # combined_cloud = self.accumulated_clouds[0].to_array()
        # for idx, cloud in enumerate(self.accumulated_clouds[1:]):
        #     if idx % 1000 == 0:
        #         print(idx, " / ", len(self.accumulated_clouds))
        #     tmp = cloud.to_array()
        #     combined_cloud = np.vstack([combined_cloud, tmp])
        # pcl_combined_cloud = pcl.PointCloud()
        # pcl_combined_cloud.from_array(combined_cloud)
        # voxel_filter = pcl_combined_cloud.make_voxel_grid_filter()
        # voxel_filter.set_leaf_size(0.1, 0.1, 0.1)
        # downsampled_cloud = voxel_filter.filter()
        #
        # map_name = f"{self.output_pcd_dir}/map_cloud.pcd"
        # pcl.save(downsampled_cloud, map_name)
        # Close the bag file
        bag.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change the frame_id of a PointCloud2 message and save pcd files.")
    parser.add_argument('--output_pose_path', type=str, help='Output file for the interpolated poses.')
    parser.add_argument('--output_pcd_dir', type=str, help='Output file for the accumulated map.')

    parser.add_argument('--save_on', type=bool, default=True, help='Boolean flag to save pcd files.')
    parser.add_argument('--use_deskewing', type=bool, default=False, help='Boolean flag for setting deskeweing.')

    parser.add_argument('--input_topic_name', type=str, default='/os1_cloud_node/points')
    parser.add_argument('--output_topic_name', type=str, default='/os1_cloud_node/points_w_changed_framed_id')

    parser.add_argument('--pose_txt_path', type=str, default='')
    parser.add_argument('--bag_path', type=str, default='')
    # Only for Kimera-multi
    parser.add_argument('--target_robot', type=str, default='')

    parser.add_argument('--target_dataset', type=str, default='kimera-multi', choices=['kimera-multi', 'newer-college'],
                        help='Boolean flag to save pcd files.')
    args = parser.parse_args()

    try:
        if args.target_dataset == "kimera-multi":
            print("Using Kimera-Multi dataset")
            sub_and_sync_saver = KimeraMultiSyncSaver(args)
            sub_and_sync_saver.parse_rosbag()
        elif args.target_dataset == "newer-college":
            print("Using Newer College dataset")
            sub_and_sync_saver = NewerCollegeSyncSaver(args)
            sub_and_sync_saver.parse_rosbag()

    except rospy.ROSInterruptException:
        pass
