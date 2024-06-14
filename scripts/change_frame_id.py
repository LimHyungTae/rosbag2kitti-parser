#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import numpy as np

class PointCloudFrameModifier:
    def __init__(self, save_on):
        self.save_on = save_on

        # Initialize ROS node
        rospy.init_node('pointcloud_frame_modifier', anonymous=True)

        # Subscribe to the input point cloud topic
        self.sub = rospy.Subscriber('/os1_cloud_node/points', PointCloud2, self.callback)

        # Publisher for the modified point cloud topic
        self.pub = rospy.Publisher('/os1_cloud_node_modified/points', PointCloud2, queue_size=10000)

    def callback(self, msg):
        # Change the frame_id of the PointCloud2 message
        msg.header.frame_id = "os1_lidar"

        if self.save_on:# Print the timestamp of the message
            secs = msg.header.stamp.secs
            nsecs = f"{msg.header.stamp.nsecs:09d}"
            print(f"Time: {secs}.{nsecs}")

            field_names = [field.name for field in msg.fields]
            print("Fields:", field_names)

            # Convert PointCloud2 message to PCL PointCloud
            cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            t = (list(pc2.read_points(msg, field_names=("t"), skip_nans=True)))
            print(t)
            # print(len(cloud_points), len(cloud_points[0]))
            pcl_cloud = pcl.PointCloud()
            pcl_cloud.from_list(cloud_points)

            # Save to PCD file
            filename = f"../../tmp/saved_scans/cloud_{secs}_{nsecs}.pcd"
            pcl.save(pcl_cloud, filename)
            rospy.loginfo(f"Saved point cloud to {filename}")

        # Publish the modified message
        self.pub.publish(msg)

    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    try:
        point_cloud_modifier = PointCloudFrameModifier(False)
        point_cloud_modifier.run()
    except rospy.ROSInterruptException:
        pass
