# Rosbag to KITTI Format Parsing

## Supported dataset

* Newer College dataset
* ViVid++ dataset
* Kimera-multi dataset


---

## How to Use

```
python3 sync_and_save.py --csv_path '/media/shapelim/UX960NVMe1/newer-college-dataset/01_short_experiments/ground_truth/registered_poses.csv' --folder_path '/home/shapelim/git/tmp/ouster_scan' --output_file_path './poses.txt' --output_pcd_path "./map.pcd"

```


## ViViD++ Dataset

The timestamps of the ground truths are in consistent.
* `campus_day2`: GT uses local timestamp, which matchs to the point cloud msg time, i.e. there's no time offset. 
* `city_day2`: GT uses rosbag timestamp, but the message is not in the rosbag. So, we need to use the time offset
  * See `vivid_plusplus_city_day2_time_anaylses.ods`. I sampled the five recent timestamps, and do average to get the time offset.
  * 

### How to use

After setting appropriate paths of `pose_txt_path` and `time_txt_path`, run the following command:

```angular2html
python3 save_poses_and_clouds_vivid_plusplus.py --output_pose_path ${poses txt path} --output_pcd_dir ${pcd dir}
```

For example, 

```angular2html
python3 save_poses_and_clouds_vivid_plusplus.py --output_pose_path /home/shapelim/git/tmp/vivivd++_1_0/city_day2/poses.txt --output_pcd_dir /home/shapelim/git/tmp/vivivd++_1_0/city_day2/scans
```

## Newer College Dataset

In this project, we calculated the transformation between the camera and LiDAR sensor. To achieve this, we used URDF (Unified Robot Description Format). URDF is an XML format used to describe the physical configuration of robots. It allows us to define the relationship between different parts of the robot.

To obtain the transformation between the camera base and the LiDAR, we executed the following command:

```angular2html
rosrun tf tf_echo base os_lidar
```

```angular2html
At time 0.000
- Translation: [-0.084, -0.025, 0.050]
- Rotation: in Quaternion [0.000, 0.000, 0.924, 0.383]
            in RPY (radian) [0.000, -0.000, 2.356]
            in RPY (degree) [0.000, -0.000, 135.000]

```

Here, `base` refers to the left camera frame. More details about the description can be found [here](https://ori-drs.github.io/newer-college-dataset/stereo-cam/platform-stereo/).


## Kimera-Multi Dataset

TBU
