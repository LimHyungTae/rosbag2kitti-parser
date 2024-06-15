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

## Newer College Dataset

TBU

## Kimera-Multi Dataset

TBU
