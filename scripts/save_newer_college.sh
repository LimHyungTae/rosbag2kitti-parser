target_scene="01_short_experiments"
output_dir="../../tmp/newer-college/${target_scene}"
output_pcd_dir="${output_dir}/ouster"

# Remove the output directory if it exists
# sudo rm -rf ${output_dir}

# Create the output directory
mkdir -p "${output_pcd_dir}"

python3 save_poses_and_clouds_kimera_multi.py \
  --output_pose_path "${output_dir}/poses.txt" \
  --output_pcd_dir $output_pcd_dir \
  --save_on False \
  --bag_path /media/shapelim/UX960NVMe1/newer-college-dataset/01_short_experiments/rosbag/rooster_2020-03-10-10-36-30_0.bag \
  --pose_txt_path /media/shapelim/UX960NVMe1/newer-college-dataset/01_short_experiments/ground_truth/registered_poses.csv \
  --target_dataset "newer-college"


