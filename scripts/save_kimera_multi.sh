target_robot="thoth" # acl_jackal, acl_jackal2, hathor, sparkal1, sparkal2, thoth
output_dir="../../tmp/kimera-multi/10_14_${target_robot}"
output_pcd_dir="${output_dir}/velodyne"

# Remove the output directory if it exists
# sudo rm -rf ${output_dir}

# Create the output directory
mkdir -p "${output_pcd_dir}"

python3 save_poses_and_clouds_kimera_multi.py --output_pose_path "../../tmp/kimera-multi/10_14_${target_robot}/poses.txt" --output_pcd_dir $output_pcd_dir --save_on False --bag_path /media/shapelim/UX960NVMe1/kimera-multi/campus_outdoor_1014/10_14_${target_robot}.bag --pose_txt_path /media/shapelim/UX960NVMe1/kimera-multi/campus_outdoor_1014/${target_robot}_gt_odom.csv --target_robot ${target_robot}


