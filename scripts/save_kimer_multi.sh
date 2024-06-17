target_robot=acl_jackal # acl_jackal, acl_jackal2, hathor, sparkal1, sparkal2, thoth
 
python3 save_poses_and_clouds_kimera_multi.py --output_pose_path "../../tmp/test.txt" --output_pcd_dir "../../tmp/kimera_multi/hehe" --save_on False --bag_path /media/shapelim/UX960NVMe1/kimera-multi/campus_outdoor_1014/10_14_${target_robot}.bag --pose_txt_path /media/shapelim/UX960NVMe1/kimera-multi/campus_outdoor_1014/${target_robot}_gt_odom.csv

