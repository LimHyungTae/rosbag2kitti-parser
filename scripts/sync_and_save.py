import os
import pandas as pd
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def load_data(csv_path, folder_path):
  # Load the data into a DataFrame
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

  # List all files in the folder
  files_in_folder = os.listdir(folder_path)
  files_in_folder = sorted(files_in_folder)

  # Extract timestamps from the file names
  timestamps_of_scans = []
  for file in files_in_folder:
    basename = os.path.basename(file)
    parts = basename.split('_')
    sec = int(parts[1])
    nsec = int(parts[2].split('.')[0])
    timestamp = sec + nsec * 1e-9
    timestamps_of_scans.append(timestamp)

  return timestamps_of_poses, poses, timestamps_of_scans, files_in_folder


# It strongly assumes that data are already sorted by timestamps
def interpolate_poses(timestamps_of_poses, poses, timestamps_of_scans):
  poses_sync = []
  key_times = np.array(timestamps_of_poses)
  key_rots = R.from_matrix([pose[:3, :3] for pose in poses])

  slerp = Slerp(key_times, key_rots)

  for timestamp in timestamps_of_scans:
    if timestamp < key_times[0]:
      poses_sync.append(poses[0])
    elif timestamp > key_times[-1]:
      poses_sync.append(poses[-1])
    else:
      interp_rot = slerp(timestamp)
      idx = np.searchsorted(key_times, timestamp) - 1
      pose0 = poses[idx]
      pose1 = poses[idx + 1]

      m = (timestamp - key_times[idx]) / (key_times[idx + 1] - key_times[idx])
      n = 1 - m

      # interp_pose = np.eye(4)
      # interp_pose[:3, :3] = interp_rot.as_matrix()
      # interp_pose[0, 3] = m * pose1[0, 3] + n * pose0[0, 3]
      # interp_pose[1, 3] = m * pose1[1, 3] + n * pose0[1, 3]
      # interp_pose[2, 3] = m * pose1[2, 3] + n * pose0[2, 3]
      #
      # poses_sync.append(interp_pose)

      poses_sync.append(poses[idx])

  return poses_sync


def save_poses(poses_sync, output_file):
  with open(output_file, 'w') as f:
    for pose in poses_sync:
      pose_flattened = pose[:3, :].flatten()
      pose_str = ' '.join(map(str, pose_flattened))
      f.write(pose_str + '\n')


def load_and_transform_scans(folder_path, files_in_folder, poses_sync):
  aggregated_points = []
  for i, file in enumerate(tqdm(files_in_folder, desc="Processing scans")):
    pcd = o3d.io.read_point_cloud(os.path.join(folder_path, file))
    points = np.asarray(pcd.points)

    # Apply transformation
    pose = poses_sync[i]
    transformed_points = (pose @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]

    # Add intensity (scan index)
    intensities = np.full((transformed_points.shape[0], 1), i)

    # Combine points and intensities
    points_with_intensity = np.hstack((transformed_points, intensities))
    aggregated_points.append(points_with_intensity)

  map_cloud = np.vstack(aggregated_points)
  return map_cloud


def voxelize_and_save(points, voxel_size, output_file):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points[:, :3])

  print("Voxelizing the point cloud...")
  voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
  print("Saving files to", output_file, "...")
  o3d.io.write_point_cloud(output_file, voxel_down_pcd)
  print("Done!")


def main(csv_path, folder_path, output_file, output_pcd_name):
  timestamps_of_poses, poses, timestamps_of_scans, scan_names = load_data(csv_path, folder_path)
  poses_sync = interpolate_poses(timestamps_of_poses, poses, timestamps_of_scans)
  save_poses(poses_sync, output_file)

  map_cloud = load_and_transform_scans(folder_path, scan_names, poses_sync)
  voxelize_and_save(map_cloud, 0.3, output_pcd_name)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Interpolate poses and save in KITTI format.")
  parser.add_argument('--csv_path', type=str, help='Path to the CSV file.')
  parser.add_argument('--folder_path', type=str, help='Path to the folder containing .pcd files.')
  parser.add_argument('--output_file_path', type=str, help='Output file for the interpolated poses.')
  parser.add_argument('--output_pcd_path', type=str, help='Output file for the accumulated map.')

  args = parser.parse_args()
  main(args.csv_path, args.folder_path, args.output_file_path, args.output_pcd_path)
