import os
import pandas as pd
import argparse

def main(csv_path, folder_path):
  # Load the data into a DataFrame
  data = pd.read_csv(csv_path, delimiter=',')

  # Ensure the 'sec' and 'nsec' columns are treated as integers
  data['sec'] = data['sec'].astype(int)
  data['nsec'] = data['nsec'].astype(int)

  # List all files in the folder
  files_in_folder = os.listdir(folder_path)
  files_in_folder = sorted(files_in_folder)

  # Extract the base names of the files to easily check against
  files_in_folder_basenames = [os.path.basename(file) for file in files_in_folder]

  # Check if corresponding pose files exist
  missing_files = []
  existing_files = []

  for i in range(len(data)):
    sec = data['sec'][i]
    nsec = data['nsec'][i]
    # Ensure nsec is padded to 9 digits
    nsec_padded = str(nsec).zfill(9)
    file_name = f"cloud_{sec}_{nsec_padded}.pcd"

    if file_name in files_in_folder_basenames:
      existing_files.append(file_name)
    else:
      missing_files.append(file_name)

  # Output results
  print(f"Existing files ({len(existing_files)})")
  # for file in existing_files:
  #   print(file)

  print(f"\nMissing files ({len(missing_files)})")
  # for file in missing_files:
  #   print(file)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Check for corresponding pose files in a directory.")
  parser.add_argument('--csv_path', type=str, help='Path to the CSV file.')
  parser.add_argument('--folder_path', type=str, help='Path to the folder containing .pcd files.')

  args = parser.parse_args()
  main(args.csv_path, args.folder_path)