import os


def find_duplicate_files(folder1, folder2):
  # Get list of files in both folders
  files_in_folder1 = set(os.listdir(folder1))
  files_in_folder2 = set(os.listdir(folder2))

  # Find common files
  duplicate_files = files_in_folder1.intersection(files_in_folder2)

  # Print the duplicate files and their count
  duplicate_count = len(duplicate_files)
  if duplicate_count > 0:
    print(f"Found {duplicate_count} duplicate files:")
    for file in duplicate_files:
      print(file)
  else:
    print("No duplicate files found.")

  # Return the list of duplicate files
  return list(duplicate_files)


if __name__ == "__main__":
  folder1 = '/home/shapelim/git/tmp/ouster_scan'
  folder2 = '/home/shapelim/git/tmp/saved_scans'

  duplicate_files = find_duplicate_files(folder1, folder2)