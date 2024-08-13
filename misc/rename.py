import os

def rename_files(directory, substring_to_remove):
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the substring is in the filename
        if substring_to_remove in filename:
            # Construct the new filename by removing the substring
            new_filename = filename.replace(substring_to_remove, "")
            # Full paths for old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed "{filename}" to "{new_filename}"')

# Example usage
directory = '/data/malaria_proj/building_data/unification/models/images'
substring_to_remove = 'RGB-PanSharpen_'
rename_files(directory, substring_to_remove)
