from . import db
import os
import shutil
import numpy as np
import hashlib
from tqdm import tqdm

def find_images(directory):
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(extension in file.lower() for extension in image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def get_appropriate_incremental_name(src_file, dest_folder):
    """
    Find an appropriate filename incrementing the file name if a file with the same name already exists.
    """
    file_name = os.path.basename(src_file)
    dest_file = os.path.join(dest_folder, file_name)
    root, ext = os.path.splitext(file_name)
    i = 0
    while os.path.exists(dest_file):
        i += 1
        dest_file = os.path.join(dest_folder, f"{root}_{i}{ext}")
    return dest_file


def save_individual_images(base_path, df, person, ignore_list=[]):
    print(f"Will create folder {person} in {base_path} and copy images there.")
    folder_path = f"{base_path}{person}"
    os.makedirs(folder_path, exist_ok=True)

    if person is None:
        image_list = db.get_all_images_of_non_individuals(df)
    else:
        image_list = db.get_all_images_of_individual(df, person)

    for image_path in image_list:
        if image_path in ignore_list:
            continue
        dest_path = get_appropriate_incremental_name(image_path, folder_path)
        shutil.copy(image_path, dest_path)
        print(f"Copied file to {dest_path}")


def save_all_individual_from_album(base_path, df, allow_copies=False):
    persons = db.get_all_ids(df)

    print(
        f"Will now copy all files into individual folders. allow_copies={allow_copies}"
    )
    ignore_list = []

    for i,person in tqdm(np.ndenumerate(persons),total=len(persons)):
        try:
            if np.isnan(person):
                person = None
        except Exception:
            pass  # This is bad practice!

        save_individual_images(base_path, df, person, ignore_list)
        if not allow_copies:  # make sure images are not copied several times
            ignore_list.extend(db.get_all_images_of_individual(df, person))


def backup(file_path, folder_path):
    if not os.path.exists(file_path):
        return  # Calculations short so skip backup
    os.makedirs(folder_path, exist_ok=True)
    dest_path = get_appropriate_incremental_name(file_path, folder_path)
    shutil.copy2(file_path, dest_path)
    print(f"Backuped {file_path} to {dest_path}.")

def save_csv(csv_storage_path, df):
    dir_path = os.path.dirname(csv_storage_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    db.save(df, csv_storage_path)

def find_duplicates(rootdir):
    # Create a dictionary to store file hashes and paths
    hash_dict = {}
    duplicates = []
    # Traverse the directory tree recursively
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                # Open the file and calculate its hash
                with open(os.path.join(subdir, file), "rb") as f:
                    hash = hashlib.md5(f.read()).hexdigest()
                # Check if the hash already exists in the dictionary
                if hash in hash_dict:
                    # Print the duplicate file paths
                    print(
                        f"Duplicate images found: {os.path.join(subdir, file)} and {hash_dict[hash]}"
                    )
                    duplicates.append([os.path.join(subdir, file), hash_dict[hash]])
                else:
                    # Add the hash and file path to the dictionary
                    hash_dict[hash] = os.path.join(subdir, file)

    return duplicates


def remove_duplicates(duplicates):
    """
    Remove the first value of the detected duplicate array
    """
    for d in duplicates:
        os.remove(d[0])
        print(f"Removed duplicate {d[0]}!")
