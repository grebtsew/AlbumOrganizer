from . import db
import os
import shutil
import numpy as np
import hashlib
from tqdm import tqdm
import cv2

from enum import Enum


class ResizeMode(Enum):
    SCALE = 1
    CROP = 2
    RESIZE = 3


def transform_images_size(
    album_path,
    target_path,
    width=640,
    height=480,
    fx=0.5,
    fy=0.5,
    mode=ResizeMode.RESIZE,
):
    """
    Resize, rescale or crop images, returns target folder
    """
    target = f"{target_path}"
    dest = get_appropriate_incremental_name("resize", target)
    os.makedirs(dest)

    print("----- Album Resizing -----")
    print(f"Created folder {dest} for image resizing results.")
    print(f"Mode {mode}")
    print(f"Width : {width}, Height : {height}, Fx : {fx}, Fy : {fy}")

    all_image_paths = find_images(album_path)
    for image_path in tqdm(all_image_paths, total=len(all_image_paths)):
        img = cv2.imread(image_path)

        if mode == ResizeMode.SCALE:
            # might be interesting to update interpolation for some occasions
            resized_img = cv2.resize(
                img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR
            )
        elif mode == ResizeMode.CROP:
            h, w = img.shape[:2]
            startx = w // 2 - (width // 2)
            starty = h // 2 - (height // 2)
            endx = startx + width
            endy = starty + height
            resized_img = img[starty:endy, startx:endx]
        elif mode == ResizeMode.RESIZE:
            resized_img = cv2.resize(img, (width, height))
        else:
            print("WARNING: Unsupported transform mode used!")
            return

        dest_path = get_appropriate_incremental_name(image_path, dest)
        cv2.imwrite(dest_path, resized_img)
    return dest


def find_images(directory):
    """
    Find all images in subdirectories.
    Returns a list of paths to each image.
    """
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
    """
    Saves images from album in a folder, renames images if names collide.
    """
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
    """
    Performs the copying and so on for the main function.
    """
    persons = db.get_all_ids(df)

    print(
        f"Will now copy all files into individual folders. allow_copies={allow_copies}"
    )
    ignore_list = []

    for i, person in tqdm(np.ndenumerate(persons), total=len(persons)):
        try:
            if np.isnan(person):
                person = None
        except Exception:
            pass  # This is bad practice!

        save_individual_images(base_path, df, person, ignore_list)
        if not allow_copies:  # make sure images are not copied several times
            ignore_list.extend(db.get_all_images_of_individual(df, person))


def backup(file_path, folder_path):
    """
    Store file in backup folder, for potential later use.
    """
    if not os.path.exists(file_path):
        return  # Calculations short, no checkpoint created, so skip backup
    os.makedirs(folder_path, exist_ok=True)
    dest_path = get_appropriate_incremental_name(file_path, folder_path)
    shutil.copy2(file_path, dest_path)
    print(f"Backuped {file_path} to {dest_path}.")


def save_csv(csv_storage_path, df):
    """
    Saves a pandas dataframe to file.
    """
    dir_path = os.path.dirname(csv_storage_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    db.save(df, csv_storage_path)


def find_duplicates(rootdir):
    """
    Find image duplicates in subdirectories using file encodings.
    """
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
