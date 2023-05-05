from . import db
import os
import shutil
import numpy as np

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
    Copies a file from src_file to dest_folder, incrementing the file name if a file with the same name already exists.
    """
    file_name = os.path.basename(src_file)
    dest_file = os.path.join(dest_folder, file_name)
    root, ext = os.path.splitext(file_name)
    i = 0
    while os.path.exists(dest_file):
        i += 1
        dest_file = os.path.join(dest_folder, f"{root}_{i}{ext}")
    return dest_file

def save_individual_images(base_path,df,person,ignore_list=[]):

    print(f"Will create folder {person} in {base_path} and copy images there.")
    folder_path = f"{base_path}{person}"
    os.makedirs(folder_path, exist_ok=True)

    image_list = db.get_all_images_of_individual(df,person)
    for image_path in image_list:
        if image_path in ignore_list:
            continue
        dest_path = get_appropriate_incremental_name(image_path, folder_path)
        shutil.copy(image_path, dest_path)
        print(f"Copied file to {dest_path}")
       
def save_all_individual_from_album(base_path,df, allow_copies=False):
    persons = db.get_all_ids(df)
    #TODO: handle none value too!
    print(f"Will now copy all files into individual folders. allow_copies={allow_copies}")
    ignore_list=[]
    
    for i,person in enumerate(persons):
        print(f"Currently on {i+1}/{len(persons)}")
        save_individual_images(base_path, df,person, ignore_list)
        if not allow_copies: # make sure images are not copied several times
            ignore_list.extend( db.get_all_images_of_individual(df,person))