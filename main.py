"""
Run this file to execute implementation
"""
from utils import ai
from utils import db
from utils import file
from utils import run
import os
from tabulate import tabulate


if __name__ == "__main__":
    # TODO: load df mode
    # TODO: create a user interact script
    # TODO: move code to tests

    album_path = "./data/test_images/"

    backup_checkpoints = True
    backup_folder = "./data/backups/"

    backup_csv = True
    csv_storage_path = "./data/tmp/fr_db.csv"

    checkpoint_path1 = "./data/tmp/detect_faces_checkpoint.pkl"
    checkpoint_path2 = "./data/tmp/compare_person_checkpoint.pkl"

    # Perform calculations to create dataset of all images and
    #  faces recognized and compared to personal ids
    # These calculations are performed with checkpointing!
    df = run.face_recognition_on_album(
        album_path,
        workers=8,
        tolerance=0.6,
        checkpoint_interval=100,
        backup_checkpoints=backup_checkpoints,
        backup_folder=backup_folder,
        checkpoint_path1=checkpoint_path1,
        checkpoint_path2=checkpoint_path2,
    )

    # Save to CSV file
    file.save_csv(csv_storage_path, df)

    if backup_csv:
        file.backup(csv_storage_path, backup_folder)


    # Pretty print (DataFrame):
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    print(df)

    # TODO: test on larger dataset

    # TODO: write tests
    # TODO: create test ci/cd
    # TODO: add gui

    # TODO: crop all faces of individual and create large coolage
    # TODO: extra feature: generate gallery slideshow depending on fun settings like color, warmth, peoples or not, resolution, size, format, features, intensity, new/old, order, scenario/inside, text_recognition, aspect ratio, orientation, sharp/smooth edges, animal and object detection
