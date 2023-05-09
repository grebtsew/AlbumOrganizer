"""
Run this file to execute implementation
"""
from utils import file
from utils import run
import os
from tabulate import tabulate
import shutil
from pyfiglet import Figlet

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Album Organizer"))

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

    # Remove current target
    target_path = "./target/"
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    # Create new target
    file.save_all_individual_from_album(target_path,df, allow_copies=False) 

    # TODO: test on larger dataset
    # TODO: test on docker