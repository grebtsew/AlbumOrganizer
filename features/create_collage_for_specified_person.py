"""
Run this file to execute face collage generation
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
from utils import db
from utils import file
from utils import run
from utils import features
import os
from pyfiglet import Figlet


if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Album Organizer"))

    album_path = "./data/test_images/"
    target_path = "./target/"

    # Path to the album root folder
    album_path = "./data/test_images/"

    """
    Backup checkpoints, they can later be used to skip part of the calculation, like caching.
    """
    backup_checkpoints = True
    backup_folder = "./data/backups/"

    """
    Backup the Dataframe, so that it won't accidentally be replaced during next run.
    """
    backup_csv = True
    csv_storage_path = "./data/tmp/fr_db.csv"

    """
    Load a Dataframe, instead of performing the calculation.
    WARNING: it is important that this dataframe csv file has correct format!
    """
    load_df = False
    df_path = "./data/tmp/fr_db.csv"

    if not load_df:
        """
        Storage paths for checkpoints.
        """
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

    else:
        df = db.load_dataframe(df_path)

    # Pretty print (DataFrame):
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    print(df)

    collage = features.create_face_collage(df, ["Unknown11"], target_path, (1920, 1080))
