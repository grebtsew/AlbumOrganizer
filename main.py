"""
Run this file to execute implementation
"""
from utils import file
from utils import run
from utils import db
import os
from tabulate import tabulate
import shutil
from pyfiglet import Figlet

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Album Organizer"))

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

    # Remove current target
    try:
        target_path = "./target/"
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
    except OSError:
        pass  # This happens in docker container if we do not have permission!

    # Create new target
    file.save_all_individual_from_album(target_path, df, allow_copies=False)

    # TODO: test on larger dataset

