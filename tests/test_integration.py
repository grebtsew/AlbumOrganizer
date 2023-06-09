import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import run


def test_face_recognition_integration():
    """
    Test the main function and thereby the integration of different functions
    """

    album_path = "./data/test_images/"

    backup_checkpoints = False
    backup_folder = "./data/backups/"

    checkpoint_path1 = "./data/tmp/detect_faces_checkpoint.pkl"
    checkpoint_path2 = "./data/tmp/compare_person_checkpoint.pkl"

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

    assert len(df) == 19  # check amount of faces+images accurate
