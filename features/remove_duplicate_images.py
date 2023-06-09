"""
Run this file to execute removeal of duplicate images.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils import file
import os
from pyfiglet import Figlet

# This function is tested in test_file.py.

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Album Organizer"))
    print("----- Handle Duplicates -----")

    album_path = "./data/test_images/"

    duplicate_path_tuples = file.find_duplicates(album_path)
    print(f"Found duplicates in album: {len(duplicate_path_tuples)}")
    file.remove_duplicates(duplicate_path_tuples)
