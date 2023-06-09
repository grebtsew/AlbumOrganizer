"""
Run this file to execute resizing of images.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils import file
import os
from pyfiglet import Figlet


if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Album Organizer"))

    album_path = "./data/test_images/"
    target_path = "./target/"

    # Wanted dims or scale
    width = 640
    height = 480
    fx = 0.5
    fy = 0.5

    # Perform three kinds of resizing
    file.transform_images_size(
        album_path, target_path, width, height, fx, fy, file.ResizeMode.CROP
    )
    file.transform_images_size(
        album_path, target_path, width, height, fx, fy, file.ResizeMode.RESIZE
    )
    file.transform_images_size(
        album_path, target_path, width, height, fx, fy, file.ResizeMode.SCALE
    )
