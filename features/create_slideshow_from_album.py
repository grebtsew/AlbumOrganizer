"""
Run this file to execute implementation
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils import file
import os
from pyfiglet import Figlet

from enum import Enum


class Level(Enum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3


def generate_slideshow(
    album_path,
    target_path,
    # Color analysis
    color_dominance=(0, 0, 0),  # bgr, check for color dominant occurrences
    color_diversity=Level.LOW,  # high large diversity
    color_warmth=0,  # low = cold, high = warm
    color_intensity=Level.LOW,
    # Image file specific
    min_image_quality=Level.LOW,
    min_image_resolution=(0, 0),
    max_image_resolution=(4000, 4000),
    image_file_formats=[".jpg", ".png", ".jpeg", ".gif"],
    aspect_ratio_range=(0, 2),
    camera_orientation_range=range(1, 8),  # value 1-8 depending on EXIF
    # Text detection
    text_occurrences=Level.LOW,
    # Corner detection
    image_smooth_edges=Level.LOW,
    # Sentient Analysis
    image_feeling="calm",
    # Environment Analysis
    environment="inside",  # "outside"
    # Feature extraction
    sift_features=Level.LOW,
    # Face detection
    people=Level.NONE,
    # Object detection
    allowed_objects=None,  # create a list of strings of object names
    not_allowed_objects=None,
):
    """
    This function loops through an album and finds images depending on a vast amount of filters and creates a slideshow folder with images in target path.
    """
    pass


if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Album Organizer"))

    album_path = "./data/test_images/"
    target_path = "./target/"

    # Slideshow filters

    # Color analysis
    color_dominance = (0, 0, 0)  # bgr, check for color dominant occurrences
    color_diversity = Level.LOW  # high large diversity
    color_warmth = 0  # low = cold, high = warm
    color_intensity = Level.LOW
    # Image file specific
    min_image_quality = Level.LOW
    min_image_resolution = (0, 0)
    max_image_resolution = (4000, 4000)
    image_file_formats = [".jpg", ".png", ".jpeg", ".gif"]
    aspect_ratio_range = (0, 2)
    camera_orientation_range = range(1, 8)  # value 1-8 depending on EXIF
    # Text detection
    text_occurrences = Level.LOW
    # Corner detection
    image_smooth_edges = Level.LOW
    # Sentient Analysis
    image_feeling = "calm"
    # Environment Analysis
    environment = "inside"  # "outside"
    # Feature extraction
    sift_features = Level.LOW
    # Face detection
    people = Level.NONE
    # Object detection
    allowed_objects = None  # create a list of strings of object names
    not_allowed_objects = None

    generate_slideshow(
        album_path,
        target_path,
        color_dominance,
        color_diversity,
        color_warmth,
        color_intensity,
        min_image_quality,
        min_image_resolution,
        max_image_resolution,
        image_file_formats,
        aspect_ratio_range,
        camera_orientation_range,
        text_occurrences,
        image_smooth_edges,
        image_feeling,
        environment,
        sift_features,
        people,
        allowed_objects,
        not_allowed_objects,
    )
