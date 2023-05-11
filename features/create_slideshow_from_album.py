"""
Run this file to execute implementation
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.features import Level
from pyfiglet import Figlet
from utils import features

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
    image_intensity = Level.LOW
    image_contrast = Level.LOW
    # Image file specific
    min_image_quality = Level.LOW
    min_image_resolution = (0, 0)
    max_image_resolution = (4000, 4000)
    image_file_formats = [".jpg", ".png", ".jpeg", ".gif"]
    aspect_ratio_range = (0, 2)
    # Text detection
    text = Level.LOW
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

    workers = 8
    checkpoint_path = "./data/tmp/slideshow_checkpoint.pkl"
    csv_path = "./data/tmp/ss_db.csv"
    checkpoint_interval = 50

    features.create_slideshow(
        album_path,
        target_path,
        workers,
        checkpoint_path,
        csv_path,
        checkpoint_interval,
        color_dominance,
        color_diversity,
        color_warmth,
        image_intensity,
        image_contrast,
        min_image_quality,
        min_image_resolution,
        max_image_resolution,
        image_file_formats,
        aspect_ratio_range,
        text,
        image_smooth_edges,
        image_feeling,
        environment,
        sift_features,
        people,
        allowed_objects,
        not_allowed_objects,
    )
