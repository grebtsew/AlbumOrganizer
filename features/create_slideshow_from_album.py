"""
Run this file to execute implementation
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.features import Level
from pyfiglet import Figlet
from utils import features
import shutil

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("Album Organizer"))

    album_path = "./data/test_images/"
    target_path = "./target/"

    # Slideshow filters
    # PLAY WITH FILTERS HERE!

    # Color analysis
    color_dominance = None  # bgr, check for color dominant occurrences
    color_diversity = None  # high large diversity
    color_warmth = None  # low = cold, high = warm
    image_intensity = None
    image_contrast = None
    # Image file specific
    min_image_quality = None
    min_image_resolution = (0, 0)
    max_image_resolution = (4000, 4000)
    image_file_formats = None  # List of formats
    aspect_ratio_range = (0, 2)
    # Text detection
    text_amount = None
    text = None
    # Corner detection
    image_smooth_edges = None
    # Sentient Analysis
    image_feeling = None  # list of feelings
    # Environment Analysis
    environment = None  # "outside"
    # Feature extraction
    sift_features = None
    # Face detection
    people = None
    # Object detection
    allowed_objects = None  # create a list of strings of object names
    not_allowed_objects = None

    workers = 1
    checkpoint_path = "./data/tmp/slideshow_checkpoint.pkl"
    csv_path = "./data/tmp/ss_db.csv"
    checkpoint_interval = 50

    # Remove current target
    try:
        target_path = "./target/"
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
    except OSError:
        pass  # This happens in docker container if we do not have permission!

    features.create_slideshow(
        album_path=album_path,
        target_path=target_path,
        workers=workers,
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
        checkpoint_interval=checkpoint_interval,
        color_dominance=color_dominance,
        color_diversity=color_diversity,
        color_warmth=color_warmth,
        image_intensity=image_intensity,
        image_contrast=image_contrast,
        min_image_quality=min_image_quality,
        min_image_resolution=min_image_resolution,
        max_image_resolution=max_image_resolution,
        image_file_formats=image_file_formats,
        aspect_ratio_range=aspect_ratio_range,
        text_amount=text_amount,
        text=text,
        image_smooth_edges=image_smooth_edges,
        image_feeling=image_feeling,
        environment=environment,
        sift_features=sift_features,
        people=people,
        allowed_objects=allowed_objects,
        not_allowed_objects=not_allowed_objects,
    )
