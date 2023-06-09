#!/bin/bash

if [ "$1" == "--slideshow" ]; then
  echo "Executing the slideshow Python file..."
  python3 /app/features/create_slideshow_from_album.py
elif [ "$1" == "--collage" ]; then
  echo "Executing the collage Python file..."
  python3 /app/features/create_collage_for_specified_person.py
elif [ "$1" == "--image-resize" ]; then
  echo "Executing the Resize Image Python file..."
  python3 /app/features/resize_all_images.py
elif [ "$1" == "--remove-duplicates" ]; then
  echo "Executing the Remove Duplicate Images Python file..."
  python3 /app/features/remove_duplicate_images.py
elif [ "$1" == "--detect-duplicates" ]; then
  echo "Executing the Detect Duplicate Images Python file..."
  python3 /app/features/detect_duplicate_images.py
elif [ "$1" == "--pytest" ]; then
  echo "Executing pytest..."
  pytest /app/tests
else
  echo "No argument provided, executing the main Python file..."
  python3 /app/main.py
fi
