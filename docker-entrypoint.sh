#!/bin/bash

if [ "$1" == "--slideshow" ]; then
  echo "Executing the slideshow Python file..."
  python /app/features/slideshow.py
elif [ "$1" == "--collage" ]; then
  echo "Executing the collage Python file..."
  python /app/features/collage.py
elif [ "$1" == "--imageresize" ]; then
  echo "Executing the Resize Image Python file..."
  python /app/features/resize_all_images.py
elif [ "$1" == "--removeduplicates" ]; then
  echo "Executing the Remove Duplicate Images Python file..."
  python /app/features/remove_duplicate_images.py
elif [ "$1" == "--pytest" ]; then
  echo "Executing pytest..."
  pytest /app/
else
  echo "No argument provided, executing the main Python file..."
  python /app/main.py
fi

imageresize, --removeduplicates