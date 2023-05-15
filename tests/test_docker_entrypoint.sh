#!/bin/bash

#
# This file is used to test all modes in docker_entrypoint.sh
#

docker-compose build albumorganizer 
docker-compose run albumorganizer 
docker-compose run albumorganizer --slideshow
docker-compose run albumorganizer --collage
docker-compose run albumorganizer --image-resize
docker-compose run albumorganizer --remove-duplicates
docker-compose run albumorganizer --detect-duplicates
docker-compose run albumorganizer --pytest
