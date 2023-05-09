#!/bin/bash

#
# This file is used to test all modes in docker_entrypoint.sh
#

docker-compose run albumorganizer 
docker-compose run albumorganizer --slideshow
docker-compose run albumorganizer --collage
docker-compose run albumorganizer --imageresize
docker-compose run albumorganizer --removeduplicates
docker-compose run albumorganizer --pytest
