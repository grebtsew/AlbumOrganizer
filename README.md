# AlbumOrganizer

AlbumOrganizer is a Python3 tool that organizes your photo albums by recognizing persons in your photos and creating a file structure where photos of each individual are moved. The implementation is dockerized and uses pytest. Extra features include generation of face collages per person and sentient slideshows.

# Installation 

## Docker

To install AlbumOrganizer using Docker, you need to have Docker installed on your system. You can then build and run the container using the docker-compose file using the following command:

```bash
docker-compose run albumorganizer
```

## Locally

To install AlbumOrganizer locally, follow the steps in the DockerFile. Python3.8 is recommended as it was used during development. Run the following command to install python packages:

```bash
pip3 install -r ./requirements.txt
```


# Usage

Replace album_path with the path to your photo album. AlbumOrganizer will scan your photos, recognize persons, and create a file structure where photos of each individual are moved to a separate folder. The general execution pipeline includes the following steps:
1. Copy photo album to the `./data` folder for docker volume. (Optional). 
2. Update album root path and settings in `./main.py`.
3. Run the implementation by executing `./main.py`.
4. All results will be stored in `./target` folder.

Using docker the implementation, simply execute the following command in your terminal:

```bash
docker-compose run albumorganizer
```
Locally execute the `./main.py` file by running the following command:
```
python3 ./main.py
```
Additional features can be found in the `./features` folder.

# Extra Features

These features can be found in the `./features` folder. Change settings the files before running.
## Face Collages

AlbumOrganizer can also generate face collages per person. To generate face collages, simply add the --collage flag to the command:

```bash
docker-compose run albumorganizer --collage
```

AlbumOrganizer will then generate a face collage for each individual in the photo album.
## Sentient Slideshows

AlbumOrganizer can also create sentient slideshows. To create a sentient slideshow, simply add the --slideshow flag to the command:

```bash
docker-compose run albumorganizer --slideshow
```

AlbumOrganizer will then create a sentient slideshow using the photos in the photo album.

## Resize All Images

AlbumOrganizer can also resize all images in an album, either crop, resize or rescale. To resize the dataset, simply add the --imageresize flag to the command:

```bash
docker-compose run albumorganizer --imageresize
```

AlbumOrganizer will then create a new folder in target containing the resized images.

## Remove Duplicate Images

AlbumOrganizer can also detect and remove duplicate images in albums. To remove duplicates, simply add the --removeduplicates flag to the command:

```bash
docker-compose run albumorganizer --removeduplicates
```

AlbumOrganizer will then create a sentient slideshow using the photos in the photo album.
# Testing

AlbumOrganizer comes with a set of tests that can be run using pytest. To run the tests, execute the following command in your terminal:

```arduino
docker run -it albumorganizer:latest pytest
```

# Contributing

If you would like to contribute to AlbumOrganizer, please fork the repository and create a pull request. We welcome all contributions, big or small!

# Disclamer
Some of the code in this implementation is generated using ChatGPT4.

# License

AlbumOrganizer is released under the MIT License.


#TODO: add demos!