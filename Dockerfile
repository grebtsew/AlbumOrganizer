# We use ubuntu:20.04 to get python3.8 included
FROM ubuntu:20.04 

ENV DEBIAN_FRONTEND noninteractive

LABEL "maintained" "Grebtsew 23-05-09"

# Install tesseract separately
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng

# Install rest of the programs
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libmagickwand-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/bin/tesseract:${PATH}"

# set the working directory
WORKDIR /app

# copy the requirements file into the container
COPY requirements.txt .

# install the Python packages
RUN pip3 install -r requirements.txt

# copy the application code into the container
COPY . .

RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
