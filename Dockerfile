FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3.8 python3-pip

# set the working directory
WORKDIR /app

# copy the requirements file into the container
COPY requirements.txt .

# install the Python packages
RUN pip3 install -r requirements.txt

# copy the application code into the container
COPY . .

# expose the port
EXPOSE 8080

# run the application
CMD ["python3", "./src/main.py"]
