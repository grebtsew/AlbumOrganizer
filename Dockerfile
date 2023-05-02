##
# For windows containers!
##

# Specify the Rust base image
FROM rust:latest

# Set the working directory to /app
WORKDIR /app

# Copy the Cargo.toml and Cargo.lock files to the container
COPY Cargo.toml Cargo.lock ./

# Install the project dependencies
RUN cargo build --release

# Copy the project source code to the container
COPY src/ ./src/

# Build the project
RUN cargo build --release

# Set the startup command for the container
CMD ["./target/release/album_organizer"]
