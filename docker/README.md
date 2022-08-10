# Docker support for rl-baseline3-zoo DGX

## Build docker image

This custom image installs:
 
- all dependencies installed via apt-get
- stable-baseline3

Build command:

	docker build -t dgx-cpg-locomotion:latest \
		-f docker/Dockerfile . 

Note: 

* Build from cpg-locomotion dir so that docker can copy this package into the image.

## Make docker container 

Nvidia Gpu:

	docker run -it --privileged --net=host \
         --name=cpg-locomotion \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         --runtime=nvidia \
         --gpus all \
         dgx-cpg-locomotion:latest \
         bash

## Running container

Run,

    ./docker/run.sh cpg-locomotion

Afterwards, cd to ```/root/cpg-locomotion```. You can now run ```train.py``` according to the instructions in top-level README.md

## Test GUI

Run in container,

    apt install -y xorg
    xclock

## Remove container

Run,

	docker rm -f cpg-locomotion
