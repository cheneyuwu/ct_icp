FROM ubuntu:20.04

CMD ["/bin/bash"]

# Args for setting up non-root users, example command to use your own user:
# docker build -t ct_icp \
#   --build-arg USERID=$(id -u) \
#   --build-arg GROUPID=$(id -g) \
#   --build-arg USERNAME=$(whoami) \
#   --build-arg HOMEDIR=${HOME} .
ARG GROUPID=0
ARG USERID=0
ARG USERNAME=root
ARG HOMEDIR=/root

RUN if [ ${GROUPID} -ne 0 ]; then addgroup --gid ${GROUPID} ${USERNAME}; fi \
  && if [ ${USERID} -ne 0 ]; then adduser --disabled-password --gecos '' --uid ${USERID} --gid ${GROUPID} ${USERNAME}; fi

ENV DEBIAN_FRONTEND=noninteractive

## Switch to specified user to create directories
USER ${USERID}:${GROUPID}

## Switch to root to install dependencies
USER 0:0

## Dependencies
RUN apt update && apt upgrade -q -y
RUN apt update && apt install -q -y cmake git build-essential
RUN apt update && apt install -q -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
RUN apt update && apt install -q -y freeglut3-dev
RUN apt update && apt install -q -y python3 python3-distutils python3-pip
RUN pip3 install pybind11

## Switch to specified user
USER ${USERID}:${GROUPID}

## run the container (example command)
# docker run -it --name ct_icp \
#   --privileged \
#   --network=host \
#   --gpus all \
#   -e DISPLAY=$DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix \
#   -v ${HOME}:${HOME}:rw \
#   -v ${HOME}/ASRL:${HOME}/ASRL:rw \
#   -v /media/yuchen/T7/ASRL/data/KITTI_raw:${HOME}/ASRL/data/KITTI_raw \
#   ct_icp