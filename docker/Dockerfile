# Dockerfile to build custom image for rl-stablebaseline3 in DGX

FROM nvcr.io/nvidia/pytorch:21.08-py3

ARG DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -q -y --no-install-recommends \
    sudo \
    swig \
    cmake \
    ffmpeg 

# Force Install of missing dependencies
RUN apt-get -y -f install

# copy files to container
COPY . /root/cpg-locomotion
RUN pip install -r /root/cpg-locomotion/requirements.txt --ignore-installed

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["bash"]