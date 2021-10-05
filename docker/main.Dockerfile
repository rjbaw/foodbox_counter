#ARG CUDA_VERSION=10.0
ARG UBUNTU_VERSION=18.04
#FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM amd64/ubuntu:${UBUNTU_VERSION}

#ARG L4T_VERSION=19.12-py3
#FROM nvcr.io/nvidia/pytorch:${L4T_VERSION}

LABEL maintainer="ezvk7740"

RUN mkdir -p /workspace

# Install requried libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
#    libcurl4-openssl-dev \
    software-properties-common \ 
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    apt-utils\
    make\
    g++\
    vim\
    curl    
 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential\
    libssl-dev\
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libtiff-dev\
    libjpeg-dev\
    zlib1g-dev\
    libblas-dev\
    liblapack-dev\
    libblas3\
    liblapack3\
    gfortran\
    libgtk2.0-dev\
    pkg-config\
    qv4l2\
    v4l-utils\
    cmake \
    libatlas-base-dev\
    libgstreamer1.0-0\
    gstreamer1.0-plugins-base\
    gstreamer1.0-plugins-good\
    gstreamer1.0-plugins-bad\
    gstreamer1.0-plugins-ugly\
    gstreamer1.0-libav\
    gstreamer1.0-doc\
    gstreamer1.0-tools\
    gstreamer1.0-x\
    gstreamer1.0-alsa\
    gstreamer1.0-gl\
    gstreamer1.0-gtk3\
    gstreamer1.0-qt5\
    gstreamer1.0-pulseaudio\
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev\
    ffmpeg\
    libhdf5-dev\
    libhdf5-openmpi-dev\
    libavcodec-dev\
    libavformat-dev\
    libswscale-dev

RUN cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip &&\
    pip install -U pip

COPY requirements.txt /tmp
COPY pylon_6.1.1.19861-deb0_amd64.deb /tmp
COPY pylon-supplementary-package-for-mpeg-4_1.0.1.117-deb0_amd64.deb /tmp
COPY pypylon-1.6.0-cp36-cp36m-linux_x86_64.whl /tmp

RUN cd /tmp && \
    pip install -r requirements.txt &&\
    dpkg -i pylon_6.1.1.19861-deb0_amd64.deb &&\
    dpkg -i pylon-supplementary-package-for-mpeg-4_1.0.1.117-deb0_amd64.deb &&\
    pip install pypylon-1.6.0-cp36-cp36m-linux_x86_64.whl\
    pip uninstall opencv-python-headless

RUN cd /tmp &&\
    wget https://github.com/opencv/opencv/archive/4.2.0.tar.gz &&\
    tar -xzf 4.2.0.tar.gz &&\
    cd opencv-4.2.0 &&\
    mkdir build &&\
    cd build &&\
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -DBUILD_opencv_python3=ON -DPYTHON_DEFAULT_EXECUTABLE=$(which python3) -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") -D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") -D WITH_GSTREAMER=ON .. &&\
    make -j$(nproc) &&\
    make install &&\
    ldconfig

RUN apt-get -y autoremove &&\
    apt-get -y autoclean
RUN rm -rf /var/cache/apt
RUN rm -r /tmp/*
WORKDIR /workspace

#RUN chmod u+x /workspace/truck/detect_plate.sh
#ENTRYPOINT /workspace/truck/detect_plate.sh && /bin/bash/jetson_clocks && /bin/bash

RUN ["/bin/bash"]
