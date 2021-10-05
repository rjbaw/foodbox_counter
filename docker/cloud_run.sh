#!/bin/sh

HOST_IP=`hostname -I | awk '{print $1}'`
REPOSITORY='ezvk7740/nongjok'
IMAGE_TYPE='nongjok'
CODE_NAME='20.04'
TAG='v1'

# setup pulseaudio cookie
if [ x"$(pax11publish -d)" = x ]; then
    start-pulseaudio-x11;
    echo `pax11publish -d | grep --color=never -Po '(?<=^Cookie: ).*'`
fi

# run container
xhost +local:root
docker run -it \
  -e DISPLAY=$DISPLAY \
  -e PULSE_SERVER=tcp:$HOST_IP:4713 \
  -e PULSE_COOKIE_DATA=`pax11publish -d | grep --color=never -Po '(?<=^Cookie: ).*'` \
  -e QT_GRAPHICSSYSTEM=native \
  -e QT_X11_NO_MITSHM=1 \
  -v /dev/shm:/dev/shm \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority:rw\
  -v /var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket:ro \
  -v ${XDG_RUNTIME_DIR}/pulse/native:/run/user/1000/pulse/native \
  -v ~/:/workspace/home \
  --privileged \
  --gpus all \
  --ipc=host \
  --net=host \
  --cap-add=NET_ADMIN\
  --cap-add=NET_RAW\
  --rm \
  --name ${IMAGE_TYPE}-${CODE_NAME}-${TAG} \
  ${REPOSITORY}:${TAG}
xhost -local:root

#  --device=/dev/video0:/dev/video0 \
#  --cgroupns "host" \
#  --cgroupns private \
#  -v /dev/bus/usb:/dev/bus/usb \
#  -v "/etc/group:/etc/group:ro" \
#  -v "/etc/passwd:/etc/passwd:ro" \
#  -v "/etc/shadow:/etc/shadow:ro" \
#  -v "/etc/sudoers.d:/etc/sudoers.d:ro" \

#  -v /usr/local/cuda:/usr/local/cuda\
#  -v "/home/$USER/:/home/$USER/" \
#  -e USER=$USER \
#  -v /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib64 \
#  -v /usr/local/cuda/lib64:/usr/local/cuda/lib64 \
#  --device=/dev/sda\
#  --add-host=docker:10.154.148.1\
#  -v /usr/local/lib/openmpi:/usr/local/lib/openmpi\
#  --device=/dev/sdb1\
#  -v /usr/lib:/usr/lib\
#  --user=$(id -u) \
#  --workdir=/home/$USER \
#  -v "/mnt/sdb:/mnt/sdb" \
#  -v "/mnt/sdb:/mnt/sda" \
