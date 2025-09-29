# Manibot_scene_graph

Install Docker from here: [https://docs.docker.com/engine/install/ubuntu](https://docs.docker.com/engine/install/ubuntu)

To use docker without sudo follow the instruction here: [https://docs.docker.com/engine/install/linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall)

And run :
```bash
newgrp docker
```

Enable X11 forwarding
```bash
xhost +
```

Create aliases for running docker with and without Nvidia:
```bash
bash create_aliases.sh
```

Install Nvidia container toolkit: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Run sample CUDA container:
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

If you get an error `Failed to initialize NVML: Unknown Error in Docker`, follow the instructions here: 
1. `sudo nano /etc/nvidia-container-runtime/config.toml`, then change `no-cgroups = false`, save

2. Restart docker daemon: `sudo systemctl restart docker`, then you can test by running `sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`

Pull the image from DockerHub:
```bash
docker pull 3liyounes/clio-realsense:latest
```
Create a container from the image:
```bash
docker_run_usb --name=clio 3liyounes/clio-realsense:latest
```

OR

Create a docker image and container for the docker file:
```bash
docker build -t clio .
docker_run_usb --name=clio clio
```

Connect with a running container:
```bash
docker start clio
docker exec -it clio bash
```
or attach Visual Studio Code to the container.

Clone this package into `/catkin_ws/src` in the container and build the workspace:
```bash
cd /catkin_ws/src
git clone https://github.com/Alonso94/scene_graph_annotator.git
cd /catkin_ws
catkin build
source devel/setup.bash
```
Run 
```bash
roslaunch clio_annotator clio_realsense.launch
```

If rviz does not open, try on the host machine:
```bash
xhost +
```

If the semantic inference models are not downloaded:
```bash
cd /catkin_ws/src/scene_graph_annotator/clio_annotator/scripts/semantic_inference
bash download_models.sh
```
In case you had a problem with realsense camera:
```bash
apt update && apt install -y ros-noetic-realsense2-camera ros-noetic-realsense2-description
```