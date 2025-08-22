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

Create a docker image for Manibot scene graph:
```bash
docker build -t clio .
```

Create a container from the image:
```bash
docker_run_nvidia --name=clio clio
docker_run_usb --name=clio-realsense clio-realsense
```

Connect with a running container:
```bash
docker start clio
docker exec -it clio bash
docker start clio-realsense
docker exec -it clio-realsense bash
```
or attach Visual Studio Code to the container.

Run 
```
roslaunch clio_ros realsense.launch
```