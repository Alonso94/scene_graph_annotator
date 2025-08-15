# Base image of ubuntu 20.04, cuda 12.1 and ZED SDK 4.1
FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu20.04

SHELL ["/bin/bash", "-c"]

# Setup ROS noetic
RUN apt-get update -y || true && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata curl locales lsb-release gnupg2 mesa-utils apt-transport-https && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    echo 'export LIBGL_ALWAYS_INDIRECT=1' >> ~/.bashrc && \
    apt-get update || true && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ros-noetic-desktop-full build-essential cmake usbutils libusb-1.0-0-dev git --allow-unauthenticated

# Install Packages
RUN apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
    python3-catkin-tools python3-vcstool python3-virtualenv

# Initialize rosdep
RUN rosdep init && rosdep update

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Install newer CMake
ARG CMAKE_VER=3.26.4
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-x86_64.tar.gz && \
    tar -xf cmake-${CMAKE_VER}-linux-x86_64.tar.gz -C /opt && \
    ln -sf /opt/cmake-${CMAKE_VER}-linux-x86_64/bin/* /usr/local/bin/ && \
    cmake --version && rm cmake-${CMAKE_VER}-linux-x86_64.tar.gz

# Install GTSAM
RUN apt-get update && apt-get install -y libtbb-dev libboost-all-dev && \
    git clone -b 4.2.0 https://github.com/borglab/gtsam.git /tmp/gtsam && \
    cd /tmp/gtsam && mkdir build && cd build && \
    cmake -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF .. && \
    make -j"$(nproc)" && make install && ldconfig && rm -rf /tmp/gtsam

# Setup ROS workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws
RUN source /opt/ros/noetic/setup.bash && catkin init && \
    catkin config -DCMAKE_BUILD_TYPE=Release --skiplist khronos_eval -DSEMANTIC_INFERENCE_USE_TRT=OFF

# Clone CLIO and its dependencies
WORKDIR /catkin_ws/src
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    git clone https://github.com/MIT-SPARK/Clio.git clio --recursive && \
    vcs import . < clio/install/clio.rosinstall

RUN source /opt/ros/noetic/setup.bash && rosdep install --from-paths . --ignore-src -r -y

RUN sudo apt install -y ros-dev-tools nlohmann-json3-dev python3-dev
RUN rosdep update

# Checkout compatible branches/tags
RUN cd /catkin_ws/src/hydra && git checkout clio
RUN cd /catkin_ws/src/spark_dsg && git checkout clio
RUN cd /catkin_ws/src/config_utilities && git checkout archive/ros_noetic
RUN cd /catkin_ws/src/semantic_inference && git checkout archive/ros_noetic

WORKDIR /catkin_ws
RUN source /opt/ros/noetic/setup.bash && catkin build --cmake-args -Wno-dev

# setting up open-set segmentation
RUN mkdir -p /environments && \
    python3 -m virtualenv --system-site-packages -p /usr/bin/python3 /environments/clio_ros && \
    bash -c "source /environments/clio_ros/bin/activate && \
    pip install /catkin_ws/src/semantic_inference/semantic_inference[openset]"

# setting up Clio python code
RUN python3 -m virtualenv --download -p /usr/bin/python3 /environments/clio && \
    bash -c "source /environments/clio/bin/activate && \
    pip install -e /catkin_ws/src/clio"

# source the environment and set up entrypoint
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "source /environments/clio_ros/bin/activate" >> /root/.bashrc

# Set working directory
WORKDIR /catkin_ws