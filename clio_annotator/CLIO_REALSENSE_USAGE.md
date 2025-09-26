# CLIO RealSense Launch Files Usage

This package provides two complementary launch files for running CLIO with RealSense cameras:

1. **`clio_realsense.launch`** - Core RealSense + CLIO integration
2. **`clio_aeg.launch`** - Adds AEG (LLM) annotation on top of the core functionality

## Architecture

- **`clio_realsense.launch`**: Handles RealSense camera + CLIO scene graph processing
- **`clio_aeg.launch`**: Includes `clio_realsense.launch` + adds AEG annotation layer

## Core Functionality (clio_realsense.launch)

### Features
- **Real-time RealSense camera integration**: Automatically launches and configures RealSense camera
- **CLIO scene graph processing**: Full CLIO pipeline with semantic inference and segmentation
- **Task server integration**: Supports object and place tasks from YAML files
- **Visualization**: Integrated RViz and Hydra visualizer

### Basic Usage

#### Simple Launch (Default Configuration)

```bash
roslaunch clio_annotator clio_realsense.launch
```

#### With Custom Tasks

```bash
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=/path/to/your/tasks.yaml \
    place_tasks_file:=/path/to/your/regions.yaml
```

#### With Specific Camera Serial Number

```bash
roslaunch clio_annotator clio_realsense.launch \
    serial_no:=_831612073525
```

## AEG Annotation Functionality (clio_aeg.launch)

### Features
- **All clio_realsense.launch functionality**
- **LLM-based scene graph annotation**: Uses OpenAI GPT models for semantic enhancement
- **Automated annotation**: Processes scene graph updates and adds semantic information

### Usage

#### Basic AEG Launch

```bash
# Make sure to set OpenAI API key first
export OPENAI_API_KEY=your_api_key_here
roslaunch clio_annotator clio_aeg.launch \
    object_tasks_file:=/catkin_ws/src/tasks.yaml \
    place_tasks_file:=/catkin_ws/src/regions.yaml
```

#### With Custom LLM Parameters

```bash
export OPENAI_API_KEY=your_api_key_here
roslaunch clio_annotator clio_aeg.launch \
    object_tasks_file:=/catkin_ws/src/tasks.yaml \
    place_tasks_file:=/catkin_ws/src/regions.yaml \
    llm_model:=gpt-4o \
    temperature:=0.5 \
    annotate_every:=5
```

## Key Parameters

### Camera Configuration
- `camera_name` (default: "camera"): Name of the camera node
- `serial_no` (default: ""): Camera serial number (use underscore prefix for numbers)
- `device_type` (default: ""): Device type filter (e.g., "d435")
- `enable_sync` (default: true): Synchronize camera frames
- `align_depth` (default: true): Align depth to color frames
- `enable_pointcloud` (default: true): Generate pointcloud

### CLIO Configuration
- `dataset_name` (default: "realsense"): Dataset configuration name
- `sensor_min_range` (default: 0.3): Minimum sensor range in meters
- `sensor_max_range` (default: 4.0): Maximum sensor range in meters
- `robot_frame` (default: "camera_link"): Robot base frame
- `map_frame` (default: "world"): Map reference frame

### Task Configuration
- `object_tasks_file` (default: "/catkin_ws/src/tasks.yaml"): Path to object tasks file
- `place_tasks_file` (default: "/catkin_ws/src/regions.yaml"): Path to place tasks file

### AEG Annotation
- `run_aeg_annotation` (default: false): Enable LLM-based annotation
- Make sure to set `OPENAI_API_KEY` environment variable if using AEG annotation

### Visualization
- `start_visualizer` (default: true): Start Hydra visualizer
- `start_rviz` (default: true): Start RViz

## Docker Usage

When running in Docker container, use:

```bash
# Start the Docker container with USB support
docker_run_usb --name=clio-realsense clio

# Inside the container
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=/catkin_ws/src/tasks.yaml \
    place_tasks_file:=/catkin_ws/src/regions.yaml
```

## Prerequisites

1. **RealSense camera connected** via USB
2. **Task files prepared**: Create `tasks.yaml` and `regions.yaml` files
3. **OpenAI API key** (if using AEG annotation): `export OPENAI_API_KEY=your_api_key`

## Differences from Manual Commands

This launch file **replaces** the manual sequence:
```bash
# OLD WAY (manual commands):
roslaunch realsense2_camera rs_camera.launch align_depth:=true enable_sync:=true publish_tf:=true pointcloud:=true
roslaunch clio_ros realsense.launch object_tasks_file:=/catkin_ws/src/tasks.yaml place_tasks_file:=/catkin_ws/src/regions.yaml
```

With a **single command**:
```bash
# NEW WAY (single command):
roslaunch clio_annotator clio_realsense.launch object_tasks_file:=/catkin_ws/src/tasks.yaml place_tasks_file:=/catkin_ws/src/regions.yaml
```

## Architecture

The launch file:
1. **Includes** `realsense2_camera/launch/rs_camera.launch` for camera driver
2. **Includes** `clio_ros/launch/realsense.launch` for CLIO processing
3. **Remaps** topics to connect RealSense outputs to CLIO inputs
4. **Optionally** launches AEG annotation node
5. **Handles** all visualization components

This approach avoids code duplication and leverages the existing, tested launch files from both packages.