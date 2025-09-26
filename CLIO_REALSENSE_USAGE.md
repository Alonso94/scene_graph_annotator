# CLIO RealSense Docker Usage Guide

This guide shows how to run CLIO with RealSense camera integration in Docker, with support for different environments.

## Quick Start

### 1. Basic Usage (Office Environment)
```bash
# Run CLIO with RealSense camera in office environment
roslaunch clio_annotator clio_realsense.launch
```

### 2. Environment-Specific Usage
```bash
# Office environment (default)
roslaunch clio_annotator clio_realsense.launch environment:=office

# Supermarket environment
roslaunch clio_annotator clio_realsense.launch environment:=supermarket

# Home environment  
roslaunch clio_annotator clio_realsense.launch environment:=home
```

### 3. With AEG Annotation
```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run with AEG annotation (office environment)
roslaunch clio_annotator clio_aeg.launch

# Run with AEG annotation (supermarket environment)
roslaunch clio_annotator clio_aeg.launch environment:=supermarket
```

### 4. Custom Task/Region Files
```bash
# Override with custom task and region files
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=/path/to/custom_tasks.yaml \
    place_tasks_file:=/path/to/custom_regions.yaml
```

## Docker Integration

### Docker Run Command
```bash
# Replace /path/to/workspace with your actual workspace path
docker run -it --rm \
    --privileged \
    --net=host \
    --device=/dev/dri:/dev/dri \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /path/to/workspace:/catkin_ws/src \
    manibot_scene_graph:latest \
    roslaunch clio_annotator clio_realsense.launch environment:=office
```

### With AEG in Docker
```bash
# Set OpenAI API key and run with annotation
docker run -it --rm \
    --privileged \
    --net=host \
    --device=/dev/dri:/dev/dri \
    -e DISPLAY=$DISPLAY \
    -e OPENAI_API_KEY=your_api_key_here \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /path/to/workspace:/catkin_ws/src \
    manibot_scene_graph:latest \
    roslaunch clio_annotator clio_aeg.launch environment:=supermarket
```

## Launch File Parameters

### clio_realsense.launch

#### Camera Parameters
- `camera_name`: Camera namespace (default: "camera")
- `serial_no`: Specific camera serial number (optional)
- `usb_port_id`: Specific USB port (optional)  
- `device_type`: Filter by device type, e.g. "d435" (optional)

#### Environment Configuration  
- `environment`: Environment type - "office", "supermarket", "home" (default: "office")
- `object_tasks_file`: Override default object tasks file (optional)
- `place_tasks_file`: Override default place tasks file (optional)

#### CLIO Configuration
- `robot_id`: Unique robot identifier (default: 0)
- `robot_frame`: Robot base frame (default: camera_link)  
- `dataset_name`: Dataset name for config parsing (default: "realsense")
- `sensor_min_range`: Minimum sensor range in meters (default: 0.3)
- `sensor_max_range`: Maximum sensor range in meters (default: 4.0)

#### Visualization
- `start_visualizer`: Start Hydra visualizer (default: true)
- `start_rviz`: Start RViz (default: true)

### clio_aeg.launch

Inherits all parameters from `clio_realsense.launch` plus:

#### AEG Annotation Parameters
- `llm_model`: OpenAI model to use (default: "gpt-4o-mini")
- `temperature`: LLM temperature parameter (default: 0.8)
- `annotate_every`: Annotate every N scene graph updates (default: 10)

## Environment Files

The launch files automatically use task and region files from the `environments/` directory:

```
environments/
├── office/
│   ├── tasks.yaml      # Office-specific tasks
│   └── regions.yaml    # Office spatial regions
├── supermarket/
│   ├── tasks.yaml      # Shopping/retail tasks  
│   └── regions.yaml    # Store sections
└── home/
    ├── tasks.yaml      # Daily living tasks
    └── regions.yaml    # Home rooms/areas
```

## Output Topics

### Core CLIO Topics
- `/clio_node/backend/dsg`: Scene graph updates
- `/task_server/objects`: Object task embeddings
- `/task_server/places`: Place task embeddings

### RealSense Topics  
- `/camera/color/image_raw`: RGB images
- `/camera/aligned_depth_to_color/image_raw`: Aligned depth
- `/camera/color/camera_info`: Camera calibration

### Visualization
- `/clio_node/visualization/dsg_mesh`: Scene graph mesh for visualization
- `/clio_node/frontend/pose_graph`: Robot poses and trajectory

### With AEG Annotation
- `/annotated_dsg_update`: Enhanced scene graph with LLM annotations

## Troubleshooting

### Camera Not Detected
```bash
# Check if camera is recognized
lsusb | grep Intel

# Check RealSense devices
rs-enumerate-devices
```

### Environment File Not Found
```bash
# Check if environment files exist
ls $(rospack find clio_annotator)/environments/office/

# Create custom environment
mkdir -p $(rospack find clio_annotator)/environments/custom
# Copy and modify existing task/region files
```

### Docker USB Permission Issues
```bash
# Add USB rules for RealSense (run on host)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666"' | sudo tee /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Missing OpenAI API Key
```bash
# Set API key for AEG annotation
export OPENAI_API_KEY=sk-your-api-key-here

# Or add to your .bashrc
echo 'export OPENAI_API_KEY=sk-your-api-key-here' >> ~/.bashrc
```

## Examples

### 1. Office Robot Assistant
```bash
# Run CLIO in office mode for document and supply management
roslaunch clio_annotator clio_aeg.launch \
    environment:=office \
    llm_model:=gpt-4o-mini \
    temperature:=0.5
```

### 2. Retail Store Navigation
```bash  
# Run in supermarket mode for shopping assistance
roslaunch clio_annotator clio_realsense.launch \
    environment:=supermarket \
    sensor_max_range:=6.0
```

### 3. Home Service Robot
```bash
# Run in home mode for household tasks
roslaunch clio_annotator clio_aeg.launch \
    environment:=home \
    annotate_every:=5
```

### 4. Custom Environment
```bash
# Create custom task files and use them
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=/custom/warehouse_tasks.yaml \
    place_tasks_file:=/custom/warehouse_regions.yaml \
    dataset_name:=warehouse
```

## Performance Tips

1. **Sensor Range**: Adjust `sensor_max_range` based on environment size
2. **Annotation Frequency**: Increase `annotate_every` to reduce LLM calls  
3. **Visualization**: Set `start_visualizer:=false` for headless operation
4. **Docker Resources**: Ensure sufficient memory for LLM processing

## Integration with Other Systems

### ROS Navigation Stack
```bash
# Use CLIO's scene graph for navigation planning
rostopic echo /clio_node/backend/dsg
```

### Custom Task Planning
```bash
# Subscribe to annotated updates for high-level reasoning
rostopic echo /annotated_dsg_update
```