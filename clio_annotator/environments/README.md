# CLIO Environment Configuration Guide

This directory contains pre-configured task and region files for different environments that can be used with CLIO's task-driven scene graph generation.

## Directory Structure

```
environments/
├── office/
│   ├── tasks.yaml      # Object manipulation tasks for office
│   └── regions.yaml    # Spatial regions for office environment
├── supermarket/
│   ├── tasks.yaml      # Shopping and retail tasks
│   └── regions.yaml    # Store sections and areas
├── home/
│   ├── tasks.yaml      # Daily living tasks
│   └── regions.yaml    # Home rooms and functional areas
└── README.md           # This guide
```

## Usage with Launch Files

### Basic Usage
```bash
# For office environment
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=$(find clio_annotator)/environments/office/tasks.yaml \
    place_tasks_file:=$(find clio_annotator)/environments/office/regions.yaml

# For supermarket environment  
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=$(find clio_annotator)/environments/supermarket/tasks.yaml \
    place_tasks_file:=$(find clio_annotator)/environments/supermarket/regions.yaml

# For home environment
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=$(find clio_annotator)/environments/home/tasks.yaml \
    place_tasks_file:=$(find clio_annotator)/environments/home/regions.yaml
```

### With AEG Annotation
```bash
# Example with office environment and AEG annotation
export OPENAI_API_KEY=your_api_key_here
roslaunch clio_annotator clio_aeg.launch \
    object_tasks_file:=$(find clio_annotator)/environments/office/tasks.yaml \
    place_tasks_file:=$(find clio_annotator)/environments/office/regions.yaml
```

## File Format Reference

### Tasks File Format (`tasks.yaml`)

Tasks files contain a YAML list of natural language task descriptions. These should be:
- **Specific**: Clear, actionable instructions
- **Object-focused**: Reference specific objects or object types
- **CLIP-compatible**: Use language that semantic embeddings can understand

**Example structure:**
```yaml
- "find stapler"
- "bring me a pen" 
- "clean the whiteboard"
- "organize papers on desk"
```

**Guidelines for task creation:**
- Use imperative voice ("find X", "bring Y", "clean Z")
- Be specific about objects ("stapler" vs "office supplies")
- Include interaction verbs (find, bring, clean, organize, move)
- Consider the robot's capabilities (manipulation, navigation)

### Regions File Format (`regions.yaml`)

Region files contain a YAML list of spatial/semantic area descriptions for place-based reasoning:

**Example structure:**
```yaml
- "workspace area"
- "meeting room" 
- "reception desk"
- "filing area"
```

**Guidelines for region creation:**
- Focus on functional areas rather than just room names
- Use descriptive spatial terms ("area", "section", "zone")
- Include both large spaces ("conference room") and local areas ("desk surface")
- Consider the environment's workflow and usage patterns

## Creating a New Environment

### Step 1: Create Directory Structure
```bash
mkdir -p environments/your_environment_name
```

### Step 2: Create Tasks File
Create `environments/your_environment_name/tasks.yaml`:

```yaml
# Your Environment Task File
# Add environment-specific tasks focusing on realistic scenarios

- "task 1 description"
- "task 2 description"
# ... add 15-25 tasks
```

### Step 3: Create Regions File  
Create `environments/your_environment_name/regions.yaml`:

```yaml
# Your Environment Region Task File
# Add spatial and functional areas

- "region 1 name"
- "region 2 name"  
# ... add 10-20 regions
```

### Step 4: Test Configuration
```bash
roslaunch clio_annotator clio_realsense.launch \
    object_tasks_file:=$(find clio_annotator)/environments/your_environment_name/tasks.yaml \
    place_tasks_file:=$(find clio_annotator)/environments/your_environment_name/regions.yaml
```

## Environment Examples

### Office Environment
**Focus**: Professional workspace tasks
**Tasks**: Document management, office supplies, meetings
**Regions**: Workspaces, meeting areas, storage

### Supermarket Environment  
**Focus**: Retail and shopping activities
**Tasks**: Product location, restocking, shopping assistance
**Regions**: Store sections, aisles, service areas

### Home Environment
**Focus**: Daily living activities
**Tasks**: Household items, cleaning, personal belongings
**Regions**: Rooms, functional areas, storage spaces

## Best Practices

### Task Design
1. **Realistic scenarios**: Base tasks on actual use cases
2. **Varied complexity**: Mix simple ("find X") and complex ("organize Y") tasks
3. **Environment-specific**: Leverage unique objects in each environment
4. **Action diversity**: Include finding, bringing, cleaning, organizing tasks

### Region Design
1. **Hierarchical thinking**: Include both large areas and specific locations
2. **Functional focus**: Emphasize what happens in each area
3. **Navigation-friendly**: Use terms that help with spatial reasoning
4. **Complete coverage**: Ensure all major areas are represented

### Testing and Iteration
1. **Start small**: Begin with 10-15 tasks and regions, expand based on testing
2. **Monitor CLIP performance**: Some terms work better with semantic embeddings
3. **Real-world validation**: Test with actual camera data in the environment
4. **Iterative refinement**: Adjust based on CLIO's clustering behavior

## Integration with AEG Annotation

When using AEG annotation, the LLM will analyze these tasks and regions to provide enhanced semantic understanding. Consider:

- **Clear object references**: Help the LLM understand object relationships
- **Contextual information**: Provide enough detail for reasoning
- **Consistent terminology**: Use similar language patterns across tasks

## Troubleshooting

### Common Issues
1. **Poor clustering**: Tasks may be too vague or overlapping
2. **Missing objects**: Add more specific object-focused tasks  
3. **Incomplete regions**: Ensure spatial coverage of the environment
4. **CLIP embedding issues**: Some terms may not embed well - test alternatives

### Debugging Tips
- Use `rostopic echo /task_server/objects` to see task embeddings
- Monitor CLIO's clustering output for task-object associations
- Test individual tasks with CLIP models to verify semantic understanding

## Advanced Configuration

### Custom Parameters
You can create environment-specific configuration files for different settings:

```yaml
# environments/your_environment/config.yaml
dataset_name: "your_environment"
sensor_min_range: 0.3
sensor_max_range: 5.0  # Adjust based on environment size
```

### Multi-Environment Usage
For environments that combine multiple settings:

```yaml
# Combined tasks file
- "find office stapler"     # office tasks
- "locate grocery items"    # supermarket tasks  
- "bring household items"   # home tasks
```

This allows CLIO to handle mixed environments or transitional spaces.