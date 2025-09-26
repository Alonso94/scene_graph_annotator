cmake_minimum_required(VERSION 3.0.2)
project(clio_aeg_annotation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  hydra_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  Annotation.msg  # Custom msg for AEG fields (e.g., fine_grained_category)
  AnnotatedDsgUpdate.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  hydra_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs hydra_msgs message_runtime
)

# Install Python modules
catkin_install_python(PROGRAMS
  src/clio_aeg_annotator.py
  src/clio_aeg_llm.py  # Your AEG agent implementation
  src/general_LLM.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch and config
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY src/base_prompts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/base_prompts
)