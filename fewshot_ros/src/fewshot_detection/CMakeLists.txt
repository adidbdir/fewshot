cmake_minimum_required(VERSION 3.0.2)
project(fewshot_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  BBox.msg
  Pose3D.msg
  FewshotDetection.msg
)

add_service_files(
  FILES
  FewshotDetectionService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/fewshot_service.py
    scripts/test_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 