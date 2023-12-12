cmake_minimum_required(VERSION 2.8.3)
project(rb5_control)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  serial
  tf
  rospy
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs 
  roscpp 
  serial
  tf
  rospy 
  std_msgs
  sensor_msgs
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/rb5_control
  ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/rb5_control_node.cpp)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME}_node
    #ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    #LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
