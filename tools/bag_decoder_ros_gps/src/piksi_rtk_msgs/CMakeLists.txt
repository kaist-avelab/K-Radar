cmake_minimum_required(VERSION 2.8.3)
project(piksi_rtk_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
    rospy
    message_generation
    std_msgs
    geometry_msgs
    sensor_msgs
)

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
    AgeOfCorrections.msg
    BaselineEcef.msg
    BaselineHeading.msg
    BaselineNed.msg
    BasePosEcef.msg
    BasePosLlh.msg
    DeviceMonitor_V2_3_15.msg
    Dops.msg
    DopsMulti.msg
    ExtEvent.msg
    GpsTime.msg
    GpsTimeMulti.msg
    Heartbeat.msg
    ImuAuxMulti.msg
    ImuRawMulti.msg
    InfoWifiCorrections.msg
    Log.msg
    MagRaw.msg
    MeasurementState_V2_4_1.msg
    Observation.msg
    PosEcef.msg
    PosEcefCov.msg
    PositionWithCovariance.msg
    PositionWithCovarianceStamped.msg
    PosLlh.msg
    PosLlhCov.msg
    PosLlhMulti.msg
    ReceiverState.msg
    ReceiverState_V2_2_15.msg
    ReceiverState_V2_3_15.msg
    ReceiverState_V2_4_1.msg
    TrackingState.msg
    TrackingState_V2_2_15.msg
    TrackingState_V2_3_15.msg
    TrackingState_V2_6_3.msg
    UartState.msg
    UartState_V2_3_15.msg
    UtcTimeMulti.msg
    VelEcef.msg
    VelEcefCov.msg
    VelNed.msg
    VelNedCov.msg
    VelocityWithCovariance.msg
    VelocityWithCovarianceStamped.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
    SettingsWrite.srv
    SettingsReadReq.srv
    SettingsReadResp.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS
#  LIBRARIES
  CATKIN_DEPENDS
    message_runtime
    std_msgs
    geometry_msgs
#  DEPENDS
)
