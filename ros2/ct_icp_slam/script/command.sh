# ROOT directory of this repository
WORKING_DIR=$(pwd)

EXTERNAL_ROOT=${WORKING_DIR}/cmake-build-Release/external/install/Release
LGMATH_ROOT=${WORKING_DIR}/cmake-build-Release/lgmath/install/Release
STEAM_ROOT=${WORKING_DIR}/cmake-build-Release/steam/install/Release
CT_ICP_ROOT=${WORKING_DIR}/cmake-build-Release/ct_icp/install/Release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LGMATH_ROOT}/lib:${STEAM_ROOT}/lib:${CT_ICP_ROOT}/lib


## First launch RViz for visualization
source /opt/ros/galactic/setup.bash
ros2 run rviz2 rviz2 -d ${WORKING_DIR}/ros2/ct_icp_slam/rviz/slam.rviz # launch rviz

## Run odometry
source ${WORKING_DIR}/ros2/install/setup.bash
ros2 run ct_icp_slam ct_icp_slam --ros-args --params-file ${WORKING_DIR}/ros2/ct_icp_slam/config/default_config.yaml
