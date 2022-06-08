# ROOT directory of this repository
EXTERNAL_ROOT=$(pwd)/cmake-build-Release/external/install/Release
LGMATH_ROOT=$(pwd)/cmake-build-Release/lgmath/install/Release
STEAM_ROOT=$(pwd)/cmake-build-Release/steam/install/Release
CT_ICP_ROOT=$(pwd)/cmake-build-Release/ct_icp/install/Release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LGMATH_ROOT}/lib:${STEAM_ROOT}/lib:${CT_ICP_ROOT}/lib

source $(pwd)/ros2/install/setup.bash

ros2 run ct_icp_slam ct_icp_slam --ros-args --params-file $(pwd)/ros2/ct_icp_slam/config/default_config.yaml
