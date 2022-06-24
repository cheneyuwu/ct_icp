# ROOT directory of this repository
WORKING_DIR=$(pwd)
WORKING_DIR=${HOME}/ASRL/ct_icp

## Installation
cd ${WORKING_DIR}
bash ct_icp_build.sh Release "Unix Makefiles"

## Add libraries
EXTERNAL_ROOT=${WORKING_DIR}/cmake-build-Release/external/install/Release
LGMATH_ROOT=${WORKING_DIR}/cmake-build-Release/lgmath/install/Release
STEAM_ROOT=${WORKING_DIR}/cmake-build-Release/steam/install/Release
CT_ICP_ROOT=${WORKING_DIR}/cmake-build-Release/ct_icp/install/Release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LGMATH_ROOT}/lib:${STEAM_ROOT}/lib:${CT_ICP_ROOT}/lib

## First launch RViz for visualization
source /opt/ros/galactic/setup.bash
ros2 run rviz2 rviz2 -d ${WORKING_DIR}/ros2/ct_icp_slam/rviz/slam.rviz # launch rviz

################# existing datasets #################
## Run odometry
# - change config file
source ${WORKING_DIR}/ros2/install/setup.bash
ros2 run ct_icp_slam ct_icp_slam --ros-args --params-file ${WORKING_DIR}/ros2/ct_icp_slam/config/kitti_raw_elastic_config.yaml

################# boreas #################
## Run odometry
# - change config file
source ${WORKING_DIR}/ros2/install/setup.bash
ros2 run ct_icp_slam ct_icp_slam --ros-args --params-file ${WORKING_DIR}/ros2/ct_icp_slam/config/boreas_config.yaml
## Evaluate odometry
DATASET_DIR=/home/yuchen/ASRL/data/boreas/sequences
RESULT_DIR=/home/yuchen/ASRL/temp/cticp/boreas/lidar/elastic
source /home/yuchen/ASRL/venv/bin/activate
python generate_boreas_odometry_result.py --dataset ${DATASET_DIR} --path ${RESULT_DIR} --sensor velodyne
python -m pyboreas.eval.odometry --gt ${DATASET_DIR} --pred ${RESULT_DIR}/boreas_odometry_result

################# aeva #################
## Run odometry
# - change config file
source ${WORKING_DIR}/ros2/install/setup.bash
ros2 run ct_icp_slam ct_icp_slam --ros-args --params-file ${WORKING_DIR}/ros2/ct_icp_slam/config/aeva_elastic_config.yaml

## Evaluate odometry
DATASET_DIR=/home/yuchen/ASRL/data/boreas/sequences
RESULT_DIR=/home/yuchen/ASRL/temp/doppler_odometry/boreas/aeva/elastic # change output directory
source /home/yuchen/ASRL/venv/bin/activate
cd /home/yuchen/ASRL/ct_icp/ros2/ct_icp_slam/script
python generate_boreas_odometry_result.py --dataset ${DATASET_DIR} --path ${RESULT_DIR} --sensor aeva
python -m pyboreas.eval.odometry_aeva --gt ${DATASET_DIR} --pred ${RESULT_DIR}/boreas_odometry_result

## Visualize a path
source /opt/ros/galactic/setup.bash
source /home/yuchen/ASRL/venv/bin/activate
cd /home/yuchen/ASRL/ct_icp/ros2/ct_icp_slam/script
python plot_boreas_poses_rviz.py # modify directory inside
