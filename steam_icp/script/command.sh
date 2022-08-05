# ROOT directory of this repository
WORKING_DIR=$(pwd)
WORKING_DIR=${HOME}/ASRL/ct_icp

## Installation
cd ${WORKING_DIR}
bash steam_icp_build.sh

## Add libraries
EXTERNAL_ROOT=${WORKING_DIR}/cmake-build-Release/external/install/Release
LGMATH_ROOT=${WORKING_DIR}/cmake-build-Release/lgmath/install/Release
STEAM_ROOT=${WORKING_DIR}/cmake-build-Release/steam/install/Release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LGMATH_ROOT}/lib:${STEAM_ROOT}/lib

## First launch RViz for visualization
source /opt/ros/galactic/setup.bash
ros2 run rviz2 rviz2 -d ${WORKING_DIR}/steam_icp/rviz/steam_icp.rviz # launch rviz

################# any datasets #################
## Run odometry - change config file
source ${WORKING_DIR}/steam_icp/install/setup.bash
ros2 run steam_icp steam_icp --ros-args --params-file ${WORKING_DIR}/steam_icp/config/aeva_steam_config.yaml

################# boreas #################
## Evaluate odometry
source ${WORKING_DIR}/venv/bin/activate
cd ${WORKING_DIR}/steam_icp/script
DATASET_DIR=${HOME}/ASRL/data/boreas/sequences
RESULT_DIR=${HOME}/ASRL/temp/doppler_odometry/boreas/velodyne/elastic # change output directory
python generate_boreas_odometry_result.py --dataset ${DATASET_DIR} --path ${RESULT_DIR} --sensor velodyne
python -m pyboreas.eval.odometry --gt ${DATASET_DIR} --pred ${RESULT_DIR}/boreas_odometry_result

################# aeva #################
## Evaluate odometry
source ${WORKING_DIR}/venv/bin/activate
cd ${WORKING_DIR}/steam_icp/script
DATASET_DIR=${HOME}/ASRL/data/boreas/sequences
RESULT_DIR=${HOME}/ASRL/temp/doppler_odometry/boreas/aeva/elastic # change output directory
python generate_boreas_odometry_result.py --dataset ${DATASET_DIR} --path ${RESULT_DIR} --sensor aeva
python -m pyboreas.eval.odometry_aeva --gt ${DATASET_DIR} --pred ${RESULT_DIR}/boreas_odometry_result

## Visualize a path
source ${WORKING_DIR}/venv/bin/activate
source /opt/ros/galactic/setup.bash
cd ${WORKING_DIR}/ros2/ct_icp_slam/script
python plot_boreas_poses_rviz.py # modify directory inside
