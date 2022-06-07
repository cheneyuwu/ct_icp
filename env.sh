EXTERNAL_ROOT=$(pwd)/cmake-build-Release/external/install/Release
LGMATH_ROOT=$(pwd)/cmake-build-Release/lgmath/install/Release
STEAM_ROOT=$(pwd)/cmake-build-Release/steam/install/Release
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXTERNAL_ROOT}/Ceres/lib:${EXTERNAL_ROOT}/glog/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LGMATH_ROOT}/lib:${STEAM_ROOT}/lib