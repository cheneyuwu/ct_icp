#!/bin/bash

BUILD_TYPE=$1
GENERATOR=$2
WITH_PYTHON_BINDING=$3
WITH_VIZ=$4

if [ -z "$BUILD_TYPE" ]
then
	BUILD_TYPE="Release"
fi

if [ -z "$GENERATOR" ]
then
	GENERATOR="Unix Makefiles"
fi

if [ -z "$WITH_PYTHON_BINDING" ]
then
	WITH_PYTHON_BINDING=OFF
fi

if [ -z "$WITH_VIZ" ]
then
	WITH_VIZ=OFF
fi


# Setting variables
SRC_DIR=$(pwd)
EXT_SRC_DIR="${SRC_DIR}/external"
LGMATH_SRC_DIR="${SRC_DIR}/lgmath"
STEAM_SRC_DIR="${SRC_DIR}/steam"
CT_ICP_SRC_DIR="${SRC_DIR}/src"

BUILD_DIR="${SRC_DIR}/cmake-build-${BUILD_TYPE}"
EXT_BUILD_DIR=$BUILD_DIR/external
LGMATH_BUILD_DIR=$BUILD_DIR/lgmath
STEAM_BUILD_DIR=$BUILD_DIR/steam
CT_ICP_BUILD_DIR=$BUILD_DIR/ct_icp

mkdir -p $BUILD_DIR
mkdir -p $EXT_BUILD_DIR
mkdir -p $LGMATH_BUILD_DIR
mkdir -p $STEAM_BUILD_DIR
mkdir -p $CT_ICP_BUILD_DIR

check_status_code() {
   if [ $1 -ne 0 ]
   then
	echo "[CT_ICP] Failure. Exiting."
	exit 1
   fi
}

echo "[CT_ICP] -- [EXTERNAL DEPENDENCIES] -- Generating the cmake project"
cd ${EXT_BUILD_DIR}
cmake -G "$GENERATOR" -S $EXT_SRC_DIR -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DWITH_VIZ3D=$WITH_VIZ
check_status_code $?

echo "[CT_ICP] -- [EXTERNAL DEPENDENCIES] -- building CMake Project"
cmake --build . --config $BUILD_TYPE
check_status_code $?

echo "[CT_ICP] -- [LGMATH] -- Generating the cmake project"
cd ${LGMATH_BUILD_DIR}
cmake -G "$GENERATOR" -S $LGMATH_SRC_DIR \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DUSE_AMENT=OFF \
	-DEigen3_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/Eigen3/share/eigen3/cmake \
	-DCMAKE_INSTALL_PREFIX=${LGMATH_BUILD_DIR}/install/${BUILD_TYPE}
check_status_code $?

echo "[CT_ICP] -- [LGMATH] -- building CMake Project"
cmake --build . --config $BUILD_TYPE --target install --parallel 6
check_status_code $?

echo "[CT_ICP] -- [STEAM] -- Generating the cmake project"
cd ${STEAM_BUILD_DIR}
cmake -G "$GENERATOR" -S $STEAM_SRC_DIR \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DUSE_AMENT=OFF \
	-DEigen3_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/Eigen3/share/eigen3/cmake \
	-Dlgmath_DIR=${LGMATH_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/lgmath \
	-DCMAKE_INSTALL_PREFIX=${STEAM_BUILD_DIR}/install/${BUILD_TYPE}
check_status_code $?

echo "[CT_ICP] -- [STEAM] -- building CMake Project"
cmake --build . --config $BUILD_TYPE --target install --parallel 6
check_status_code $?

echo "[CT_ICP] -- [CT_ICP] -- Generating the cmake project"
cd ${CT_ICP_BUILD_DIR}
cmake -G "$GENERATOR" -S $CT_ICP_SRC_DIR \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DWITH_VIZ3D=$WITH_VIZ \
 	-DWITH_PYTHON_BINDING=${WITH_PYTHON_BINDING} \
	-DEigen3_DIR=${EXT_BUILD_DIR}/install/${BUILD_TYPE}/Eigen3/share/eigen3/cmake \
	-Dlgmath_DIR=${LGMATH_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/lgmath \
	-Dsteam_DIR=${STEAM_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/steam \
	-DCMAKE_INSTALL_PREFIX=${CT_ICP_BUILD_DIR}/install/${BUILD_TYPE}
check_status_code $?

echo "[CT_ICP] -- [CT_ICP] -- building CMake Project"
cmake --build . --config $BUILD_TYPE --target install --parallel 6
check_status_code $?

echo "[CT_ICP] -- [CT_ICP_SLAM] -- building ros2 package"
cd ${SRC_DIR}/ros2
source /opt/ros/galactic/setup.bash
colcon build --symlink-install \
	--cmake-args \
		-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  	-Dlgmath_DIR=${LGMATH_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/lgmath \
  	-Dsteam_DIR=${STEAM_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/steam \
		-Dct_icp_DIR=${CT_ICP_BUILD_DIR}/install/${BUILD_TYPE}/lib/cmake/ct_icp

