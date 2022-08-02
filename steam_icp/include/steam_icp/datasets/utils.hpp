#pragma once

#include <Eigen/Core>

#include "steam_icp/dataset.hpp"

namespace steam_icp {

Sequence::SeqError evaluateOdometry(const ArrayPoses &poses_gt, const ArrayPoses &poses_estimated);

}  // namespace steam_icp