#pragma once

#include <fstream>

#include "steam_icp/odometry.hpp"

namespace steam_icp {

class SplineOdometry : public Odometry {
 public:
  enum class STEAM_LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options : public Odometry::Options {
    // sensor vehicle transformation
    Eigen::Matrix<double, 4, 4> T_sr = Eigen::Matrix<double, 4, 4>::Identity();
    // trajectory
    double knot_spacing = 0;
    // velocity prior (no side slipping)
    Eigen::Matrix<double, 6, 6> vp_cov = Eigen::Matrix<double, 6, 6>::Identity();
    // radial velocity
    STEAM_LOSS_FUNC rv_loss_func = STEAM_LOSS_FUNC::L2;
    double rv_cov_inv = 1.0;
    double rv_loss_threshold = 1.0;
    // optimization
    bool verbose = false;
    int max_iterations = 1;
    unsigned int num_threads = 1;
  };

  SplineOdometry(const Options &options) : Odometry(options), options_(options) {
    using namespace steam::traj;
    steam_trajectory_ = bspline::Interface::MakeShared(Time(options_.knot_spacing));
  }

  ~SplineOdometry() override {
    velocity_debug_file_.open(options_.debug_path + "/velocity.txt", std::ios::out);
    for (auto &query_time : velocity_query_times_) {
      const auto w_mr_inr = steam_trajectory_->getVelocityInterpolator(query_time)->evaluate();
      velocity_debug_file_ << 0 << " " << query_time.nanosecs() << " " << w_mr_inr.transpose() << std::endl;
    }
    velocity_debug_file_.close();
  }

  RegistrationSummary registerFrame(const std::vector<Point3D> &frame) override;

 private:
  std::vector<Point3D> initializeFrame(int index_frame, const std::vector<Point3D> &const_frame);
  void icp(int index_frame, std::vector<Point3D> &keypoints, RegistrationSummary &summary);

  const Options options_;

  steam::traj::bspline::Interface::Ptr steam_trajectory_;

  std::ofstream velocity_debug_file_;
  std::vector<steam::traj::Time> velocity_query_times_;

  STEAM_ICP_REGISTER_ODOMETRY("SPLINE", SplineOdometry);
};

}  // namespace steam_icp